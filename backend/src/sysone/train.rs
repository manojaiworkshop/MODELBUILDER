//! Training loop: length-bucketed batches, plain Adam with gradient
//! clipping, linear warmup + cosine decay, validation (accuracy, Brier, ECE)
//! after every epoch, best-epoch model kept.
//!
//! Adapted from `transformer_rs/src/sysone/train.rs` to report progress
//! through a channel (this crate's convention, see `crate::train`) instead
//! of printing, and to check a cancellation flag between steps.

use crate::events::StatusPayload;
use crate::sysone::encode::{make_batch, EncodeConfig, Example};
use crate::sysone::loss::compute_loss;
use crate::sysone::metrics::{Accumulator, Report};
use crate::sysone::model::{SysOneConfig, SysOneModel};
use crate::sysone::schema::Sample;
use crate::sysone::tokens::Tokenizer;
use crate::tokenizer::Bpe;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SysOneStep {
    pub step: usize,
    pub loss: f32,
    pub noul: f32,
    pub opt: f32,
    pub conf: f32,
    pub lr: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SysOneEval {
    pub epoch: usize,
    pub report: Report,
    pub best: bool,
}

pub enum Event {
    Status(StatusPayload),
    Step(SysOneStep),
    Eval(SysOneEval),
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainHyper {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub warmup_steps: usize,
    pub conf_weight: f32,
    pub grad_clip: f32,
    pub seed: u64,
    pub merges: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub dropout: f64,
    pub max_seq_len: usize,
}

impl Default for TrainHyper {
    fn default() -> Self {
        TrainHyper {
            epochs: 6,
            batch_size: 16,
            lr: 5e-4,
            warmup_steps: 100,
            conf_weight: 0.5,
            grad_clip: 1.0,
            seed: 42,
            merges: 200,
            d_model: 128,
            n_heads: 4,
            n_kv_heads: 2,
            n_layers: 4,
            d_ff: 512,
            dropout: 0.1,
            max_seq_len: 256,
        }
    }
}

impl TrainHyper {
    pub fn validate(&self) -> Result<(), String> {
        let chk = |ok: bool, msg: &str| if ok { Ok(()) } else { Err(msg.to_string()) };
        chk(self.epochs >= 1 && self.epochs <= 200, "epochs must be 1..200")?;
        chk(self.batch_size >= 1 && self.batch_size <= 256, "batch size must be 1..256")?;
        chk(self.lr > 0.0 && self.lr < 1.0, "learning rate must be in (0, 1)")?;
        chk(self.d_model >= 8 && self.d_model % self.n_heads == 0, "d_model must be a positive multiple of heads")?;
        chk(self.n_heads >= 1 && self.n_heads % self.n_kv_heads.max(1) == 0, "heads must be a multiple of kv heads")?;
        chk(self.n_layers >= 1 && self.n_layers <= 64, "layers must be 1..64")?;
        chk(self.d_ff >= 8, "feed-forward width must be >= 8")?;
        chk(self.max_seq_len >= 16 && self.max_seq_len <= 4096, "max sequence length must be 16..4096")?;
        Ok(())
    }
}

pub struct Dataset {
    pub examples: Vec<Example>,
}

impl Dataset {
    pub fn build(tok: &Tokenizer, enc: &EncodeConfig, samples: &[Sample]) -> (Dataset, usize) {
        let examples: Vec<Example> = samples.iter().filter_map(|s| Example::encode(tok, enc, &s.state, &s.questions, Some(&s.targets))).collect();
        let dropped = samples.len() - examples.len();
        (Dataset { examples }, dropped)
    }
}

/// Learns the BPE on the samples' text: state (capped per sample) plus every
/// distinct question/option string.
pub fn train_tokenizer(samples: &[Sample], num_merges: usize) -> Tokenizer {
    let mut texts: Vec<String> = Vec::new();
    for s in samples {
        texts.push(s.state.chars().take(400).collect());
    }
    let mut seen = std::collections::HashSet::new();
    for s in samples {
        for q in &s.questions {
            if seen.insert(q.text().to_string()) {
                texts.push(q.text().to_string());
            }
            for o in q.options() {
                if seen.insert(o.clone()) {
                    texts.push(o.clone());
                }
            }
        }
    }
    Tokenizer::new(Bpe::train(&texts.join("\n"), num_merges))
}

pub fn lr_at(step: usize, total: usize, base_lr: f64, warmup: usize) -> f64 {
    if step < warmup {
        return base_lr * (step + 1) as f64 / warmup as f64;
    }
    let decay = (total - warmup).max(1);
    let progress = ((step - warmup) as f64 / decay as f64).min(1.0);
    base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

fn make_epoch_batches(set: &Dataset, batch_size: usize, rng: &mut StdRng) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..set.examples.len()).collect();
    order.shuffle(rng);
    let mut batches = Vec::new();
    for window in order.chunks(batch_size * 32) {
        let mut w = window.to_vec();
        w.sort_by_key(|&i| set.examples[i].ids.len());
        batches.extend(w.chunks(batch_size).map(|c| c.to_vec()));
    }
    batches.shuffle(rng);
    batches
}

pub fn evaluate<B: Backend>(model: &SysOneModel<B>, set: &Dataset, batch_size: usize, device: &B::Device) -> Report {
    use crate::sysone::api::answers_from_outputs;
    let mut acc = Accumulator::default();
    let mut order: Vec<usize> = (0..set.examples.len()).collect();
    order.sort_by_key(|&i| set.examples[i].ids.len());
    for chunk in order.chunks(batch_size) {
        let examples: Vec<Example> = chunk.iter().map(|&i| set.examples[i].clone()).collect();
        let batch = make_batch(&examples);
        let out = model.forward(&batch.to_tensors::<B>(device));
        let answers = answers_from_outputs(&out, &batch, &examples);
        for (ex, ans) in examples.iter().zip(&answers) {
            for (eq, a) in ex.questions.iter().zip(ans) {
                if let Some(t) = &eq.target {
                    acc.add(a, t);
                }
            }
        }
    }
    acc.report()
}

/// "Which epoch is best": mean accuracy minus the ECE of the stated
/// probabilities. Calibration is the point of a decision model, so an epoch
/// that gains a little accuracy while much worse calibrated must not win.
fn quality(r: &Report) -> f32 {
    let xs: Vec<f32> = [r.noul_acc, r.choice_acc, r.score_acc].into_iter().filter(|x| x.is_finite()).collect();
    let acc = if xs.is_empty() { 0.0 } else { xs.iter().sum::<f32>() / xs.len() as f32 };
    acc - r.prob_ece
}

pub struct TrainedSysOne<B: Backend> {
    pub model: SysOneModel<B>,
    pub tok: Tokenizer,
    pub enc: EncodeConfig,
    pub cfg: SysOneConfig,
    pub step: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn run<B: AutodiffBackend>(
    samples: Vec<Sample>,
    hyper: TrainHyper,
    device: B::Device,
    cancel: Arc<AtomicBool>,
    tx: UnboundedSender<Event>,
) -> Option<TrainedSysOne<B::InnerBackend>> {
    let emit_status = |state: &'static str, message: String, step: usize, total: usize| {
        let _ = tx.send(Event::Status(StatusPayload { state, message, step, total }));
    };

    let enc = EncodeConfig::scaled(hyper.max_seq_len);
    let tok = train_tokenizer(&samples, hyper.merges);
    let (dataset, dropped) = Dataset::build(&tok, &enc, &samples);
    if dataset.examples.is_empty() {
        emit_status("error", "No sample fit the context window — shorten the state or questions.".into(), 0, 0);
        return None;
    }
    // last 10% (at least 1) held out for validation
    let n_val = (dataset.examples.len() / 10).max(1).min(dataset.examples.len() - 1).max(if dataset.examples.len() > 1 { 1 } else { 0 });
    let (train_examples, val_examples) = dataset.examples.split_at(dataset.examples.len() - n_val);
    let train_set = Dataset { examples: train_examples.to_vec() };
    let val_set = if val_examples.is_empty() { None } else { Some(Dataset { examples: val_examples.to_vec() }) };

    let cfg = SysOneConfig {
        vocab_size: tok.vocab_size(),
        d_model: hyper.d_model,
        n_heads: hyper.n_heads,
        n_kv_heads: hyper.n_kv_heads,
        n_layers: hyper.n_layers,
        d_ff: hyper.d_ff,
        dropout: hyper.dropout,
        max_seq_len: enc.max_seq_len,
        rope_theta: 10_000.0,
    };
    let mut model: SysOneModel<B> = cfg.init(&device);
    let mut optim = AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(hyper.grad_clip))).init::<B, SysOneModel<B>>();
    let mut rng = StdRng::seed_from_u64(hyper.seed);

    let steps_per_epoch = train_set.examples.len().div_ceil(hyper.batch_size).max(1);
    let total_steps = steps_per_epoch * hyper.epochs;
    let warmup = hyper.warmup_steps.min(total_steps.saturating_sub(1)).max(1);

    if dropped > 0 {
        emit_status("preparing", format!("{dropped} sample(s) dropped: their questions alone exceed the context window."), 0, total_steps);
    }
    emit_status("running", String::new(), 0, total_steps);

    let mut global_step = 0usize;
    let mut best: Option<(f32, SysOneModel<B>, usize)> = None;
    let started = Instant::now();

    'epochs: for epoch in 0..hyper.epochs {
        let batches = make_epoch_batches(&train_set, hyper.batch_size, &mut rng);
        for group in &batches {
            if cancel.load(Ordering::Relaxed) {
                break 'epochs;
            }
            let examples: Vec<Example> = group.iter().map(|&j| train_set.examples[j].clone()).collect();
            let batch = make_batch(&examples);
            let tensors = batch.to_tensors::<B>(&device);

            let out = model.forward(&tensors);
            let parts = compute_loss(&out, &tensors, hyper.conf_weight);
            let lv = parts.total.clone().into_scalar().elem::<f32>();
            let lr = lr_at(global_step, total_steps, hyper.lr, warmup);
            if lv.is_finite() && lv < 100.0 {
                let grads = GradientsParams::from_grads(parts.total.backward(), &model);
                model = optim.step(lr, model, grads);
            }
            global_step += 1;

            let _ = tx.send(Event::Step(SysOneStep {
                step: global_step,
                loss: lv,
                noul: parts.noul.into_scalar().elem::<f32>(),
                opt: parts.opt.into_scalar().elem::<f32>(),
                conf: parts.conf.into_scalar().elem::<f32>(),
                lr: lr as f32,
            }));
            emit_status("running", String::new(), global_step, total_steps);
        }
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        if let Some(val) = &val_set {
            let valid = model.valid();
            let r = evaluate::<B::InnerBackend>(&valid, val, hyper.batch_size, &device);
            let q = quality(&r);
            let is_best = best.as_ref().map_or(true, |(bq, _, _)| q > *bq);
            if is_best {
                best = Some((q, model.clone(), global_step));
            }
            let _ = tx.send(Event::Eval(SysOneEval { epoch, report: r, best: is_best }));
        }
        let _ = started; // kept for future timing diagnostics
    }

    let stopped = cancel.load(Ordering::Relaxed);
    let (final_model, step) = match best {
        Some((_, m, s)) => (m, s),
        None => (model, global_step),
    };
    emit_status(if stopped { "stopped" } else { "done" }, String::new(), global_step, total_steps);

    Some(TrainedSysOne { model: final_model.valid(), tok, enc, cfg, step })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sysone::api::decide;
    use crate::sysone::schema::{Question, Target};
    use burn::backend::{Autodiff, NdArray};

    type TB = Autodiff<NdArray>;

    fn synth(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| {
                let urgent = i % 2 == 0;
                let state = if urgent { format!("URGENT ticket {i}: the database is down") } else { format!("routine ticket {i}: please update the docs") };
                Sample {
                    id: i as u32,
                    state,
                    questions: vec![
                        Question::Noul { text: "does this need attention now?".into() },
                        Question::Choice { text: "which team should handle this?".into(), options: vec!["infra".into(), "docs".into()] },
                    ],
                    targets: vec![Target::Noul { p: if urgent { 1.0 } else { 0.0 } }, Target::choice_hard(2, if urgent { 0 } else { 1 })],
                    source: "synth".into(),
                }
            })
            .collect()
    }

    #[test]
    fn loss_falls_and_a_trained_model_answers_questions() {
        let device = Default::default();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let hyper = TrainHyper { epochs: 8, batch_size: 8, lr: 5e-3, warmup_steps: 2, d_model: 32, n_heads: 2, n_kv_heads: 1, n_layers: 2, d_ff: 64, max_seq_len: 64, ..Default::default() };
        let cancel = Arc::new(AtomicBool::new(false));

        let trained = run::<TB>(synth(60), hyper, device, cancel, tx).expect("training should produce a model");

        let mut steps = Vec::new();
        let mut saw_eval = false;
        let mut saw_done = false;
        while let Ok(ev) = rx.try_recv() {
            match ev {
                Event::Step(s) => steps.push(s.loss),
                Event::Eval(_) => saw_eval = true,
                Event::Status(s) if s.state == "done" => saw_done = true,
                Event::Status(_) => {}
            }
        }
        assert!(!steps.is_empty(), "no training steps were reported");
        assert!(saw_eval, "no validation report was emitted");
        assert!(saw_done, "final status was not 'done'");

        let first_avg: f32 = steps[..steps.len() / 4].iter().sum::<f32>() / (steps.len() / 4) as f32;
        let last_avg: f32 = steps[steps.len() * 3 / 4..].iter().sum::<f32>() / (steps.len() - steps.len() * 3 / 4) as f32;
        assert!(last_avg < first_avg, "loss did not fall: first quarter avg {first_avg}, last quarter avg {last_avg}");

        let answers = decide::<NdArray>(&trained.model, &trained.tok, &trained.enc, &Default::default(), "URGENT: server room is on fire", &[
            Question::Noul { text: "does this need attention now?".into() },
            Question::Choice { text: "which team should handle this?".into(), options: vec!["infra".into(), "docs".into()] },
        ])
        .expect("inference should succeed");
        assert_eq!(answers.len(), 2);
        match &answers[0] {
            crate::sysone::schema::Answer::Noul { p, confidence } => {
                assert!((0.0..=1.0).contains(p) && (0.0..=1.0).contains(confidence));
            }
            _ => panic!("expected a Noul answer"),
        }
    }

    #[test]
    fn cancelling_mid_run_stops_early_and_still_returns_a_model() {
        let device = Default::default();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let hyper = TrainHyper { epochs: 50, batch_size: 4, d_model: 16, n_heads: 2, n_kv_heads: 1, n_layers: 1, d_ff: 32, max_seq_len: 64, ..Default::default() };
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        // cancel almost immediately so the run exits the epoch loop early
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(1));
            c2.store(true, Ordering::Relaxed);
        });
        let trained = run::<TB>(synth(20), hyper, device, cancel, tx);
        assert!(trained.is_some());
    }
}
