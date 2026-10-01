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
use crate::sysone::schema::{Question, Sample, Target};
use crate::sysone::tokens::Tokenizer;
use crate::tokenizer::Bpe;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
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
    /// "cpu" (all cores) or "gpu" (Vulkan); the trained weights always end up on the CPU for serving.
    #[serde(default)]
    pub device: String,
}

impl Default for TrainHyper {
    fn default() -> Self {
        TrainHyper {
            epochs: 30,
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
            device: "cpu".into(),
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
        chk(matches!(self.device.as_str(), "" | "cpu" | "gpu"), "device must be \"cpu\" or \"gpu\"")?;
        Ok(())
    }
}

pub struct Dataset {
    pub examples: Vec<Example>,
}

/// Largest context the trainer will grow to by itself (attention cost grows with its square).
pub const MAX_AUTO_SEQ_LEN: usize = 1024;
const SEQ_STEPS: [usize; 5] = [256, 384, 512, 768, 1024];
/// How much of a sample's state we want to keep room for when sizing the context.
const WANT_STATE_TOKENS: usize = 64;

/// The smallest context length (never below `requested`) in which at least 98% of the samples fit
/// their questions plus a useful part of their state. Long samples (tool-calling ones carry tool
/// names as options and many argument questions) would otherwise be dropped from training.
pub fn fit_max_seq_len(tok: &Tokenizer, requested: usize, samples: &[Sample]) -> usize {
    if requested >= MAX_AUTO_SEQ_LEN || samples.is_empty() {
        return requested;
    }
    // sizes depend on the budgets, which depend on the length: check each candidate
    let mut candidates: Vec<usize> = std::iter::once(requested).chain(SEQ_STEPS.iter().copied().filter(|&l| l > requested)).collect();
    candidates.dedup();
    let state_cost: Vec<usize> = samples.iter().map(|s| tok.bpe.encode(&s.state.chars().take(400).collect::<String>()).len().min(WANT_STATE_TOKENS)).collect();
    for &l in &candidates {
        let cfg = EncodeConfig::scaled(l);
        let fits = samples
            .iter()
            .zip(&state_cost)
            .filter(|(s, &st)| Example::encode(tok, &cfg, "", &s.questions, None).is_some_and(|e| e.ids.len() + st <= l))
            .count();
        if fits * 100 >= samples.len() * 98 {
            return l;
        }
    }
    MAX_AUTO_SEQ_LEN
}

/// A copy of `s` with its questions in a random order and, for every Choice question, its options
/// in a random order (targets follow). Score levels are ordered, so they stay as they are. Used
/// freshly every epoch: with only a few samples per tool, a model otherwise learns *where* the
/// right option sits in each sample rather than which option the request calls for.
pub fn augment(s: &Sample, rng: &mut StdRng) -> Sample {
    let mut order: Vec<usize> = (0..s.questions.len()).collect();
    order.shuffle(rng);
    let (mut questions, mut targets) = (Vec::with_capacity(order.len()), Vec::with_capacity(order.len()));
    for &i in &order {
        match (&s.questions[i], &s.targets[i]) {
            (Question::Choice { text, options }, Target::Choice { probs }) => {
                let mut perm: Vec<usize> = (0..options.len()).collect();
                perm.shuffle(rng);
                questions.push(Question::Choice { text: text.clone(), options: perm.iter().map(|&k| options[k].clone()).collect() });
                targets.push(Target::Choice { probs: perm.iter().map(|&k| probs[k]).collect() });
            }
            (q, t) => {
                questions.push(q.clone());
                targets.push(t.clone());
            }
        }
    }
    Sample { questions, targets, ..s.clone() }
}

impl Dataset {
    /// Encodes every sample. One that doesn't fit has its trailing questions dropped one at a time
    /// until it does (it is only dropped entirely if even one question can't fit).
    /// Returns the dataset, the samples as actually encoded (trimmed ones shortened), the number
    /// of samples dropped, and the number that were trimmed. `dataset.examples[i]` encodes `kept[i]`.
    pub fn build(tok: &Tokenizer, enc: &EncodeConfig, samples: &[Sample]) -> (Dataset, Vec<Sample>, usize, usize) {
        let (mut trimmed, mut dropped) = (0, 0);
        let mut examples = Vec::with_capacity(samples.len());
        let mut kept = Vec::with_capacity(samples.len());
        for s in samples {
            let mut keep = s.questions.len();
            let found = loop {
                if keep == 0 {
                    break None;
                }
                if let Some(e) = Example::encode(tok, enc, &s.state, &s.questions[..keep], Some(&s.targets[..keep])) {
                    break Some(e);
                }
                keep -= 1;
            };
            match found {
                Some(e) => {
                    trimmed += (keep < s.questions.len()) as usize;
                    examples.push(e);
                    kept.push(Sample { questions: s.questions[..keep].to_vec(), targets: s.targets[..keep].to_vec(), ..s.clone() });
                }
                None => dropped += 1,
            }
        }
        (Dataset { examples }, kept, dropped, trimmed)
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

/// Warmup steps actually used: at most 10% of the run, so a short run still reaches its full
/// learning rate (a fixed 100 steps would make an 18-step run all warmup and barely train).
pub fn effective_warmup(requested: usize, total_steps: usize) -> usize {
    requested.min((total_steps / 10).max(1)).max(1)
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

    let tok = train_tokenizer(&samples, hyper.merges);
    // long samples (tool calls) need a longer context: grow it to fit rather than drop them
    let max_seq_len = fit_max_seq_len(&tok, hyper.max_seq_len, &samples);
    let enc = EncodeConfig::scaled(max_seq_len);
    let (dataset, kept, dropped, trimmed) = Dataset::build(&tok, &enc, &samples);
    if dataset.examples.is_empty() {
        emit_status("error", "No sample fit the context window — shorten the state or questions.".into(), 0, 0);
        return None;
    }
    // Shuffle before holding anything out: samples arrive grouped (by tool, by page), so the
    // "last 10%" would otherwise be a few whole groups the model never trained on.
    let mut pairs: Vec<(Sample, Example)> = kept.into_iter().zip(dataset.examples).collect();
    pairs.shuffle(&mut StdRng::seed_from_u64(hyper.seed ^ 0x5eed));
    // last 10% (at least 1) held out for validation
    let n_val = (pairs.len() / 10).max(1).min(pairs.len() - 1).max(if pairs.len() > 1 { 1 } else { 0 });
    let val_pairs = pairs.split_off(pairs.len() - n_val);
    let (train_samples, train_examples): (Vec<Sample>, Vec<Example>) = pairs.into_iter().unzip();
    let train_set = Dataset { examples: train_examples };
    let val_set = if val_pairs.is_empty() { None } else { Some(Dataset { examples: val_pairs.into_iter().map(|(_, e)| e).collect() }) };

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
    // AdamW (decoupled weight decay) so the weights don't just grow to memorise a small dataset
    let mut optim = AdamWConfig::new().with_weight_decay(0.01).with_grad_clipping(Some(GradientClippingConfig::Norm(hyper.grad_clip))).init::<B, SysOneModel<B>>();
    let mut rng = StdRng::seed_from_u64(hyper.seed);

    let steps_per_epoch = train_set.examples.len().div_ceil(hyper.batch_size).max(1);
    let total_steps = steps_per_epoch * hyper.epochs;
    let warmup = effective_warmup(hyper.warmup_steps, total_steps);

    // kept for the whole run (and the final status), so it isn't lost when the status updates
    let mut notes: Vec<String> = Vec::new();
    if max_seq_len != hyper.max_seq_len {
        notes.push(format!("Context length raised from {} to {max_seq_len} so the samples fit.", hyper.max_seq_len));
    }
    if trimmed > 0 {
        notes.push(format!("{trimmed} sample(s) were still too long and lost their last question(s)."));
    }
    if dropped > 0 {
        notes.push(format!("{dropped} of {} sample(s) were too long even for {max_seq_len} tokens and left out.", samples.len()));
    }
    let dropped_note = notes.join(" ");
    if !dropped_note.is_empty() {
        emit_status("preparing", dropped_note.clone(), 0, total_steps);
    }
    emit_status("running", dropped_note.clone(), 0, total_steps);

    let mut global_step = 0usize;
    let mut best: Option<(f32, SysOneModel<B>, usize)> = None;
    let started = Instant::now();

    let mut aug_rng = StdRng::seed_from_u64(hyper.seed ^ 0xa09);
    'epochs: for epoch in 0..hyper.epochs {
        // a fresh view of every training sample each epoch (questions and options reshuffled)
        let epoch_set = Dataset {
            examples: train_samples.iter().filter_map(|s| { let a = augment(s, &mut aug_rng); Example::encode(&tok, &enc, &a.state, &a.questions, Some(&a.targets)) }).collect(),
        };
        let train_set = if epoch_set.examples.len() == train_set.examples.len() { &epoch_set } else { &train_set };
        let batches = make_epoch_batches(train_set, hyper.batch_size, &mut rng);
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
            emit_status("running", dropped_note.clone(), global_step, total_steps);
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
    emit_status(if stopped { "stopped" } else { "done" }, dropped_note, global_step, total_steps);

    Some(TrainedSysOne { model: final_model.valid(), tok, enc, cfg, step })
}

/// Copy a trained model onto the CPU backend (the server keeps and serves it there) by
/// round-tripping its weights through an in-memory record.
fn to_cpu<B: Backend>(t: TrainedSysOne<B>) -> TrainedSysOne<burn::backend::NdArray> {
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    type Cpu = burn::backend::NdArray;
    let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = rec.record(t.model.into_record(), ()).expect("serialising the trained weights");
    let cpu_dev = Default::default();
    let record = rec.load::<<SysOneModel<Cpu> as Module<Cpu>>::Record>(bytes, &cpu_dev).expect("loading the weights on the CPU");
    let model = t.cfg.init::<Cpu>(&cpu_dev).load_record(record);
    TrainedSysOne { model, tok: t.tok, enc: t.enc, cfg: t.cfg, step: t.step }
}

/// Train on the device the user picked: the CPU (all cores) or the GPU (Vulkan). A GPU that
/// can't be opened falls back to the CPU with a note rather than failing the run.
pub fn run_selected(
    samples: Vec<Sample>,
    hyper: TrainHyper,
    cancel: Arc<AtomicBool>,
    tx: UnboundedSender<Event>,
) -> Option<TrainedSysOne<burn::backend::NdArray>> {
    use burn::backend::{wgpu::WgpuDevice, Autodiff, NdArray, Wgpu};
    if hyper.device == "gpu" {
        let (s2, h2, c2, t2) = (samples.clone(), hyper.clone(), cancel.clone(), tx.clone());
        let tried = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || run::<Autodiff<Wgpu>>(s2, h2, WgpuDevice::default(), c2, t2).map(to_cpu)));
        match tried {
            Ok(r) => return r,
            Err(_) => {
                let _ = tx.send(Event::Status(StatusPayload { state: "preparing", message: "The GPU could not be used (no Vulkan device, or an operation it doesn't support) — training on the CPU instead.".into(), step: 0, total: 0 }));
            }
        }
    }
    run::<Autodiff<NdArray>>(samples, hyper, Default::default(), cancel, tx)
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
                    call: None,
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

    /// Samples shaped like tool-calling ones: many questions, long option labels.
    fn long_samples(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| {
                let opts: Vec<String> = (0..4).map(|k| format!("fused-track__view_fused_track_data_option_number_{k}")).collect();
                let questions: Vec<Question> = (0..5).map(|q| Question::Choice { text: format!("which value for argument number {q}?"), options: opts.clone() }).collect();
                let targets = (0..5).map(|q| Target::choice_hard(4, (i + q) % 4)).collect();
                Sample { id: i as u32, state: format!("please show me the latest rows number {i}"), questions, targets, source: "t".into(), call: None }
            })
            .collect()
    }

    #[test]
    fn long_samples_grow_the_context_instead_of_being_dropped() {
        let samples = long_samples(30);
        let tok = train_tokenizer(&samples, 50);
        // at 256 these do not fit (each is ~5 questions x 4 long options)...
        let (_, _, dropped_small, trimmed_small) = Dataset::build(&tok, &EncodeConfig::scaled(256), &samples);
        assert!(dropped_small + trimmed_small > 0, "the fixture must be too long for 256 to prove anything");
        // ...so the trainer picks a longer context in which everything fits untouched
        let l = fit_max_seq_len(&tok, 256, &samples);
        assert!(l > 256 && l <= MAX_AUTO_SEQ_LEN, "{l}");
        let (ds, _, dropped, trimmed) = Dataset::build(&tok, &EncodeConfig::scaled(l), &samples);
        assert_eq!((dropped, trimmed, ds.examples.len()), (0, 0, 30));
        // short samples keep the context the user asked for
        assert_eq!(fit_max_seq_len(&tok, 256, &synth(10)), 256);
        assert_eq!(fit_max_seq_len(&tok, 2048, &samples), 2048);
    }

    #[test]
    fn a_sample_that_still_does_not_fit_is_trimmed_not_dropped() {
        let samples = long_samples(5);
        let tok = train_tokenizer(&samples, 50);
        let (ds, kept, dropped, trimmed) = Dataset::build(&tok, &EncodeConfig::scaled(160), &samples);
        assert_eq!(kept.len(), ds.examples.len());
        assert_eq!(dropped, 0);
        assert!(trimmed > 0);
        assert_eq!(ds.examples.len(), 5);
        assert!(ds.examples.iter().all(|e| e.ids.len() <= 160 && !e.questions.is_empty()));
    }

    #[test]
    fn augmenting_reorders_but_keeps_every_answer_correct() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(7);
        let s = Sample {
            id: 1,
            state: "s".into(),
            questions: vec![
                Question::Noul { text: "a?".into() },
                Question::Choice { text: "pick".into(), options: vec!["w".into(), "x".into(), "y".into(), "z".into()] },
                Question::Score { text: "how much".into(), levels: vec!["low".into(), "mid".into(), "high".into()] },
            ],
            targets: vec![Target::Noul { p: 1.0 }, Target::choice_hard(4, 2), Target::Score { probs: vec![0.0, 1.0, 0.0] }],
            source: "t".into(),
            call: None,
        };
        let mut moved = false;
        for _ in 0..20 {
            let a = augment(&s, &mut rng);
            a.validate().unwrap();
            assert_eq!(a.questions.len(), 3);
            // the gold option of the choice is still "y", wherever it ended up
            let (q, t) = a.questions.iter().zip(&a.targets).find(|(q, _)| matches!(q, Question::Choice { .. })).unwrap();
            let Target::Choice { probs } = t else { panic!() };
            assert_eq!(q.options()[probs.iter().position(|p| *p == 1.0).unwrap()], "y");
            // score levels keep their order, and their target
            let (q, t) = a.questions.iter().zip(&a.targets).find(|(q, _)| matches!(q, Question::Score { .. })).unwrap();
            assert_eq!(q.options(), ["low", "mid", "high"]);
            assert!(matches!(t, Target::Score { probs } if probs[1] == 1.0));
            moved |= a.questions[0].text() != "a?" || a.questions[1].options()[0] != "w";
        }
        assert!(moved, "augmentation never changed anything");
    }

    #[test]
    fn warmup_is_a_small_part_of_a_short_run() {
        assert_eq!(effective_warmup(100, 18), 1);
        assert_eq!(effective_warmup(100, 500), 50);
        assert_eq!(effective_warmup(100, 5000), 100);
        assert!(lr_at(5, 18, 1e-3, effective_warmup(100, 18)) > 0.9e-3 * 0.4, "a short run must reach a useful learning rate quickly");
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

    /// `cargo test --release bench_devices -- --ignored --nocapture`: same run on CPU and GPU.
    #[test]
    #[ignore]
    fn bench_devices() {
        for device in ["cpu", "gpu"] {
            let hyper = TrainHyper { epochs: 10, batch_size: 16, d_model: 128, n_heads: 4, n_kv_heads: 2, n_layers: 4, d_ff: 512, max_seq_len: 256, device: device.into(), ..Default::default() };
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            let t0 = Instant::now();
            let trained = run_selected(synth(320), hyper, Arc::new(AtomicBool::new(false)), tx).expect("model");
            let mut n = 0;
            let mut last = 0.0;
            while let Ok(ev) = rx.try_recv() {
                match ev { Event::Step(s) => { n += 1; last = s.loss; } Event::Status(s) if s.state == "preparing" => println!("{}", s.message), _ => {} }
            }
            println!("{device}: {n} steps in {:.1}s ({:.2} steps/s), last loss {last:.3}, step {}", t0.elapsed().as_secs_f32(), n as f32 / t0.elapsed().as_secs_f32(), trained.step);
        }
    }
}
