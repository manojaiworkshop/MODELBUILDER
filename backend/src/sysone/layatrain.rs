//! Fine-tuning the pretrained Laya checkpoint on System One samples (see `laya.rs`).
//!
//! Each sample's questions become separate sequences (Laya reads one question at a time). Every
//! epoch re-draws up to `laya_questions` questions per sample, with choice options reshuffled.
//! The loss is the cross-entropy against the sample's target distribution — a proper scoring rule,
//! like the from-scratch model's. After training, a temperature per question type is fitted on the
//! held-out questions so the probabilities are calibrated for *this* dataset.

use crate::events::StatusPayload;
use crate::sysone::laya::{self, Encoded, LayaConfig, LayaModel, LayaTok, TrainedLaya};
use crate::sysone::metrics::Accumulator;
use crate::sysone::schema::{Question, Sample, Target};
use crate::sysone::train::{augment, effective_warmup, lr_at, quality, Event, SysOneEval, SysOneStep, TrainHyper};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

fn target_probs(t: &Target) -> Vec<f32> {
    match t {
        Target::Noul { p } => vec![1.0 - p, *p],
        Target::Choice { probs } | Target::Score { probs } => probs.clone(),
    }
}

struct Item {
    enc: Encoded,
    target: Vec<f32>,
}

/// Up to `per` questions of `s` (random ones), after reshuffling its options.
fn items_of(tok: &LayaTok, s: &Sample, per: usize, max_len: usize, head_max_len: usize, rng: &mut StdRng, shuffle: bool) -> Vec<(Question, Target, Item)> {
    let a = if shuffle { augment(s, rng) } else { s.clone() };
    let mut idx: Vec<usize> = (0..a.questions.len()).collect();
    idx.shuffle(rng);
    idx.truncate(per);
    idx.into_iter()
        .filter_map(|i| {
            let enc = tok.encode(&a.state, &a.questions[i], max_len, head_max_len)?;
            Some((a.questions[i].clone(), a.targets[i].clone(), Item { enc, target: target_probs(&a.targets[i]) }))
        })
        .collect()
}

/// Batches of item indices: shuffled, then bucketed by length within windows so padding stays small.
fn batches(items: &[Item], size: usize, rng: &mut StdRng) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.shuffle(rng);
    let mut out = Vec::new();
    for w in order.chunks(size * 16) {
        let mut w = w.to_vec();
        w.sort_by_key(|&i| items[i].enc.ids.len());
        out.extend(w.chunks(size).map(|c| c.to_vec()));
    }
    out.shuffle(rng);
    out
}

fn logits_of<B: Backend>(model: &LayaModel<B>, tok: &LayaTok, device: &B::Device, items: &[&Item]) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(items.len());
    for chunk in items.chunks(16) {
        let pairs: Vec<(&Encoded, &[f32])> = chunk.iter().map(|i| (&i.enc, i.target.as_slice())).collect();
        let b = laya::make_batch(&pairs, tok.pad);
        let z = model.forward(&b.tensors::<B>(model.window(), device)).into_data().convert::<f32>().to_vec::<f32>().unwrap_or_default();
        for (r, it) in chunk.iter().enumerate() {
            out.push(z[r * b.k..r * b.k + it.enc.markers.len()].to_vec());
        }
    }
    out
}

/// The temperature in 0.3..=4 that minimises cross-entropy on `(logits, target)` pairs.
fn fit_temperature(pairs: &[(&Vec<f32>, &Vec<f32>)]) -> f32 {
    if pairs.len() < 8 {
        return 1.0;
    }
    let nll = |t: f32| -> f32 {
        pairs
            .iter()
            .map(|(z, tg)| {
                let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let lse = z.iter().map(|x| ((x - m) / t).exp()).sum::<f32>().ln();
                -z.iter().zip(tg.iter()).map(|(x, g)| g * ((x - m) / t - lse)).sum::<f32>()
            })
            .sum::<f32>()
    };
    let mut best = (f32::INFINITY, 1.0);
    let mut t = 0.3;
    while t <= 4.0 {
        let v = nll(t);
        if v < best.0 {
            best = (v, t);
        }
        t += 0.05;
    }
    best.1
}

#[allow(clippy::too_many_arguments)]
pub fn run<B: AutodiffBackend>(samples: Vec<Sample>, hyper: TrainHyper, device: B::Device, cancel: Arc<AtomicBool>, tx: UnboundedSender<Event>) -> Option<TrainedLaya<B::InnerBackend>> {
    let status = |state: &'static str, message: String, step: usize, total: usize| {
        let _ = tx.send(Event::Status(StatusPayload { state, message, step, total }));
    };
    let dir = laya::cache_dir();
    if !laya::is_downloaded(&dir) {
        status("error", "Laya hasn't been downloaded yet — import it first.".into(), 0, 0);
        return None;
    }
    status("preparing", "Loading Laya (421M parameters) onto the GPU…".into(), 0, 0);
    let cfg = match LayaConfig::load(&dir) {
        Ok(c) => c,
        Err(e) => {
            status("error", e, 0, 0);
            return None;
        }
    };
    let tok = match LayaTok::load(&dir) {
        Ok(t) => t,
        Err(e) => {
            status("error", e, 0, 0);
            return None;
        }
    };
    let mut model: LayaModel<B> = match LayaModel::load(&dir, &cfg, hyper.laya_layers, &device) {
        Ok(m) => m,
        Err(e) => {
            status("error", e, 0, 0);
            return None;
        }
    };
    let max_len = hyper.laya_max_len;
    let head_max_len = cfg.head_max_len.min(max_len / 2).max(16);

    let mut rng = StdRng::seed_from_u64(hyper.seed);
    let mut samples = samples;
    samples.shuffle(&mut rng);
    let n_val = (samples.len() / 10).max(1).min(samples.len().saturating_sub(1));
    let val_samples = samples.split_off(samples.len() - n_val);
    let train_samples = samples;
    if train_samples.is_empty() {
        status("error", "Need at least 2 samples to train on (one is held out for validation).".into(), 0, 0);
        return None;
    }

    // a fixed validation set: up to 2 questions per held-out sample, at most 400 in all
    let mut val: Vec<(Question, Target, Item)> = Vec::new();
    for s in &val_samples {
        val.extend(items_of(&tok, s, 2, max_len, head_max_len, &mut rng, false));
    }
    val.truncate(400);
    let val_items: Vec<&Item> = val.iter().map(|(_, _, i)| i).collect();

    let per = hyper.laya_questions;
    let items_per_epoch = train_samples.len() * per;
    let steps_per_epoch = items_per_epoch.div_ceil(hyper.batch_size).max(1);
    let total = steps_per_epoch * hyper.epochs;
    let warmup = effective_warmup(hyper.warmup_steps, total);
    let note = format!("Fine-tuning the top {} of {} encoder layers + the decision head; {} tokens per question.", hyper.laya_layers, cfg.layers, max_len);
    status("running", note.clone(), 0, total);

    let mut optim = AdamWConfig::new().with_weight_decay(0.01).with_grad_clipping(Some(GradientClippingConfig::Norm(hyper.grad_clip))).init::<B, LayaModel<B>>();
    let mut best: Option<(f32, LayaModel<B::InnerBackend>, usize)> = None;
    let inner_device = device.clone();
    let mut global = 0usize;

    'epochs: for epoch in 0..hyper.epochs {
        let mut items: Vec<Item> = Vec::with_capacity(items_per_epoch);
        for s in &train_samples {
            items.extend(items_of(&tok, s, per, max_len, head_max_len, &mut rng, true).into_iter().map(|(_, _, i)| i));
        }
        if items.is_empty() {
            status("error", "None of the questions fit in the token budget — raise \"tokens per question\".".into(), 0, 0);
            return None;
        }
        for group in batches(&items, hyper.batch_size, &mut rng) {
            if cancel.load(Ordering::Relaxed) {
                break 'epochs;
            }
            let pairs: Vec<(&Encoded, &[f32])> = group.iter().map(|&i| (&items[i].enc, items[i].target.as_slice())).collect();
            let batch = laya::make_batch(&pairs, tok.pad);
            let t = batch.tensors::<B>(model.window(), &device);
            let logits = model.forward(&t);
            let rows = (t.target.clone() * log_softmax(logits, 1)).sum_dim(1).neg(); // [b, 1]
            let total_loss = rows.clone().mean();
            let lv = total_loss.clone().into_scalar().elem::<f32>();
            let lr = lr_at(global, total, hyper.lr, warmup);
            if lv.is_finite() && lv < 100.0 {
                let grads = GradientsParams::from_grads(total_loss.backward(), &model);
                model = optim.step(lr, model, grads);
            }
            // per-type loss for the live chart (yes/no vs choice/score)
            let per_row = rows.detach().into_data().convert::<f32>().to_vec::<f32>().unwrap_or_default();
            let (mut n_s, mut n_o, mut l_s, mut l_o) = (0f32, 0f32, 0f32, 0f32);
            for (r, &i) in group.iter().enumerate() {
                if items[i].enc.qtype == laya::QTYPE_NOUL {
                    n_s += 1.0;
                    l_s += per_row.get(r).copied().unwrap_or(0.0);
                } else {
                    n_o += 1.0;
                    l_o += per_row.get(r).copied().unwrap_or(0.0);
                }
            }
            global += 1;
            let _ = tx.send(Event::Step(SysOneStep { step: global, loss: lv, noul: if n_s > 0.0 { l_s / n_s } else { 0.0 }, opt: if n_o > 0.0 { l_o / n_o } else { 0.0 }, conf: 0.0, lr: lr as f32 }));
            status("running", note.clone(), global, total);
        }
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        let valid = model.valid();
        let z = logits_of::<B::InnerBackend>(&valid, &tok, &inner_device, &val_items);
        let mut acc = Accumulator::default();
        for ((q, tg, _), z) in val.iter().zip(&z) {
            acc.add(&laya::answer_from_logits(q, z, &[1.0; 3]), tg);
        }
        let r = acc.report();
        let q = quality(&r);
        let is_best = best.as_ref().map_or(true, |(bq, _, _)| q > *bq);
        if is_best {
            best = Some((q, valid, global));
        }
        let _ = tx.send(Event::Eval(SysOneEval { epoch, report: r, best: is_best }));
    }

    let stopped = cancel.load(Ordering::Relaxed);
    let (final_model, step) = match best {
        Some((_, m, s)) => (m, s),
        None => (model.valid(), global),
    };
    // calibrate for this dataset
    let z = logits_of::<B::InnerBackend>(&final_model, &tok, &inner_device, &val_items);
    let mut temperature = [1.0f32; 3];
    for (qt, slot) in temperature.iter_mut().enumerate() {
        let pairs: Vec<(&Vec<f32>, &Vec<f32>)> = z.iter().zip(&val_items).filter(|(_, it)| it.enc.qtype == qt as i64).map(|(z, it)| (z, &it.target)).collect();
        *slot = fit_temperature(&pairs);
    }
    let _ = rng.gen::<u8>();
    status(if stopped { "stopped" } else { "done" }, format!("{note} Fitted temperatures (choice / score / yes-no): {:.2} / {:.2} / {:.2}.", temperature[0], temperature[1], temperature[2]), global, total);
    Some(TrainedLaya { model: final_model, tok, max_len, head_max_len, temperature, step })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::{Autodiff, Wgpu};

    fn env(k: &str, d: usize) -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    }

    fn samples(n: usize) -> Vec<Sample> {
        let topics = [("the server is down and customers are shouting", 0usize), ("please update the quarterly slide deck", 1), ("what time is the lunch break", 2)];
        (0..n)
            .map(|i| {
                let (text, gold) = topics[i % 3];
                Sample {
                    id: i as u32,
                    state: format!("{text} (ticket {i}). {}", "Background: the quarterly review notes mention budgets, staffing and old tickets. ".repeat(env("LAYA_FILL", 0))),
                    questions: vec![
                        Question::Choice { text: "Which team?".into(), options: vec!["ops".into(), "marketing".into(), "hr".into()] },
                        Question::Noul { text: "Is it urgent?".into() },
                    ],
                    targets: vec![Target::choice_hard(3, gold), Target::Noul { p: if gold == 0 { 1.0 } else { 0.0 } }],
                    source: "t".into(),
                    call: None,
                }
            })
            .collect()
    }

    /// Needs the downloaded weights and a GPU: `cargo test --release laya_gpu -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn laya_gpu_finetune_learns() {
        let hyper = TrainHyper { base: "laya".into(), epochs: 3, batch_size: env("LAYA_BATCH", 8), lr: 3e-5, warmup_steps: 3, laya_layers: env("LAYA_LAYERS", 4), laya_max_len: env("LAYA_LEN", 128), laya_questions: 2, ..Default::default() };
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let t0 = std::time::Instant::now();
        let trained = run::<Autodiff<Wgpu>>(samples(env("LAYA_N", 120)), hyper, WgpuDevice::default(), Arc::new(AtomicBool::new(false)), tx).expect("a trained model");
        let (mut losses, mut last_status) = (Vec::new(), String::new());
        while let Ok(ev) = rx.try_recv() {
            match ev {
                Event::Step(s) => losses.push(s.loss),
                Event::Status(s) => last_status = format!("{} {}", s.state, s.message),
                Event::Eval(e) => println!("eval epoch {} choice_acc {:.2} noul_acc {:.2}", e.epoch, e.report.choice_acc, e.report.noul_acc),
            }
        }
        println!("{} steps in {:.1}s; first losses {:?} last {:?}\n{last_status}", losses.len(), t0.elapsed().as_secs_f32(), &losses[..3.min(losses.len())], &losses[losses.len().saturating_sub(3)..]);
        let q = (losses.len() / 4).max(1);
        let (head, tail): (f32, f32) = (losses[..q].iter().sum::<f32>() / q as f32, losses[losses.len() - q..].iter().sum::<f32>() / q as f32);
        assert!(tail < head, "loss didn't go down ({head} -> {tail})");
        let a = trained.decide(&Default::default(), "the server is down and customers are shouting (ticket 999)", &[Question::Choice { text: "Which team?".into(), options: vec!["ops".into(), "marketing".into(), "hr".into()] }]).unwrap();
        println!("{a:?}");
    }
}
