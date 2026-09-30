//! Live training loop. Runs on a blocking thread and reports through a channel:
//! per-step loss, periodic full-network snapshots (attention, routing,
//! activations, gradient norms), validation loss and text samples.

use crate::events::*;
use crate::inspect::{generate, inspect_model, IB};
use crate::model::{build, Model};
use crate::spec::{analyze, ModelSpec};
use crate::tokenizer::Tokenized;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::Autodiff;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;

type TB = Autodiff<IB>;

pub enum Event {
    Status(StatusPayload),
    Steps(Vec<StepPoint>),
    Eval(EvalPoint),
    Inspect(Box<InspectPayload>),
    Sample(SamplePayload),
}

/// The most recent weights of a run, kept so the user can inspect and generate after (or during) training.
pub struct Trained {
    pub model: Model<IB>,
    pub fingerprint: u64,
    pub seq_len: usize,
    pub step: usize,
    pub tok: Arc<Tokenized>,
    /// Trained on the Q&A dataset (its vocabulary is its own, not the Data tab's).
    pub qa: bool,
}

/// Marks every float parameter as requiring gradients. Weights that come from `from_inner` are
/// plain (untracked) tensors, so without this a continued run computes no gradients at all and
/// the loss never moves.
struct RequireGrad;

impl burn::module::ModuleMapper<TB> for RequireGrad {
    fn map_float<const D: usize>(&mut self, param: burn::module::Param<Tensor<TB, D>>) -> burn::module::Param<Tensor<TB, D>> {
        param.set_require_grad(true)
    }
}

pub type TrainedSlot = Arc<Mutex<Option<Arc<Trained>>>>;

pub struct Job {
    pub spec: ModelSpec,
    pub tok: Arc<Tokenized>,
    pub hyper: Hyper,
    pub probe: Arc<Mutex<String>>,
    pub cancel: Arc<AtomicBool>,
    pub tx: UnboundedSender<Event>,
    pub trained: TrainedSlot,
    /// Weights to keep training from (`None` = fresh random initialisation).
    pub init: Option<Model<IB>>,
    pub qa: bool,
}

fn lr_at(step: usize, total: usize, base: f64, warmup: usize) -> f64 {
    if warmup > 0 && step < warmup {
        return base * (step + 1) as f64 / warmup as f64;
    }
    let progress = (step.saturating_sub(warmup)) as f64 / (total.saturating_sub(warmup)).max(1) as f64;
    let min = base * 0.1;
    min + 0.5 * (base - min) * (1.0 + (std::f64::consts::PI * progress.min(1.0)).cos())
}

fn batch<B2: Backend>(data: &[u16], b: usize, s: usize, rng: &mut StdRng, device: &B2::Device) -> (Tensor<B2, 2, Int>, Tensor<B2, 2, Int>) {
    let (mut x, mut y) = (Vec::with_capacity(b * s), Vec::with_capacity(b * s));
    for _ in 0..b {
        let o = rng.gen_range(0..data.len() - s - 1);
        x.extend(data[o..o + s].iter().map(|&t| t as i64));
        y.extend(data[o + 1..o + s + 1].iter().map(|&t| t as i64));
    }
    (
        Tensor::from_data(TensorData::new(x, [b, s]), device),
        Tensor::from_data(TensorData::new(y, [b, s]), device),
    )
}

fn r4(x: f32) -> f32 {
    if x.is_finite() { (x * 10000.0).round() / 10000.0 } else { 0.0 }
}

pub fn run(job: Job) {
    let Job { spec, tok, hyper: h, probe, cancel, tx, trained, init, qa } = job;
    let status = |state: &'static str, message: String, step: usize| {
        let _ = tx.send(Event::Status(StatusPayload { state, message, step, total: h.steps }));
    };
    status("preparing", "Building the network…".into(), 0);

    let vocab = tok.bpe.vocab_size();
    let analysis = analyze(&spec, vocab, h.seq_len);
    if analysis.has_errors {
        return status("error", analysis.issues.iter().find(|i| i.level == "error").map(|i| i.message.clone()).unwrap_or_default(), 0);
    }
    if tok.ids.len() < h.seq_len + 2 {
        return status("error", format!("The data has only {} tokens; need at least context length + 2 ({}). Add more text or shorten the context.", tok.ids.len(), h.seq_len + 2), 0);
    }

    // Hold out the last 10% for validation when there is enough data.
    let n = tok.ids.len();
    let val_len = n / 10;
    let (train_ids, val_ids) = if val_len >= h.seq_len + 2 { (&tok.ids[..n - val_len], &tok.ids[n - val_len..]) } else { (&tok.ids[..], &tok.ids[..]) };

    let device = NdArrayDevice::default();
    <TB as Backend>::seed(&device, h.seed);
    let mut model: Model<TB> = match init {
        Some(m) => {
            use burn::module::Module;
            Model::<TB>::from_inner(m).map(&mut RequireGrad)
        }
        None => match build::<TB>(&spec, vocab, h.seq_len, &device) {
            Ok(m) => m,
            Err(e) => return status("error", e, 0),
        },
    };
    let mut optim = AdamWConfig::new()
        .with_weight_decay(h.weight_decay)
        .with_grad_clipping((h.grad_clip > 0.0).then(|| GradientClippingConfig::Norm(h.grad_clip)))
        .init::<TB, Model<TB>>();
    let loss_fn = CrossEntropyLoss::<TB>::new(None, &device);
    let mut rng = StdRng::seed_from_u64(h.seed);
    let fingerprint = spec.fingerprint();

    let mut pending: Vec<StepPoint> = Vec::new();
    let mut last_flush = Instant::now();
    let mut norms: Vec<(usize, f32, f32)> = Vec::new();

    let snapshot = |m: &Model<IB>, step: usize, norms: &[(usize, f32, f32)], with_map: bool| {
        let probe_text = probe.lock().unwrap().clone();
        let mut payload = inspect_model(m, &tok, &probe_text, &analysis, h.seq_len, Some(step), with_map);
        for l in payload.layers.iter_mut() {
            if let Some(&(_, g, w)) = norms.iter().find(|n| n.0 == l.index) {
                l.grad_norm = Some(r4(g));
                l.weight_norm = Some(r4(w));
            }
        }
        let _ = tx.send(Event::Inspect(Box::new(payload)));
        *trained.lock().unwrap() = Some(Arc::new(Trained { model: m.clone(), fingerprint, seq_len: h.seq_len, step, tok: tok.clone(), qa }));
    };

    let eval = |m: &Model<IB>, step: usize| {
        let mut rng = StdRng::seed_from_u64(1234);
        let idev = NdArrayDevice::default();
        let loss_fn = CrossEntropyLoss::<IB>::new(None, &idev);
        let mut total = 0.0f32;
        let batches = 4;
        for _ in 0..batches {
            let (x, y) = batch::<IB>(val_ids, h.batch_size.min(16), h.seq_len, &mut rng, &idev);
            let logits = m.forward(x);
            let [b, s, v] = logits.dims();
            total += loss_fn.forward(logits.reshape([b * s, v]), y.reshape([b * s])).into_scalar().elem::<f32>();
        }
        let vl = total / batches as f32;
        let _ = tx.send(Event::Eval(EvalPoint { step, val_loss: r4(vl), perplexity: r4(vl.exp().min(1e6)) }));
    };

    let sample = |m: &Model<IB>, step: usize| {
        if h.sample_tokens == 0 {
            return;
        }
        let prompt = probe.lock().unwrap().clone();
        let text = generate(m, &tok.bpe, &tok.frequent, &prompt, h.sample_tokens, h.temperature, 0, h.seed.wrapping_add(step as u64), |_, _, _| true);
        let _ = tx.send(Event::Sample(SamplePayload { step, prompt, text }));
    };

    status("running", "Training".into(), 0);
    // step 0: show the untrained network so the first curve point has context
    snapshot(&model.valid(), 0, &[], true);

    let mut done_steps = 0;
    for step in 1..=h.steps {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let t0 = Instant::now();
        let lr = lr_at(step - 1, h.steps, h.lr, h.warmup_steps);
        let (x, y) = batch::<TB>(train_ids, h.batch_size, h.seq_len, &mut rng, &device);
        let logits = model.forward(x);
        let [b, s, v] = logits.dims();
        let loss = loss_fn.forward(logits.reshape([b * s, v]), y.reshape([b * s]));
        let loss_v = loss.clone().into_scalar().elem::<f32>();
        if !loss_v.is_finite() {
            status("error", format!("Loss became {loss_v} at step {step}. Try a lower learning rate or enable gradient clipping."), step);
            return;
        }
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let probe_step = step % h.probe_every == 0 || step == h.steps;
        if probe_step {
            norms = model.block_norms(&grads);
        }
        model = optim.step(lr, model, grads);
        done_steps = step;

        let dt = t0.elapsed().as_secs_f32().max(1e-6);
        pending.push(StepPoint { step, loss: r4(loss_v), lr: lr as f32, tok_per_sec: (b * s) as f32 / dt });

        let cadence = probe_step || step % h.eval_every == 0 || step % h.sample_every == 0;
        if last_flush.elapsed() > Duration::from_millis(60) || cadence {
            let _ = tx.send(Event::Steps(std::mem::take(&mut pending)));
            last_flush = Instant::now();
        }
        if cadence {
            let m = model.valid();
            if probe_step {
                snapshot(&m, step, &norms, step % (h.probe_every * 5) == 0 || step == h.steps);
            }
            if step % h.eval_every == 0 || step == h.steps {
                eval(&m, step);
            }
            if step % h.sample_every == 0 || step == h.steps {
                sample(&m, step);
            }
            let _ = tx.send(Event::Status(StatusPayload { state: "running", message: "Training".into(), step, total: h.steps }));
        }
    }

    if !pending.is_empty() {
        let _ = tx.send(Event::Steps(std::mem::take(&mut pending)));
    }
    if cancel.load(Ordering::Relaxed) {
        // keep the weights reached so far for inspect / generate
        let m = model.valid();
        snapshot(&m, done_steps, &norms, true);
        status("stopped", "Stopped — weights kept, you can still generate.".into(), done_steps);
    } else {
        status("done", "Training finished.".into(), done_steps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Hyper;

    fn run_job(spec_json: &str, steps: usize) -> (Vec<Event>, TrainedSlot) {
        let spec: ModelSpec = serde_json::from_str(spec_json).unwrap();
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(80);
        let tok = Arc::new(Tokenized::from_text(&text, 20));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let trained: TrainedSlot = Arc::new(Mutex::new(None));
        let hyper = Hyper { steps, batch_size: 8, seq_len: 32, lr: 5e-3, probe_every: 10, eval_every: 20, sample_every: 20, sample_tokens: 20, ..Default::default() };
        run(Job { spec, tok, hyper, probe: Arc::new(Mutex::new("the quick brown".into())), cancel: Arc::new(AtomicBool::new(false)), tx, trained: trained.clone(), init: None, qa: false });
        let mut evs = Vec::new();
        while let Ok(e) = rx.try_recv() {
            evs.push(e);
        }
        (evs, trained)
    }

    const GEMMA: &str = r#"{"blocks":[
        {"id":"e","type":"embedding","d_model":48},
        {"id":"a","type":"attention","n_heads":4,"n_kv_heads":2},
        {"id":"f","type":"ffn","kind":"glu","d_ff":96},
        {"id":"a2","type":"attention","n_heads":4,"n_kv_heads":2,"window":8,"sink":true},
        {"id":"m","type":"ffn","kind":"moe","d_ff":32,"n_experts":4,"top_k":2},
        {"id":"n","type":"norm","kind":"rms"},
        {"id":"h","type":"head"}]}"#;

    #[test]
    fn loss_falls_and_all_event_kinds_are_emitted() {
        let (evs, trained) = run_job(GEMMA, 60);
        let losses: Vec<f32> = evs.iter().filter_map(|e| if let Event::Steps(s) = e { Some(s.iter().map(|p| p.loss).collect::<Vec<_>>()) } else { None }).flatten().collect();
        assert_eq!(losses.len(), 60);
        let (first, last) = (losses[..5].iter().sum::<f32>() / 5.0, losses[55..].iter().sum::<f32>() / 5.0);
        assert!(last < first * 0.6, "loss did not fall enough: {first} -> {last}");

        let last_state = evs.iter().rev().find_map(|e| if let Event::Status(s) = e { Some(s.state) } else { None });
        assert_eq!(last_state, Some("done"));
        assert!(evs.iter().any(|e| matches!(e, Event::Eval(_))));
        assert!(evs.iter().any(|e| matches!(e, Event::Sample(s) if !s.text.is_empty())));

        let last_inspect = evs.iter().rev().find_map(|e| if let Event::Inspect(p) = e { Some(p) } else { None }).unwrap();
        assert!(last_inspect.trained && last_inspect.step == Some(60));
        assert_eq!(last_inspect.layers.len(), 7);
        assert!(last_inspect.layers[1].attn.is_some() && last_inspect.layers[4].moe.is_some());
        assert!(last_inspect.layers[3].attn.as_ref().unwrap().sink_mass.is_some());
        assert!(last_inspect.layers[1].grad_norm.unwrap_or(0.0) > 0.0, "gradient norms must be reported");
        assert!(!last_inspect.map.is_empty());
        assert!(trained.lock().unwrap().is_some());
    }

    #[test]
    fn continuing_from_trained_weights_starts_at_the_earlier_loss() {
        let (first_evs, first) = run_job(GEMMA, 60);
        let end_of_first = first_evs.iter().rev().find_map(|e| if let Event::Eval(p) = e { Some(p.val_loss) } else { None }).unwrap();
        let t = first.lock().unwrap().clone().unwrap();

        let spec: ModelSpec = serde_json::from_str(GEMMA).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let hyper = Hyper { steps: 10, batch_size: 8, seq_len: 32, lr: 1e-4, warmup_steps: 0, probe_every: 5, eval_every: 5, sample_every: 100, sample_tokens: 0, ..Default::default() };
        run(Job {
            spec, tok: t.tok.clone(), hyper, probe: Arc::new(Mutex::new("the quick".into())), cancel: Arc::new(AtomicBool::new(false)), tx,
            trained: Arc::new(Mutex::new(None)), init: Some(t.model.clone()), qa: true,
        });
        let mut first_loss = None;
        while let Ok(e) = rx.try_recv() {
            if let Event::Steps(p) = e {
                first_loss.get_or_insert(p[0].loss);
            }
        }
        let first_loss = first_loss.unwrap();
        assert!(first_loss < end_of_first * 2.0 + 0.5, "continued run should start near where the last one ended ({end_of_first}), not from scratch: {first_loss}");
    }

    #[test]
    fn continued_run_actually_updates_the_weights() {
        let (_, first) = run_job(GEMMA, 30);
        let t = first.lock().unwrap().clone().unwrap();
        let spec: ModelSpec = serde_json::from_str(GEMMA).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let other = Arc::new(Tokenized::from_parts(t.tok.bpe.clone(), t.tok.bpe.encode(&"pack my box with five dozen liquor jugs. ".repeat(60)).into_iter().map(|i| i as u16).collect(), 10, 10, vec![1; t.tok.bpe.vocab_size()], false));
        let hyper = Hyper { steps: 40, batch_size: 8, seq_len: 32, lr: 5e-3, warmup_steps: 0, probe_every: 10, eval_every: 100, sample_every: 100, sample_tokens: 0, ..Default::default() };
        let cont = Arc::new(Mutex::new(None));
        run(Job { spec, tok: other, hyper, probe: Arc::new(Mutex::new("pack".into())), cancel: Arc::new(AtomicBool::new(false)), tx, trained: cont.clone(), init: Some(t.model.clone()), qa: true });
        let mut losses = vec![];
        let mut grads = vec![];
        while let Ok(e) = rx.try_recv() {
            match e {
                Event::Steps(p) => losses.extend(p.iter().map(|x| x.loss)),
                Event::Inspect(p) => grads.extend(p.layers.iter().filter_map(|l| l.grad_norm)),
                _ => {}
            }
        }
        eprintln!("continue losses first/last: {} -> {}; grad norms: {:?}", losses[0], losses.last().unwrap(), &grads[grads.len().saturating_sub(7)..]);
        assert!(losses.last().unwrap() < &(losses[0] * 0.7), "loss must fall when continuing on new data: {} -> {}", losses[0], losses.last().unwrap());
    }

    #[test]
    fn rejects_bad_specs_and_short_data() {
        let (evs, _) = run_job(r#"{"blocks":[{"id":"e","type":"embedding","d_model":48}]}"#, 5);
        assert!(matches!(evs.last(), Some(Event::Status(s)) if s.state == "error"));
    }
}
