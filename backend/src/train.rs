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
use burn::tensor::backend::AutodiffBackend;
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
    pub spec: ModelSpec,
    pub hyper: crate::events::Hyper,
    pub provenance: Provenance,
    /// Latest training / validation loss, updated as the run goes on (shared by every snapshot).
    pub metrics: Arc<Mutex<Metrics>>,
    pub fingerprint: u64,
    pub seq_len: usize,
    pub step: usize,
    pub tok: Arc<Tokenized>,
    /// Trained on the Q&A or summary dataset (its vocabulary is its own, built from that
    /// generated text, not the Data tab's).
    pub own_vocab: bool,
}

/// What a model was trained on (for the exported model card and chat template).
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    /// "text", "qa" or "summary".
    pub source: String,
    /// Styles of the pairs (empty for plain text).
    pub styles: Vec<crate::hfdata::Style>,
    /// Chat system prompt(s) used, if any.
    pub system: String,
    /// Pairs / summaries, or characters of text.
    pub examples: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Metrics {
    pub train_loss: Option<f32>,
    pub val_loss: Option<f32>,
}

/// Marks every float parameter as requiring gradients. Weights that come from `from_inner` are
/// plain (untracked) tensors, so without this a continued run computes no gradients at all and
/// the loss never moves.
struct RequireGrad;

impl<B: AutodiffBackend> burn::module::ModuleMapper<B> for RequireGrad {
    fn map_float<const D: usize>(&mut self, param: burn::module::Param<Tensor<B, D>>) -> burn::module::Param<Tensor<B, D>> {
        param.set_require_grad(true)
    }
}

pub type TrainedSlot = Arc<Mutex<Option<Arc<Trained>>>>;

#[derive(Clone)]
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
    pub own_vocab: bool,
    pub provenance: Provenance,
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

/// Copies a model's weights onto another backend (e.g. GPU -> CPU) by round-tripping them through
/// an in-memory record into a freshly built network of the same architecture.
fn move_model<B1: Backend, B2: Backend>(m: Model<B1>, spec: &ModelSpec, device: &B2::Device) -> Model<B2> {
    use burn::module::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    let (vocab, seq) = (m.vocab_size(), m.max_seq_len);
    let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = rec.record(m.into_record(), ()).expect("serialising the model weights");
    let record = rec.load::<<Model<B2> as Module<B2>>::Record>(bytes, device).expect("loading the model weights");
    build::<B2>(spec, vocab, seq, device).expect("rebuilding a model that was already built once").load_record(record)
}

/// Trains on the device the user picked. "gpu" tries CUDA first (works on NVIDIA cards, including
/// MIG slices that Vulkan can't see), then Vulkan on a real GPU adapter, and finally falls back to
/// the CPU with a note rather than failing the run.
pub fn run(mut job: Job) {
    if job.hyper.device == "gpu" {
        use burn::backend::{cuda::CudaDevice, wgpu::WgpuDevice, Cuda, Wgpu};
        if try_gpu::<Cuda>(&job, CudaDevice::default(), " on the GPU (CUDA)") {
            return;
        }
        // Not WgpuDevice::default(): with no usable GPU that quietly picks a software Vulkan
        // driver (lavapipe), which is far slower than the CPU backend.
        if try_gpu::<Wgpu>(&job, WgpuDevice::DiscreteGpu(0), " on the GPU (Vulkan)") {
            return;
        }
        let _ = job.tx.send(Event::Status(StatusPayload {
            state: "preparing",
            message: "No usable GPU (CUDA and Vulkan both failed to start) — training on the CPU instead.".into(),
            step: 0,
            total: job.hyper.steps,
        }));
        job.hyper.device = "cpu".into();
    }
    run_on::<Autodiff<IB>>(job, NdArrayDevice::default(), "", |m| m, |m| m)
}

/// Runs the job on GPU backend `G`. Returns false (having trained nothing) when the device can't
/// be opened, or if the run panics, so the caller can fall back.
fn try_gpu<G: Backend>(job: &Job, device: G::Device, label: &'static str) -> bool {
    let probe = device.clone();
    let opened = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || Tensor::<G, 1>::from_floats([1.0, 2.0], &probe).sum().into_scalar().elem::<f32>() == 3.0));
    if !matches!(opened, Ok(true)) {
        return false;
    }
    let job = job.clone();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let spec = job.spec.clone();
        let (gpu, cpu) = (device.clone(), NdArrayDevice::default());
        run_on::<Autodiff<G>>(job, device, label, |m| move_model::<IB, G>(m, &spec, &gpu), |m| move_model::<G, IB>(m, &spec, &cpu))
    }))
    .is_ok()
}

/// The training loop on backend `B`. `to_device` moves continued-from weights (kept on the CPU)
/// onto `B`; `to_cpu` brings weights back for inspection, sampling and the trained-model slot.
fn run_on<B: AutodiffBackend>(
    job: Job,
    device: B::Device,
    on: &'static str,
    to_device: impl Fn(Model<IB>) -> Model<B::InnerBackend>,
    to_cpu: impl Fn(Model<B::InnerBackend>) -> Model<IB>,
) {
    let Job { spec, tok, hyper: h, probe, cancel, tx, trained, init, own_vocab, provenance } = job;
    let status = |state: &'static str, message: String, step: usize| {
        let _ = tx.send(Event::Status(StatusPayload { state, message, step, total: h.steps }));
    };
    status("preparing", format!("Building the network{on}…"), 0);

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

    B::seed(&device, h.seed);
    let mut model: Model<B> = match init {
        Some(m) => {
            use burn::module::Module;
            Model::<B>::from_inner(to_device(m)).map(&mut RequireGrad)
        }
        None => match build::<B>(&spec, vocab, h.seq_len, &device) {
            Ok(m) => m,
            Err(e) => return status("error", e, 0),
        },
    };
    let mut optim = AdamWConfig::new()
        .with_weight_decay(h.weight_decay)
        .with_grad_clipping((h.grad_clip > 0.0).then(|| GradientClippingConfig::Norm(h.grad_clip)))
        .init::<B, Model<B>>();
    let loss_fn = CrossEntropyLoss::<B>::new(None, &device);
    let mut rng = StdRng::seed_from_u64(h.seed);
    let fingerprint = spec.fingerprint();

    let metrics = Arc::new(Mutex::new(Metrics::default()));
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
        *trained.lock().unwrap() = Some(Arc::new(Trained { model: m.clone(), spec: spec.clone(), hyper: h.clone(), provenance: provenance.clone(), metrics: metrics.clone(), fingerprint, seq_len: h.seq_len, step, tok: tok.clone(), own_vocab }));
    };

    // Validation runs on the training device; only snapshots and samples need the CPU copy.
    let eval = |m: &Model<B::InnerBackend>, step: usize| {
        let mut rng = StdRng::seed_from_u64(1234);
        let loss_fn = CrossEntropyLoss::<B::InnerBackend>::new(None, &device);
        let mut total = 0.0f32;
        let batches = 4;
        for _ in 0..batches {
            let (x, y) = batch::<B::InnerBackend>(val_ids, h.batch_size.min(16), h.seq_len, &mut rng, &device);
            let logits = m.forward(x);
            let [b, s, v] = logits.dims();
            total += loss_fn.forward(logits.reshape([b * s, v]), y.reshape([b * s])).into_scalar().elem::<f32>();
        }
        let vl = total / batches as f32;
        metrics.lock().unwrap().val_loss = Some(r4(vl));
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

    status("running", format!("Training{on}"), 0);
    // step 0: show the untrained network so the first curve point has context
    snapshot(&to_cpu(model.valid()), 0, &[], true);

    let mut done_steps = 0;
    for step in 1..=h.steps {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let t0 = Instant::now();
        let lr = lr_at(step - 1, h.steps, h.lr, h.warmup_steps);
        let (x, y) = batch::<B>(train_ids, h.batch_size, h.seq_len, &mut rng, &device);
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
        metrics.lock().unwrap().train_loss = Some(r4(loss_v));

        let dt = t0.elapsed().as_secs_f32().max(1e-6);
        pending.push(StepPoint { step, loss: r4(loss_v), lr: lr as f32, tok_per_sec: (b * s) as f32 / dt });

        let is_eval = step % h.eval_every == 0 || step == h.steps;
        let is_sample = step % h.sample_every == 0 || step == h.steps;
        let cadence = probe_step || is_eval || is_sample;
        if last_flush.elapsed() > Duration::from_millis(60) || cadence {
            let _ = tx.send(Event::Steps(std::mem::take(&mut pending)));
            last_flush = Instant::now();
        }
        if cadence {
            let m = model.valid();
            if is_eval {
                eval(&m, step);
            }
            if probe_step || is_sample {
                let cpu = to_cpu(m);
                if probe_step {
                    snapshot(&cpu, step, &norms, step % (h.probe_every * 5) == 0 || step == h.steps);
                }
                if is_sample {
                    sample(&cpu, step);
                }
            }
            let _ = tx.send(Event::Status(StatusPayload { state: "running", message: format!("Training{on}"), step, total: h.steps }));
        }
    }

    if !pending.is_empty() {
        let _ = tx.send(Event::Steps(std::mem::take(&mut pending)));
    }
    if cancel.load(Ordering::Relaxed) {
        // keep the weights reached so far for inspect / generate
        snapshot(&to_cpu(model.valid()), done_steps, &norms, true);
        status("stopped", "Stopped — weights kept, you can still generate.".into(), done_steps);
    } else {
        status("done", "Training finished.".into(), done_steps);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::events::Hyper;

    pub(crate) fn run_job(spec_json: &str, steps: usize) -> (Vec<Event>, TrainedSlot) {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(80);
        run_job_tok(spec_json, steps, Arc::new(Tokenized::from_text(&text, 20)))
    }

    pub(crate) fn run_job_tok(spec_json: &str, steps: usize, tok: Arc<Tokenized>) -> (Vec<Event>, TrainedSlot) {
        let spec: ModelSpec = serde_json::from_str(spec_json).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let trained: TrainedSlot = Arc::new(Mutex::new(None));
        let hyper = Hyper { steps, batch_size: 8, seq_len: 32, lr: 5e-3, probe_every: 10, eval_every: 20, sample_every: 20, sample_tokens: 20, ..Default::default() };
        run(Job { spec, tok, hyper, probe: Arc::new(Mutex::new("the quick brown".into())), cancel: Arc::new(AtomicBool::new(false)), tx, trained: trained.clone(), init: None, own_vocab: false, provenance: Provenance::default() });
        let mut evs = Vec::new();
        while let Ok(e) = rx.try_recv() {
            evs.push(e);
        }
        (evs, trained)
    }

    pub(crate) const GEMMA: &str = r#"{"blocks":[
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
            trained: Arc::new(Mutex::new(None)), init: Some(t.model.clone()), own_vocab: true, provenance: Provenance::default(),
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
        run(Job { spec, tok: other, hyper, probe: Arc::new(Mutex::new("pack".into())), cancel: Arc::new(AtomicBool::new(false)), tx, trained: cont.clone(), init: Some(t.model.clone()), own_vocab: true, provenance: Provenance::default() });
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
    fn moving_a_model_between_backends_keeps_its_outputs() {
        let (_, slot) = run_job(GEMMA, 20);
        let t = slot.lock().unwrap().clone().unwrap();
        let moved = move_model::<IB, IB>(t.model.clone(), &t.spec, &NdArrayDevice::default());
        let ids = || Tensor::<IB, 2, Int>::from_data(TensorData::new((0..16).map(|i| (i * 3 % 20) as i64).collect::<Vec<_>>(), [1, 16]), &NdArrayDevice::default());
        let (a, b) = (t.model.forward(ids()).into_data().to_vec::<f32>().unwrap(), moved.forward(ids()).into_data().to_vec::<f32>().unwrap());
        let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(diff < 1e-6, "moved model must compute the same logits: {diff}");
    }

    /// Needs a Vulkan GPU: `cargo test --release gpu_training -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_training_runs_on_the_gpu_and_loss_falls() {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(80);
        let spec: ModelSpec = serde_json::from_str(GEMMA).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let trained: TrainedSlot = Arc::new(Mutex::new(None));
        let hyper = Hyper { steps: 60, batch_size: 8, seq_len: 32, lr: 5e-3, probe_every: 10, eval_every: 20, sample_every: 20, sample_tokens: 20, device: "gpu".into(), ..Default::default() };
        let t0 = std::time::Instant::now();
        run(Job { spec, tok: Arc::new(Tokenized::from_text(&text, 20)), hyper, probe: Arc::new(Mutex::new("the quick brown".into())), cancel: Arc::new(AtomicBool::new(false)), tx, trained: trained.clone(), init: None, own_vocab: false, provenance: Provenance::default() });
        let (mut losses, mut msgs) = (vec![], vec![]);
        while let Ok(e) = rx.try_recv() {
            match e {
                Event::Steps(p) => losses.extend(p.iter().map(|x| x.loss)),
                Event::Status(s) => msgs.push(format!("{}: {}", s.state, s.message)),
                _ => {}
            }
        }
        eprintln!("gpu run took {:?}; statuses: {:?}", t0.elapsed(), msgs);
        assert!(!msgs.iter().any(|m| m.contains("No usable GPU")), "fell back to the CPU: {msgs:?}");
        assert!(msgs.iter().any(|m| m.contains("on the GPU")), "{msgs:?}");
        assert!(msgs.last().unwrap().starts_with("done"), "{msgs:?}");
        assert!(losses.last().unwrap() < &(losses[0] * 0.6), "loss did not fall: {} -> {}", losses[0], losses.last().unwrap());
        assert!(trained.lock().unwrap().is_some(), "the CPU copy must be kept for generation");
    }

    #[test]
    fn rejects_bad_specs_and_short_data() {
        let (evs, _) = run_job(r#"{"blocks":[{"id":"e","type":"embedding","d_model":48}]}"#, 5);
        assert!(matches!(evs.last(), Some(Event::Status(s)) if s.state == "error"));
    }
}
