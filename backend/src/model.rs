//! A transformer assembled at runtime from the user's block stack.
//!
//! Every block implementation is adapted from the transformer_rs project:
//! - attention: `slm::model::GemmaAttention` (GQA, per-head QK-RMSNorm, RoPE,
//!   sliding window) plus `gptoss::model::GptOssAttention`'s attention sink;
//! - FFN: `GemmaMlp` (gated GLU) and `GptOssMoeMlp` (top-k mixture of experts,
//!   incl. the `kth_largest` workaround because `topk` isn't differentiable);
//! - embedding: scaled by sqrt(d_model), init std 1/sqrt(d) (from `sysone`),
//!   output head tied to the embedding matrix by default;
//! - positions: burn's sinusoidal `PositionalEncoding` (the ATIS model).
//!
//! Blocks are a Vec of an enum, so any order/number of blocks is one model.
//! Blocks compute only what the UI asked for; `want_trace` also captures
//! attention maps, routing and activations of the *first* batch item.

use crate::spec::{Activation, BlockKind, FfnKind, ModelSpec, NormKind};
use crate::trace::{AttnTrace, Heat, LayerTrace, MoeTrace, Stats, Trace};
use burn::module::{Initializer, Module, ModuleVisitor, Param};
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig, PositionalEncoding, PositionalEncodingConfig, RmsNorm, RmsNormConfig,
    RotaryEncoding, RotaryEncodingConfig,
};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::activation::{gelu_approximate, relu, silu, softmax};
use burn::tensor::backend::AutodiffBackend;

const EPS: f64 = 1e-6;
// Below this, a layer's activation heatmap shows every dim exactly; above it, evenly-sampled
// columns stand in for the rest — a canvas this wide (~stretched over a few hundred CSS px)
// stops letting a human see or hover individual columns well past this width anyway.
const MAX_HEAT_COLS: usize = 128;

fn linear<B: Backend>(i: usize, o: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(i, o).with_bias(false).init(device)
}

fn activate<B: Backend, const D: usize>(a: Activation, x: Tensor<B, D>) -> Tensor<B, D> {
    match a {
        Activation::Gelu => gelu_approximate(x),
        Activation::Silu => silu(x),
        Activation::Relu => relu(x),
    }
}

fn to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().unwrap_or_default()
}

// ---------------------------------------------------------------- norms

#[derive(Module, Debug)]
pub struct NormAny<B: Backend> {
    rms: Option<RmsNorm<B>>,
    layer: Option<LayerNorm<B>>,
}

impl<B: Backend> NormAny<B> {
    fn new(kind: NormKind, d: usize, device: &B::Device) -> Option<Self> {
        match kind {
            NormKind::None => None,
            NormKind::Rms => Some(NormAny { rms: Some(RmsNormConfig::new(d).with_epsilon(EPS).init(device)), layer: None }),
            NormKind::Layer => Some(NormAny { rms: None, layer: Some(LayerNormConfig::new(d).with_epsilon(EPS).init(device)) }),
        }
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match (&self.rms, &self.layer) {
            (Some(n), _) => n.forward(x),
            (_, Some(n)) => n.forward(x),
            _ => x,
        }
    }
}

// ------------------------------------------------------------ attention

#[derive(Module, Debug)]
pub struct Attn<B: Backend> {
    norm: Option<NormAny<B>>,
    wq: Linear<B>,
    wk: Linear<B>,
    wv: Linear<B>,
    wo: Linear<B>,
    q_norm: Option<RmsNorm<B>>,
    k_norm: Option<RmsNorm<B>>,
    rope: Option<RotaryEncoding<B>>,
    /// gpt-oss attention sink: one learned logit per head.
    sink: Option<Param<Tensor<B, 1>>>,
    dropout: Dropout,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    causal: bool,
    residual: bool,
}

/// `[batch, n_kv, seq, hd]` -> `[batch, n_kv * n_rep, seq, hd]` (GQA); from `slm::model::repeat_kv`.
fn repeat_kv<B: Backend>(t: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return t;
    }
    let [b, n_kv, s, hd] = t.dims();
    let e: Tensor<B, 5> = t.reshape([b, n_kv, 1, s, hd]).repeat_dim(2, n_rep);
    e.reshape([b, n_kv * n_rep, s, hd])
}

/// Causal mask limited to the last `window` tokens (true = blocked); from `slm::model::sliding_causal_mask`.
pub fn sliding_causal_mask<B: Backend>(batch: usize, seq: usize, window: usize, device: &B::Device) -> Tensor<B, 3, Bool> {
    let causal = generate_autoregressive_mask::<B>(batch, seq, device);
    let too_old: Tensor<B, 3, Bool> = Tensor::<B, 2, Bool>::triu_mask([seq, seq], -(window as i64) + 1, device)
        .reshape([1, seq, seq])
        .repeat_dim(0, batch);
    (causal.int() + too_old.int()).greater_elem(0)
}

impl<B: Backend> Attn<B> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        d: usize, n_heads: usize, n_kv_heads: usize, qk_norm: bool, rope: bool, theta: f32, window: usize,
        sink: bool, causal: bool, residual: bool, pre_norm: NormKind, dropout: f32, max_seq: usize, device: &B::Device,
    ) -> Self {
        let hd = d / n_heads;
        Attn {
            norm: NormAny::new(pre_norm, d, device),
            wq: linear(d, n_heads * hd, device),
            wk: linear(d, n_kv_heads * hd, device),
            wv: linear(d, n_kv_heads * hd, device),
            wo: linear(n_heads * hd, d, device),
            q_norm: qk_norm.then(|| RmsNormConfig::new(hd).with_epsilon(EPS).init(device)),
            k_norm: qk_norm.then(|| RmsNormConfig::new(hd).with_epsilon(EPS).init(device)),
            rope: rope.then(|| RotaryEncodingConfig::new(max_seq, hd).with_theta(theta).init(device)),
            sink: sink.then(|| Param::from_tensor(Tensor::zeros([n_heads], device))),
            dropout: DropoutConfig::new(dropout as f64).init(),
            n_heads,
            n_kv_heads,
            head_dim: hd,
            window,
            causal,
            residual,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, want: bool) -> (Tensor<B, 3>, Option<AttnTrace>) {
        let [batch, seq, _] = x.dims();
        let device = x.device();
        let h = match &self.norm {
            Some(n) => n.forward(x.clone()),
            None => x.clone(),
        };
        let split = |t: Tensor<B, 3>, heads: usize| -> Tensor<B, 4> { t.reshape([batch, seq, heads, self.head_dim]).swap_dims(1, 2) };

        let mut q = split(self.wq.forward(h.clone()), self.n_heads);
        let mut k = split(self.wk.forward(h.clone()), self.n_kv_heads);
        let v = split(self.wv.forward(h), self.n_kv_heads);
        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            q = qn.forward(q);
            k = kn.forward(k);
        }
        if let Some(rope) = &self.rope {
            q = rope.apply(q, 0);
            k = rope.apply(k, 0);
        }
        let n_rep = self.n_heads / self.n_kv_heads;
        let (k, v) = (repeat_kv(k, n_rep), repeat_kv(v, n_rep));

        let mut scores = q.matmul(k.swap_dims(2, 3)).div_scalar((self.head_dim as f32).sqrt());
        if self.causal {
            let m3 = if self.window > 0 {
                sliding_causal_mask::<B>(batch, seq, self.window, &device)
            } else {
                generate_autoregressive_mask::<B>(batch, seq, &device)
            };
            let mask: Tensor<B, 4, Bool> = m3.reshape([batch, 1, seq, seq]).repeat_dim(1, self.n_heads);
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }

        // Attention sink: an extra never-masked key that only joins the softmax denominator.
        let attn = match &self.sink {
            Some(s) => {
                let sink = s.val().reshape([1, self.n_heads, 1, 1]).expand([batch, self.n_heads, seq, 1]);
                softmax(Tensor::cat(vec![scores, sink], 3), 3).slice([0..batch, 0..self.n_heads, 0..seq, 0..seq])
            }
            None => softmax(scores, 3),
        };

        let trace = want.then(|| {
            let flat = to_vec(attn.clone().slice([0..1, 0..self.n_heads, 0..seq, 0..seq]));
            let per = seq * seq;
            let heads: Vec<Heat> = (0..self.n_heads).map(|h| Heat::new(seq, seq, flat[h * per..(h + 1) * per].to_vec())).collect();
            let sink_mass = self.sink.as_ref().map(|_| {
                (0..self.n_heads)
                    .map(|h| {
                        let m = &flat[h * per..(h + 1) * per];
                        let rows: f32 = (0..seq).map(|r| 1.0 - m[r * seq..(r + 1) * seq].iter().sum::<f32>()).sum();
                        crate::trace::r3(rows / seq as f32)
                    })
                    .collect()
            });
            AttnTrace { heads, sink_mass }
        });

        let attn = self.dropout.forward(attn);
        let out = attn.matmul(v).swap_dims(1, 2).reshape([batch, seq, self.n_heads * self.head_dim]);
        let out = self.wo.forward(out);
        (if self.residual { x + out } else { out }, trace)
    }
}

// ------------------------------------------------------------------ FFN

/// Gated MLP; port of `slm::model::GemmaMlp` with a selectable activation.
#[derive(Module, Debug)]
pub struct Glu<B: Backend> {
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
    dropout: Dropout,
    #[module(skip)]
    act: Activation,
}

impl<B: Backend> Glu<B> {
    fn new(d: usize, d_ff: usize, act: Activation, dropout: f32, device: &B::Device) -> Self {
        Glu { gate: linear(d, d_ff, device), up: linear(d, d_ff, device), down: linear(d_ff, d, device), dropout: DropoutConfig::new(dropout as f64).init(), act }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let g = activate(self.act, self.gate.forward(x.clone())) * self.up.forward(x);
        self.dropout.forward(self.down.forward(g))
    }
}

#[derive(Module, Debug)]
pub struct PlainMlp<B: Backend> {
    up: Linear<B>,
    down: Linear<B>,
    dropout: Dropout,
    #[module(skip)]
    act: Activation,
}

impl<B: Backend> PlainMlp<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.dropout.forward(self.down.forward(activate(self.act, self.up.forward(x))))
    }
}

/// Mixture of experts; port of `gptoss::model::GptOssMoeMlp`.
#[derive(Module, Debug)]
pub struct Moe<B: Backend> {
    router: Linear<B>,
    experts: Vec<Glu<B>>,
    top_k: usize,
}

/// k-th largest value along `dim` (keepdim). `Tensor::topk` panics under autodiff,
/// so repeatedly take the max and mask it out (from `gptoss::model::kth_largest`).
fn kth_largest<B: Backend, const D: usize>(t: Tensor<B, D>, k: usize, dim: usize) -> Tensor<B, D> {
    let mut current = t;
    let mut max_val = current.clone().max_dim(dim);
    for _ in 1..k {
        let is_max = current.clone().equal(max_val.expand(current.shape()));
        current = current.mask_fill(is_max, f32::NEG_INFINITY);
        max_val = current.clone().max_dim(dim);
    }
    max_val
}

impl<B: Backend> Moe<B> {
    /// Returns the output and the `[batch, seq, n_experts]` routing weights.
    fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, seq, d] = x.dims();
        let n = self.experts.len();
        let probs = softmax(self.router.forward(x.clone()), 2);
        let threshold = kth_largest(probs.clone(), self.top_k, 2);
        let keep = probs.clone().greater_equal(threshold.expand([batch, seq, n]));
        let gate = probs.mask_fill(keep.bool_not(), 0.0f32);
        let gate = gate.clone() / gate.sum_dim(2);
        let mut out = Tensor::<B, 3>::zeros([batch, seq, d], &x.device());
        for (e, expert) in self.experts.iter().enumerate() {
            let w = gate.clone().slice([0..batch, 0..seq, e..e + 1]);
            out = out + expert.forward(x.clone()) * w;
        }
        (out, gate)
    }
}

#[derive(Module, Debug)]
pub enum FfnInner<B: Backend> {
    Glu(Glu<B>),
    Mlp(PlainMlp<B>),
    Moe(Moe<B>),
}

#[derive(Module, Debug)]
pub struct Ffn<B: Backend> {
    norm: Option<NormAny<B>>,
    inner: FfnInner<B>,
    residual: bool,
}

impl<B: Backend> Ffn<B> {
    fn forward(&self, x: Tensor<B, 3>, want: bool) -> (Tensor<B, 3>, Option<MoeTrace>) {
        let h = match &self.norm {
            Some(n) => n.forward(x.clone()),
            None => x.clone(),
        };
        let (out, trace) = match &self.inner {
            FfnInner::Glu(m) => (m.forward(h), None),
            FfnInner::Mlp(m) => (m.forward(h), None),
            FfnInner::Moe(m) => {
                let (o, gate) = m.forward(h);
                let t = want.then(|| {
                    let [_, seq, n] = gate.dims();
                    let flat = to_vec(gate.slice([0..1, 0..seq, 0..n]));
                    let usage = (0..n).map(|e| crate::trace::r3((0..seq).map(|r| flat[r * n + e]).sum::<f32>() / seq as f32)).collect();
                    MoeTrace { n_experts: n, gates: Heat::new(seq, n, flat), usage }
                });
                (o, t)
            }
        };
        (if self.residual { x + out } else { out }, trace)
    }
}

// --------------------------------------------------------- layer + model

#[derive(Module, Debug)]
pub enum Layer<B: Backend> {
    Pos(PositionalEncoding<B>),
    Norm(NormAny<B>),
    Attn(Attn<B>),
    Ffn(Ffn<B>),
    Drop(Dropout),
}

/// A block's runtime module plus the index it has in the user's stack.
#[derive(Module, Debug)]
pub struct Placed<B: Backend> {
    layer: Layer<B>,
    index: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    embed: Embedding<B>,
    layers: Vec<Placed<B>>,
    head: Option<Linear<B>>,
    d_model: usize,
    embed_scale: bool,
    pub max_seq_len: usize,
    n_blocks: usize,
}

impl<B: Backend> Layer<B> {
    fn kind_name(&self) -> &'static str {
        match self {
            Layer::Pos(_) => "pos_encoding",
            Layer::Norm(_) => "norm",
            Layer::Attn(_) => "attention",
            Layer::Ffn(_) => "ffn",
            Layer::Drop(_) => "dropout",
        }
    }
}

/// Builds the network described by `spec`. The spec must already have passed `spec::analyze`.
pub fn build<B: Backend>(spec: &ModelSpec, vocab: usize, max_seq_len: usize, device: &B::Device) -> Result<Model<B>, String> {
    let d = spec.d_model().ok_or("model has no Embedding block")?;
    let mut embed = None;
    let mut embed_scale = true;
    let mut layers = Vec::new();
    let mut head: Option<Linear<B>> = None;
    for (index, b) in spec.blocks.iter().enumerate() {
        let layer = match &b.kind {
            BlockKind::Embedding { d_model, scale } => {
                embed_scale = *scale;
                let mut cfg = EmbeddingConfig::new(vocab, *d_model);
                if *scale {
                    // std 1/sqrt(d): after the sqrt(d) input scaling the embeddings have unit scale (from sysone).
                    cfg = cfg.with_initializer(Initializer::Normal { mean: 0.0, std: 1.0 / (*d_model as f64).sqrt() });
                }
                embed = Some(cfg.init(device));
                continue;
            }
            BlockKind::Head { tied } => {
                if !*tied {
                    head = Some(linear(d, vocab, device));
                }
                continue;
            }
            BlockKind::PosEncoding => Layer::Pos(PositionalEncodingConfig::new(d).with_max_sequence_size(max_seq_len).init(device)),
            BlockKind::Norm { kind } => Layer::Norm(NormAny::new(*kind, d, device).ok_or("a Norm block needs a norm kind")?),
            BlockKind::Dropout { p } => Layer::Drop(DropoutConfig::new(*p as f64).init()),
            BlockKind::Attention { n_heads, n_kv_heads, qk_norm, rope, rope_theta, window, sink, causal, residual, pre_norm, dropout } => Layer::Attn(Attn::new(
                d, *n_heads, *n_kv_heads, *qk_norm, *rope, *rope_theta, *window, *sink, *causal, *residual, *pre_norm, *dropout, max_seq_len, device,
            )),
            BlockKind::Ffn { kind, d_ff, activation, n_experts, top_k, residual, pre_norm, dropout } => {
                let inner = match kind {
                    FfnKind::Glu => FfnInner::Glu(Glu::new(d, *d_ff, *activation, *dropout, device)),
                    FfnKind::Mlp => FfnInner::Mlp(PlainMlp {
                        up: linear(d, *d_ff, device),
                        down: linear(*d_ff, d, device),
                        dropout: DropoutConfig::new(*dropout as f64).init(),
                        act: *activation,
                    }),
                    FfnKind::Moe => FfnInner::Moe(Moe {
                        router: linear(d, *n_experts, device),
                        experts: (0..*n_experts).map(|_| Glu::new(d, *d_ff, *activation, *dropout, device)).collect(),
                        top_k: *top_k,
                    }),
                };
                Layer::Ffn(Ffn { norm: NormAny::new(*pre_norm, d, device), inner, residual: *residual })
            }
        };
        layers.push(Placed { layer, index });
    }
    Ok(Model { embed: embed.ok_or("model has no Embedding block")?, layers, head, d_model: d, embed_scale, max_seq_len, n_blocks: spec.blocks.len() })
}

impl<B: Backend> Model<B> {
    pub fn vocab_size(&self) -> usize {
        self.embed.weight.dims()[0]
    }

    /// Logits `[batch, seq, vocab]`. No tracing (training / generation path).
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.run(ids, false).0
    }

    /// Forward pass that, when `want_trace`, also snapshots every block for batch item 0.
    pub fn run(&self, ids: Tensor<B, 2, Int>, want_trace: bool) -> (Tensor<B, 3>, Option<Trace>) {
        let [batch, seq] = ids.dims();
        let d = self.d_model;
        let mut traces: Vec<LayerTrace> = Vec::new();

        let mut x = self.embed.forward(ids);
        if self.embed_scale {
            x = x * (d as f32).sqrt();
        }
        let snap = |x: &Tensor<B, 3>, index: usize, kind: &'static str, params: usize, attn: Option<AttnTrace>, moe: Option<MoeTrace>| -> LayerTrace {
            let flat = to_vec(x.clone().slice([0..1, 0..seq, 0..d]));
            LayerTrace {
                index, kind, out_shape: [seq, d], params,
                stats: Stats::of(&flat),
                act: Heat::downsample_cols(&flat, seq, d, MAX_HEAT_COLS),
                attn, moe, grad_norm: None, weight_norm: None,
            }
        };
        if want_trace {
            traces.push(snap(&x, 0, "embedding", self.embed.num_params(), None, None));
        }

        for p in &self.layers {
            let (out, attn, moe) = match &p.layer {
                Layer::Pos(m) => (m.forward(x), None, None),
                Layer::Norm(m) => (m.forward(x), None, None),
                Layer::Drop(m) => (m.forward(x), None, None),
                Layer::Attn(m) => {
                    let (o, t) = m.forward(x, want_trace);
                    (o, t, None)
                }
                Layer::Ffn(m) => {
                    let (o, t) = m.forward(x, want_trace);
                    (o, None, t)
                }
            };
            x = out;
            if want_trace {
                traces.push(snap(&x, p.index, p.layer.kind_name(), p.layer_params(), attn, moe));
            }
        }

        let logits = match &self.head {
            Some(h) => h.forward(x),
            None => {
                let w = self.embed.weight.val().swap_dims(0, 1); // [d, vocab]
                let v = w.dims()[1];
                x.reshape([batch * seq, d]).matmul(w).reshape([batch, seq, v])
            }
        };

        let trace = want_trace.then(|| {
            let v = logits.dims()[2];
            let first = to_vec(logits.clone().slice([0..1, 0..seq, 0..v]));
            traces.push(LayerTrace {
                index: self.n_blocks - 1,
                kind: "head",
                out_shape: [seq, v],
                params: self.head.as_ref().map(|h| h.num_params()).unwrap_or(0),
                stats: Stats::of(&first),
                act: Heat::default(),
                attn: None, moe: None, grad_norm: None, weight_norm: None,
            });
            Trace { layers: traces, last_logits: first[(seq - 1) * v..].to_vec(), probe_loss: None }
        });
        (logits, trace)
    }

    /// Embedding matrix as `(row-major data, vocab, d_model)`.
    pub fn embedding_matrix(&self) -> (Vec<f32>, usize, usize) {
        let w = self.embed.weight.val();
        let [v, d] = w.dims();
        (to_vec(w), v, d)
    }
}

impl<B: Backend> Placed<B> {
    fn layer_params(&self) -> usize {
        self.layer.num_params()
    }
}

// ------------------------------------------------- gradient / weight norms

struct NormVisitor<'a> {
    grads: &'a GradientsParams,
    g2: f64,
    w2: f64,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for NormVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(g) = self.grads.get::<B::InnerBackend, D>(param.id) {
            self.g2 += g.powf_scalar(2.0).sum().into_scalar().elem::<f64>();
        }
        self.w2 += param.val().inner().powf_scalar(2.0).sum().into_scalar().elem::<f64>();
    }
}

impl<B: AutodiffBackend> Model<B> {
    /// `(block index, grad L2 norm, weight L2 norm)` per block that owns parameters.
    /// Must be called before the optimizer consumes `grads`.
    pub fn block_norms(&self, grads: &GradientsParams) -> Vec<(usize, f32, f32)> {
        let mut out = Vec::new();
        let mut v = NormVisitor { grads, g2: 0.0, w2: 0.0 };
        self.embed.visit(&mut v);
        out.push((0, v.g2.sqrt() as f32, v.w2.sqrt() as f32));
        for p in &self.layers {
            let mut v = NormVisitor { grads, g2: 0.0, w2: 0.0 };
            p.layer.visit(&mut v);
            if v.w2 > 0.0 {
                out.push((p.index, v.g2.sqrt() as f32, v.w2.sqrt() as f32));
            }
        }
        if let Some(h) = &self.head {
            let mut v = NormVisitor { grads, g2: 0.0, w2: 0.0 };
            h.visit(&mut v);
            out.push((self.n_blocks - 1, v.g2.sqrt() as f32, v.w2.sqrt() as f32));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::analyze;
    use burn::backend::NdArray;

    type B = NdArray;

    fn spec(json: &str) -> ModelSpec {
        serde_json::from_str(json).unwrap()
    }

    fn ids(b: usize, s: usize, vocab: usize) -> Tensor<B, 2, Int> {
        let data: Vec<i64> = (0..b * s).map(|i| (i * 7 % vocab) as i64).collect();
        Tensor::from_data(TensorData::new(data, [b, s]), &Default::default())
    }

    #[test]
    fn analytic_param_count_matches_burn() {
        let s = spec(
            r#"{"blocks":[
            {"id":"e","type":"embedding","d_model":32},
            {"id":"p","type":"pos_encoding"},
            {"id":"a","type":"attention","n_heads":4,"n_kv_heads":2,"sink":true},
            {"id":"f","type":"ffn","kind":"glu","d_ff":48},
            {"id":"a2","type":"attention","n_heads":4,"n_kv_heads":4,"qk_norm":false,"pre_norm":"layer","window":4},
            {"id":"m","type":"ffn","kind":"moe","d_ff":16,"n_experts":4,"top_k":2},
            {"id":"g","type":"ffn","kind":"mlp","d_ff":16,"pre_norm":"none"},
            {"id":"n","type":"norm","kind":"rms"},
            {"id":"h","type":"head","tied":false}]}"#,
        );
        let a = analyze(&s, 300, 16);
        assert!(!a.has_errors, "{:?}", a.issues);
        let m = build::<B>(&s, 300, 16, &Default::default()).unwrap();
        // sinusoidal positions have no learnable params in burn
        use burn::module::Module;
        assert_eq!(m.num_params(), a.total_params);
    }

    #[test]
    fn forward_shapes_and_trace_for_every_block_type() {
        let s = spec(
            r#"{"blocks":[
            {"id":"e","type":"embedding","d_model":32},
            {"id":"p","type":"pos_encoding"},
            {"id":"a","type":"attention","n_heads":4,"n_kv_heads":1,"sink":true,"window":3},
            {"id":"m","type":"ffn","kind":"moe","d_ff":16,"n_experts":4,"top_k":2},
            {"id":"d","type":"dropout","p":0.1},
            {"id":"h","type":"head"}]}"#,
        );
        let m = build::<B>(&s, 300, 16, &Default::default()).unwrap();
        let (logits, tr) = m.run(ids(2, 10, 300), true);
        assert_eq!(logits.dims(), [2, 10, 300]);
        let tr = tr.unwrap();
        assert_eq!(tr.layers.len(), 6);
        let attn = tr.layers[2].attn.as_ref().unwrap();
        assert_eq!(attn.heads.len(), 4);
        // rows are distributions over (window-limited) real keys + the sink
        for h in &attn.heads {
            for r in 0..10 {
                let row = &h.data[r * 10..(r + 1) * 10];
                assert!(row.iter().sum::<f32>() <= 1.001);
                for (c, &p) in row.iter().enumerate() {
                    if c > r || r - c >= 3 {
                        assert_eq!(p, 0.0, "row {r} col {c} must be masked (causal + window 3)");
                    }
                }
            }
        }
        let moe = tr.layers[3].moe.as_ref().unwrap();
        for r in 0..10 {
            let row = &moe.gates.data[r * 4..(r + 1) * 4];
            assert_eq!(row.iter().filter(|&&g| g > 0.0).count(), 2, "top-2 routing");
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn causality_no_future_leak() {
        let s = spec(
            r#"{"blocks":[{"id":"e","type":"embedding","d_model":32},
            {"id":"a","type":"attention","n_heads":4,"n_kv_heads":2},
            {"id":"f","type":"ffn","kind":"glu","d_ff":32},
            {"id":"h","type":"head"}]}"#,
        );
        let m = build::<B>(&s, 300, 16, &Default::default()).unwrap();
        let a = ids(1, 8, 300);
        let mut v: Vec<i64> = a.clone().into_data().convert::<i64>().to_vec::<i64>().unwrap();
        v[7] = 5; // change only the LAST token
        let b: Tensor<B, 2, Int> = Tensor::from_data(TensorData::new(v, [1, 8]), &Default::default());
        let la = to_vec(m.forward(a).slice([0..1, 0..7, 0..300]));
        let lb = to_vec(m.forward(b).slice([0..1, 0..7, 0..300]));
        let diff = la.iter().zip(&lb).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(diff < 1e-5, "earlier positions changed when only a later token did: {diff}");
    }
}
