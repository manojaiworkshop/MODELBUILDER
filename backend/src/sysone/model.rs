//! System One model: a bidirectional Transformer encoder with typed
//! decision heads.
//!
//! Same building blocks as the rest of this crate's causal model (RMSNorm
//! pre-norm, GQA, per-head QK-norm, RoPE, GeGLU MLP) but with **bidirectional**
//! attention driven by the structured mask in `mask.rs`, and no LM head /
//! weight tying: outputs are read off marker-token hidden states.
//!
//! Heads (all small MLPs):
//! - `noul_head`: question-marker state -> 1 logit (P(true) = sigmoid)
//! - `opt_head` : [option-marker state ; question-marker state] -> 1 logit
//!                per option; softmax over that question's options. Serves
//!                both Choice and Score (Score just reads the options as
//!                ordered levels).
//! - `conf_head`: question-marker state -> 1 logit; trained to predict
//!                whether the (detached) top decision is correct.
//!
//! RoPE positions are the linear sequence index, so a question's position
//! depends on how much text precedes it; training shuffles question order
//! so answers do not rely on absolute position.
//!
//! Ported from `transformer_rs/src/sysone/model.rs`.

use crate::sysone::encode::BatchTensors;
use crate::sysone::mask::structured_mask;
use burn::module::Initializer;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding, RotaryEncodingConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu_approximate, softmax};

const EPS: f64 = 1e-6;

#[derive(Config, Debug)]
pub struct SysOneConfig {
    pub vocab_size: usize,
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 2)]
    pub n_kv_heads: usize,
    #[config(default = 6)]
    pub n_layers: usize,
    #[config(default = 1024)]
    pub d_ff: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Must be >= the longest packed sequence.
    #[config(default = 512)]
    pub max_seq_len: usize,
    #[config(default = 10_000.0)]
    pub rope_theta: f32,
}

fn no_bias<B: Backend>(i: usize, o: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(i, o).with_bias(false).init(device)
}

/// `[batch, n_kv, seq, hd]` -> `[batch, n_kv * n_rep, seq, hd]` (GQA).
fn repeat_kv<B: Backend>(t: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return t;
    }
    let [b, n_kv, s, hd] = t.dims();
    let e: Tensor<B, 5> = t.reshape([b, n_kv, 1, s, hd]).repeat_dim(2, n_rep);
    e.reshape([b, n_kv * n_rep, s, hd])
}

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    fn new(d: usize, d_ff: usize, dropout: f64, device: &B::Device) -> Self {
        Mlp { gate: no_bias(d, d_ff, device), up: no_bias(d, d_ff, device), down: no_bias(d_ff, d, device), dropout: DropoutConfig::new(dropout).init() }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let g = gelu_approximate(self.gate.forward(x.clone())) * self.up.forward(x);
        self.dropout.forward(self.down.forward(g))
    }
}

#[derive(Module, Debug)]
struct Attention<B: Backend> {
    wq: Linear<B>,
    wk: Linear<B>,
    wv: Linear<B>,
    wo: Linear<B>,
    q_norm: RmsNorm<B>,
    k_norm: RmsNorm<B>,
    dropout: Dropout,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    fn new(cfg: &SysOneConfig, device: &B::Device) -> Self {
        assert!(cfg.d_model % cfg.n_heads == 0 && cfg.n_heads % cfg.n_kv_heads == 0);
        let head_dim = cfg.d_model / cfg.n_heads;
        let norm = || RmsNormConfig::new(head_dim).with_epsilon(EPS).init(device);
        Attention {
            wq: no_bias(cfg.d_model, cfg.n_heads * head_dim, device),
            wk: no_bias(cfg.d_model, cfg.n_kv_heads * head_dim, device),
            wv: no_bias(cfg.d_model, cfg.n_kv_heads * head_dim, device),
            wo: no_bias(cfg.n_heads * head_dim, cfg.d_model, device),
            q_norm: norm(),
            k_norm: norm(),
            dropout: DropoutConfig::new(cfg.dropout).init(),
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, rope: &RotaryEncoding<B>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let split = |t: Tensor<B, 3>, heads: usize| -> Tensor<B, 4> { t.reshape([batch, seq, heads, self.head_dim]).swap_dims(1, 2) };
        let q = self.q_norm.forward(split(self.wq.forward(x.clone()), self.n_heads));
        let k = self.k_norm.forward(split(self.wk.forward(x.clone()), self.n_kv_heads));
        let v = split(self.wv.forward(x), self.n_kv_heads);

        let q = rope.apply(q, 0);
        let k = rope.apply(k, 0);

        let n_rep = self.n_heads / self.n_kv_heads;
        let (k, v) = (repeat_kv(k, n_rep), repeat_kv(v, n_rep));

        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar((self.head_dim as f32).sqrt());
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        let attn = self.dropout.forward(softmax(scores, 3));
        let out = attn.matmul(v).swap_dims(1, 2).reshape([batch, seq, self.n_heads * self.head_dim]);
        self.wo.forward(out)
    }
}

#[derive(Module, Debug)]
struct Block<B: Backend> {
    attn_norm: RmsNorm<B>,
    attn: Attention<B>,
    ffn_norm: RmsNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    fn new(cfg: &SysOneConfig, device: &B::Device) -> Self {
        let norm = || RmsNormConfig::new(cfg.d_model).with_epsilon(EPS).init(device);
        Block { attn_norm: norm(), attn: Attention::new(cfg, device), ffn_norm: norm(), mlp: Mlp::new(cfg.d_model, cfg.d_ff, cfg.dropout, device) }
    }

    fn forward(&self, x: Tensor<B, 3>, rope: &RotaryEncoding<B>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_norm.forward(x), rope, mask);
        x.clone() + self.mlp.forward(self.ffn_norm.forward(x))
    }
}

/// Two-layer MLP head, works on any tensor rank (Linear acts on the last dim).
#[derive(Module, Debug)]
struct Head<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend> Head<B> {
    fn new(d_in: usize, d_hidden: usize, device: &B::Device) -> Self {
        Head { l1: LinearConfig::new(d_in, d_hidden).init(device), l2: LinearConfig::new(d_hidden, 1).init(device) }
    }
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.l2.forward(gelu_approximate(self.l1.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct SysOneModel<B: Backend> {
    embed: Embedding<B>,
    rope: RotaryEncoding<B>,
    blocks: Vec<Block<B>>,
    final_norm: RmsNorm<B>,
    noul_head: Head<B>,
    opt_head: Head<B>,
    conf_head: Head<B>,
    d_model: usize,
    n_heads: usize,
}

impl SysOneConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SysOneModel<B> {
        let head_dim = self.d_model / self.n_heads;
        SysOneModel {
            // std 1/sqrt(d): after the sqrt(d) input scaling the embeddings have unit scale;
            // burn's default (std 1) makes the input embedding dominate the residual stream, so
            // the encoder barely contextualizes tokens.
            embed: EmbeddingConfig::new(self.vocab_size, self.d_model)
                .with_initializer(Initializer::Normal { mean: 0.0, std: 1.0 / (self.d_model as f64).sqrt() })
                .init(device),
            rope: RotaryEncodingConfig::new(self.max_seq_len, head_dim).with_theta(self.rope_theta).init(device),
            blocks: (0..self.n_layers).map(|_| Block::new(self, device)).collect(),
            final_norm: RmsNormConfig::new(self.d_model).with_epsilon(EPS).init(device),
            noul_head: Head::new(self.d_model, self.d_model, device),
            opt_head: Head::new(2 * self.d_model, self.d_model, device),
            conf_head: Head::new(self.d_model, self.d_model, device),
            d_model: self.d_model,
            n_heads: self.n_heads,
        }
    }
}

/// Raw (pre-sigmoid / pre-softmax) outputs. `opt_logits` already has invalid
/// option slots pushed to a large negative value, so a plain softmax over
/// the last dim is the per-question distribution.
pub struct Outputs<B: Backend> {
    pub noul_logit: Tensor<B, 2>, // [B, Q]
    pub opt_logits: Tensor<B, 3>, // [B, Q, O]
    pub conf_logit: Tensor<B, 2>, // [B, Q]
}

pub const INVALID_LOGIT: f32 = -1e9;

impl<B: Backend> SysOneModel<B> {
    pub fn forward(&self, t: &BatchTensors<B>) -> Outputs<B> {
        let [b, q, o] = t.o_valid.dims();
        let d = self.d_model;

        let mask = structured_mask::<B>(t.seg.clone(), t.pad.clone(), self.n_heads);
        let mut x = self.embed.forward(t.ids.clone()) * (self.d_model as f32).sqrt();
        for block in &self.blocks {
            x = block.forward(x, &self.rope, mask.clone());
        }
        let hidden = self.final_norm.forward(x);

        let q_h = t.q_sel.clone().matmul(hidden.clone()); // [B, Q, D]
        let o_h = t.o_sel.clone().matmul(hidden).reshape([b, q, o, d]);
        let q_exp = q_h.clone().reshape([b, q, 1, d]).expand([b, q, o, d]);

        let opt_in = Tensor::cat(vec![o_h, q_exp], 3); // [B, Q, O, 2D]
        let opt_raw = self.opt_head.forward(opt_in).reshape([b, q, o]);
        let invalid = t.o_valid.clone().mul_scalar(-1.0).add_scalar(1.0).mul_scalar(INVALID_LOGIT);
        let opt_logits = opt_raw + invalid;

        Outputs {
            noul_logit: self.noul_head.forward(q_h.clone()).reshape([b, q]),
            opt_logits,
            conf_logit: self.conf_head.forward(q_h).reshape([b, q]),
        }
    }
}
