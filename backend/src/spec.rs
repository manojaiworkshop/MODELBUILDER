//! The model description the UI sends: an ordered stack of blocks, plus the
//! validation / shape-inference that gives the user live feedback while they
//! drag blocks around. Nothing here needs a tensor library.

use serde::{Deserialize, Serialize};

pub const MAX_BLOCKS: usize = 40;
pub const MAX_D_MODEL: usize = 512;
pub const MAX_PARAMS: usize = 25_000_000;

fn d_true() -> bool {
    true
}
fn d_theta() -> f32 {
    10_000.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    None,
    Rms,
    Layer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Gelu,
    Silu,
    Relu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnKind {
    /// Gated MLP: `down(act(gate(x)) * up(x))` (GeGLU / SwiGLU) — Gemma's FFN.
    Glu,
    /// Classic two-layer MLP: `down(act(up(x)))`.
    Mlp,
    /// Router picks `top_k` of `n_experts` GLU experts per token — gpt-oss's FFN.
    Moe,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockKind {
    /// Token ids -> vectors (optionally scaled by sqrt(d_model), as Gemma does).
    Embedding {
        d_model: usize,
        #[serde(default = "d_true")]
        scale: bool,
    },
    /// Fixed sinusoidal positions added to the residual stream (the ATIS model's choice).
    PosEncoding,
    Norm { kind: NormKind },
    /// Multi-head self-attention with every knob from the Gemma / gpt-oss models.
    Attention {
        n_heads: usize,
        n_kv_heads: usize,
        #[serde(default = "d_true")]
        qk_norm: bool,
        #[serde(default = "d_true")]
        rope: bool,
        #[serde(default = "d_theta")]
        rope_theta: f32,
        /// 0 = attend to the whole past; N = only the last N tokens (sliding window).
        #[serde(default)]
        window: usize,
        /// gpt-oss attention sink: a learned "attend to nothing" logit per head.
        #[serde(default)]
        sink: bool,
        #[serde(default = "d_true")]
        causal: bool,
        #[serde(default = "d_true")]
        residual: bool,
        #[serde(default = "default_norm")]
        pre_norm: NormKind,
        #[serde(default)]
        dropout: f32,
    },
    Ffn {
        kind: FfnKind,
        d_ff: usize,
        #[serde(default = "default_act")]
        activation: Activation,
        #[serde(default = "default_experts")]
        n_experts: usize,
        #[serde(default = "default_top_k")]
        top_k: usize,
        #[serde(default = "d_true")]
        residual: bool,
        #[serde(default = "default_norm")]
        pre_norm: NormKind,
        #[serde(default)]
        dropout: f32,
    },
    Dropout { p: f32 },
    /// Final projection to vocabulary logits. `tied` reuses the embedding matrix.
    Head {
        #[serde(default = "d_true")]
        tied: bool,
    },
}

fn default_norm() -> NormKind {
    NormKind::Rms
}
fn default_act() -> Activation {
    Activation::Gelu
}
fn default_experts() -> usize {
    4
}
fn default_top_k() -> usize {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSpec {
    pub id: String,
    #[serde(flatten)]
    pub kind: BlockKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelSpec {
    pub blocks: Vec<BlockSpec>,
}

impl ModelSpec {
    pub fn d_model(&self) -> Option<usize> {
        self.blocks.iter().find_map(|b| match b.kind {
            BlockKind::Embedding { d_model, .. } => Some(d_model),
            _ => None,
        })
    }
    /// Stable fingerprint of the architecture.
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        // Block ids are UI bookkeeping; only the architecture matters.
        for b in &self.blocks {
            serde_json::to_string(&b.kind).unwrap_or_default().hash(&mut h);
        }
        h.finish()
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Issue {
    pub block_id: Option<String>,
    pub level: &'static str, // "error" | "warning" | "info"
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShapeInfo {
    pub block_id: String,
    /// Output shape of this block for a single sequence: `[seq_len, width]`.
    pub out_shape: [usize; 2],
    pub params: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Analysis {
    pub issues: Vec<Issue>,
    pub shapes: Vec<ShapeInfo>,
    pub total_params: usize,
    pub has_errors: bool,
}

fn norm_params(k: NormKind, d: usize) -> usize {
    match k {
        NormKind::None => 0,
        NormKind::Rms => d,
        NormKind::Layer => 2 * d,
    }
}

/// Parameters of one block, given the model width and vocabulary size.
pub fn block_params(kind: &BlockKind, d: usize, vocab: usize) -> usize {
    match *kind {
        BlockKind::Embedding { d_model, .. } => vocab * d_model,
        BlockKind::PosEncoding | BlockKind::Dropout { .. } => 0,
        BlockKind::Norm { kind } => norm_params(kind, d),
        BlockKind::Attention { n_heads, n_kv_heads, qk_norm, sink, pre_norm, .. } => {
            if n_heads == 0 || d % n_heads != 0 {
                return 0;
            }
            let hd = d / n_heads;
            2 * d * d
                + 2 * d * n_kv_heads * hd
                + if qk_norm { 2 * hd } else { 0 }
                + if sink { n_heads } else { 0 }
                + norm_params(pre_norm, d)
        }
        BlockKind::Ffn { kind, d_ff, n_experts, pre_norm, .. } => {
            norm_params(pre_norm, d)
                + match kind {
                    FfnKind::Glu => 3 * d * d_ff,
                    FfnKind::Mlp => 2 * d * d_ff,
                    FfnKind::Moe => d * n_experts + n_experts * 3 * d * d_ff,
                }
        }
        BlockKind::Head { tied } => {
            if tied {
                0
            } else {
                d * vocab
            }
        }
    }
}

fn err(id: Option<&str>, m: impl Into<String>) -> Issue {
    Issue { block_id: id.map(String::from), level: "error", message: m.into() }
}
fn warn(id: Option<&str>, m: impl Into<String>) -> Issue {
    Issue { block_id: id.map(String::from), level: "warning", message: m.into() }
}

/// Validates the stack and infers shapes / parameter counts.
pub fn analyze(spec: &ModelSpec, vocab: usize, seq_len: usize) -> Analysis {
    let mut issues: Vec<Issue> = Vec::new();
    let blocks = &spec.blocks;

    if blocks.is_empty() {
        issues.push(err(None, "Drag an Embedding block onto the canvas to start."));
    }
    if blocks.len() > MAX_BLOCKS {
        issues.push(err(None, format!("At most {MAX_BLOCKS} blocks are allowed.")));
    }

    let n_embed = blocks.iter().filter(|b| matches!(b.kind, BlockKind::Embedding { .. })).count();
    let n_head = blocks.iter().filter(|b| matches!(b.kind, BlockKind::Head { .. })).count();
    if !blocks.is_empty() {
        if !matches!(blocks[0].kind, BlockKind::Embedding { .. }) {
            issues.push(err(Some(&blocks[0].id), "The stack must start with an Embedding block (tokens must become vectors first)."));
        }
        if n_embed > 1 {
            issues.push(err(None, "Only one Embedding block is allowed."));
        }
        if n_head == 0 {
            issues.push(err(None, "Add an LM Head block at the end so the model can predict the next token."));
        } else if n_head > 1 {
            issues.push(err(None, "Only one LM Head block is allowed."));
        } else if !matches!(blocks.last().unwrap().kind, BlockKind::Head { .. }) {
            issues.push(err(Some(&blocks.last().unwrap().id), "The LM Head must be the last block."));
        }
    }

    let d = spec.d_model().unwrap_or(0);
    if n_embed >= 1 && (d == 0 || d > MAX_D_MODEL) {
        issues.push(err(None, format!("d_model must be between 1 and {MAX_D_MODEL}.")));
    }

    let mut has_attention = false;
    let mut has_position = false;
    let mut shapes = Vec::with_capacity(blocks.len());
    let mut total = 0usize;

    for b in blocks {
        let id = b.id.as_str();
        match &b.kind {
            BlockKind::PosEncoding => has_position = true,
            BlockKind::Attention { n_heads, n_kv_heads, rope, window, causal, dropout, .. } => {
                has_attention = true;
                if *n_heads == 0 || *n_heads > 32 {
                    issues.push(err(Some(id), "n_heads must be between 1 and 32."));
                } else if d > 0 && d % n_heads != 0 {
                    issues.push(err(Some(id), format!("d_model ({d}) must be divisible by n_heads ({n_heads}).")));
                } else if d > 0 {
                    let hd = d / n_heads;
                    if *rope {
                        has_position = true;
                        if hd % 2 != 0 {
                            issues.push(err(Some(id), format!("RoPE needs an even head size (d_model/n_heads = {hd}).")));
                        }
                    }
                    if *n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
                        issues.push(err(Some(id), format!("n_kv_heads ({n_kv_heads}) must divide n_heads ({n_heads}).")));
                    }
                }
                if !*causal {
                    issues.push(warn(Some(id), "Non-causal attention lets each position see the token it must predict, so the loss will collapse without learning language. Use it only to see the effect."));
                }
                if *window > 0 && !*causal {
                    issues.push(warn(Some(id), "The sliding window only applies to causal attention and is ignored here."));
                }
                if !(0.0..1.0).contains(dropout) {
                    issues.push(err(Some(id), "dropout must be in [0, 1)."));
                }
            }
            BlockKind::Ffn { kind, d_ff, n_experts, top_k, dropout, .. } => {
                if *d_ff == 0 || *d_ff > 4096 {
                    issues.push(err(Some(id), "d_ff must be between 1 and 4096."));
                }
                if *kind == FfnKind::Moe {
                    if *n_experts < 2 || *n_experts > 16 {
                        issues.push(err(Some(id), "n_experts must be between 2 and 16."));
                    } else if *top_k == 0 || top_k > n_experts {
                        issues.push(err(Some(id), format!("top_k must be between 1 and n_experts ({n_experts}).")));
                    }
                }
                if !(0.0..1.0).contains(dropout) {
                    issues.push(err(Some(id), "dropout must be in [0, 1)."));
                }
            }
            BlockKind::Dropout { p } if !(0.0..1.0).contains(p) => {
                issues.push(err(Some(id), "p must be in [0, 1)."));
            }
            BlockKind::Head { tied: true } if n_embed == 0 => {
                issues.push(err(Some(id), "A tied head needs an Embedding block."));
            }
            _ => {}
        }

        let params = block_params(&b.kind, d, vocab);
        total = total.saturating_add(params);
        let width = match b.kind {
            BlockKind::Head { .. } => vocab,
            _ => d,
        };
        shapes.push(ShapeInfo { block_id: b.id.clone(), out_shape: [seq_len, width], params });
    }

    if total > MAX_PARAMS {
        issues.push(err(None, format!("Model has {total} parameters; this playground caps at {MAX_PARAMS}.")));
    }
    if blocks.iter().all(|b| !matches!(b.kind, BlockKind::Attention { .. })) && n_embed > 0 && n_head > 0 {
        issues.push(Issue {
            block_id: None,
            level: "info",
            message: "No attention block: every token is predicted from itself alone, so this behaves like a bigram model. Add an Attention block to give it context.".into(),
        });
    } else if has_attention && !has_position {
        issues.push(warn(None, "Attention has no positional signal (enable RoPE or add a Positional Encoding block); word order is invisible to the model."));
    }

    let has_errors = issues.iter().any(|i| i.level == "error");
    Analysis { issues, shapes, total_params: total, has_errors }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn tiny() -> ModelSpec {
        serde_json::from_str(
            r#"{"blocks":[
              {"id":"e","type":"embedding","d_model":32},
              {"id":"a","type":"attention","n_heads":4,"n_kv_heads":2},
              {"id":"f","type":"ffn","kind":"glu","d_ff":64},
              {"id":"h","type":"head"}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn parses_with_defaults_and_validates() {
        let a = analyze(&tiny(), 300, 16);
        assert!(!a.has_errors, "{:?}", a.issues);
        assert_eq!(a.shapes.last().unwrap().out_shape, [16, 300]);
    }

    #[test]
    fn catches_structural_errors() {
        let mut s = tiny();
        s.blocks.swap(0, 1);
        assert!(analyze(&s, 300, 16).has_errors);
        let mut s = tiny();
        s.blocks.pop();
        assert!(analyze(&s, 300, 16).issues.iter().any(|i| i.message.contains("LM Head")));
        let s: ModelSpec = serde_json::from_str(
            r#"{"blocks":[{"id":"e","type":"embedding","d_model":30},{"id":"a","type":"attention","n_heads":4,"n_kv_heads":2},{"id":"h","type":"head"}]}"#,
        )
        .unwrap();
        assert!(analyze(&s, 300, 16).issues.iter().any(|i| i.message.contains("divisible")));
    }
}
