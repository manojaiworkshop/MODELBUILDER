//! Plain-data snapshots of what happens inside the network for one probe
//! sentence. These are what the UI draws (heatmaps, attention maps, routing).

use serde::Serialize;

pub fn r3(x: f32) -> f32 {
    if x.is_finite() {
        (x * 1000.0).round() / 1000.0
    } else {
        0.0
    }
}

/// Row-major matrix for a heatmap.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Heat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Heat {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        Heat { rows, cols, data: data.into_iter().map(r3).collect() }
    }

    /// Keeps at most `max_cols` evenly spaced columns of a `[rows, cols]` matrix.
    pub fn downsample_cols(data: &[f32], rows: usize, cols: usize, max_cols: usize) -> Self {
        if cols <= max_cols {
            return Heat::new(rows, cols, data.to_vec());
        }
        let picks: Vec<usize> = (0..max_cols).map(|i| i * cols / max_cols).collect();
        let mut out = Vec::with_capacity(rows * max_cols);
        for r in 0..rows {
            for &c in &picks {
                out.push(data[r * cols + c]);
            }
        }
        Heat::new(rows, max_cols, out)
    }
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub mean: f32,
    pub std: f32,
    pub abs_max: f32,
}

impl Stats {
    pub fn of(v: &[f32]) -> Self {
        if v.is_empty() {
            return Stats::default();
        }
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let abs_max = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        Stats { mean: r3(mean), std: r3(var.sqrt()), abs_max: r3(abs_max) }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AttnTrace {
    /// One `[T, T]` probability matrix per head (rows = query, cols = key).
    pub heads: Vec<Heat>,
    /// Attention-sink models only: mean probability mass each head parks on the sink.
    pub sink_mass: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MoeTrace {
    pub n_experts: usize,
    /// `[T, n_experts]` routing weights (zero for experts not selected).
    pub gates: Heat,
    /// Mean routing weight per expert over the probe tokens.
    pub usage: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerTrace {
    /// Index of the block in the user's stack.
    pub index: usize,
    pub kind: &'static str,
    pub out_shape: [usize; 2],
    pub params: usize,
    pub stats: Stats,
    /// The residual stream after this block, `[T, ≤32 columns]`.
    pub act: Heat,
    pub attn: Option<AttnTrace>,
    pub moe: Option<MoeTrace>,
    /// Filled only while training (L2 norm of this block's gradients / weights).
    pub grad_norm: Option<f32>,
    pub weight_norm: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub layers: Vec<LayerTrace>,
    /// Logits at the last probe position (for the "next token" panel).
    pub last_logits: Vec<f32>,
    /// Mean cross-entropy of the probe sentence (next-token loss inside the probe).
    pub probe_loss: Option<f32>,
}
