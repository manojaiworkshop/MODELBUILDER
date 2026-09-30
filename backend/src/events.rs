//! Socket.IO payloads (client <-> server). JSON keys are camelCase for the UI.

use crate::learn::LlmConfig;
use crate::spec::{Issue, ModelSpec, ShapeInfo};
use crate::trace::LayerTrace;
use serde::{Deserialize, Serialize};

fn d_seq() -> usize { 64 }
fn d_seed() -> u64 { 42 }

// ------------------------------------------------------------- requests

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataSetReq {
    /// Pasted text. Absent = keep the current text and only change the tokenizer / sources.
    pub text: Option<String>,
    #[serde(default)]
    pub merges: usize,
    /// Whether the pasted text is part of the training data.
    #[serde(default = "d_true")]
    pub use_pasted: bool,
}

fn d_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct DocIdReq {
    pub id: String,
    #[serde(default = "d_true")]
    pub enabled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectReq {
    pub spec: ModelSpec,
    #[serde(default)]
    pub probe: String,
    #[serde(default = "d_seq")]
    pub seq_len: usize,
    #[serde(default = "d_seed")]
    pub seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Hyper {
    pub steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub lr: f64,
    pub warmup_steps: usize,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub seed: u64,
    pub probe_every: usize,
    pub eval_every: usize,
    pub sample_every: usize,
    pub sample_tokens: usize,
    pub temperature: f32,
}

impl Default for Hyper {
    fn default() -> Self {
        Hyper {
            steps: 300, batch_size: 16, seq_len: 64, lr: 3e-3, warmup_steps: 20, weight_decay: 0.01,
            grad_clip: 1.0, seed: 42, probe_every: 10, eval_every: 25, sample_every: 50, sample_tokens: 80, temperature: 0.8,
        }
    }
}

impl Hyper {
    pub fn validate(&self) -> Result<(), String> {
        let chk = |ok: bool, m: &str| if ok { Ok(()) } else { Err(m.to_string()) };
        chk((1..=100_000).contains(&self.steps), "steps must be 1..100000")?;
        chk((1..=64).contains(&self.batch_size), "batch size must be 1..64")?;
        chk((4..=256).contains(&self.seq_len), "context length must be 4..256")?;
        chk(self.lr > 0.0 && self.lr <= 1.0, "learning rate must be in (0, 1]")?;
        chk(self.probe_every >= 1 && self.eval_every >= 1 && self.sample_every >= 1, "intervals must be >= 1")?;
        chk(self.sample_tokens <= 500, "sample length must be <= 500")?;
        chk((0.0..=5.0).contains(&self.temperature), "temperature must be 0..5")?;
        chk(self.grad_clip >= 0.0 && self.weight_decay >= 0.0, "grad clip / weight decay must be >= 0")
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainStartReq {
    pub spec: ModelSpec,
    #[serde(default)]
    pub hyper: Hyper,
    #[serde(default)]
    pub probe: String,
    /// What to train on: the uploaded / pasted text, or the generated question-answer pairs.
    #[serde(default)]
    pub source: TrainSource,
    /// Start from fresh weights, or keep training the model from the previous run.
    #[serde(default)]
    pub mode: TrainMode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainSource {
    #[default]
    Text,
    Qa,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainMode {
    #[default]
    New,
    Continue,
}

#[derive(Debug, Deserialize)]
pub struct QaPagesReq {
    pub id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QaGenerateReq {
    pub llm: LlmConfig,
    pub id: String,
    /// 1-based page (or section) numbers.
    pub pages: Vec<usize>,
    #[serde(default = "d_per_page")]
    pub per_page: usize,
    #[serde(default = "d_parallel")]
    pub concurrency: usize,
}
fn d_per_page() -> usize { 5 }
fn d_parallel() -> usize { 4 }

#[derive(Debug, Deserialize)]
pub struct QaRemoveReq {
    /// Pair ids to delete; empty = delete all.
    #[serde(default)]
    pub ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateReq {
    pub prompt: String,
    #[serde(default = "d_gen")]
    pub max_new: usize,
    #[serde(default = "d_temp")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default = "d_seed")]
    pub seed: u64,
}
fn d_gen() -> usize { 120 }
fn d_temp() -> f32 { 0.8 }

// ------------------------------------------------------------ responses

#[derive(Debug, Clone, Serialize)]
pub struct TokenInfo {
    pub id: u32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NextToken {
    pub id: u32,
    pub text: String,
    pub p: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MapPoint {
    pub id: u32,
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectPayload {
    pub ok: bool,
    pub error: Option<String>,
    pub issues: Vec<Issue>,
    pub shapes: Vec<ShapeInfo>,
    pub total_params: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    /// True when the numbers come from a trained model rather than a fresh random init.
    pub trained: bool,
    pub step: Option<usize>,
    pub tokens: Vec<TokenInfo>,
    pub layers: Vec<LayerTrace>,
    pub next_tokens: Vec<NextToken>,
    pub probe_loss: Option<f32>,
    pub map: Vec<MapPoint>,
}

impl InspectPayload {
    pub fn failed(error: impl Into<String>, issues: Vec<Issue>, shapes: Vec<ShapeInfo>, total_params: usize, vocab: usize, seq_len: usize) -> Self {
        InspectPayload {
            ok: false, error: Some(error.into()), issues, shapes, total_params, vocab_size: vocab, seq_len,
            trained: false, step: None, tokens: vec![], layers: vec![], next_tokens: vec![], probe_loss: None, map: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DataInfo {
    pub ok: bool,
    pub error: Option<String>,
    pub chars: usize,
    pub tokens: usize,
    pub vocab_size: usize,
    pub merges: usize,
    /// chars / tokens
    pub compression: f32,
    /// First tokens of the corpus, to show the tokenizer at work.
    pub preview: Vec<TokenInfo>,
    /// Number of text sources (pasted text + documents) that went in.
    pub sources: usize,
    /// Bytes of source text.
    pub bytes: u64,
    /// The corpus was cut at the token limit.
    pub truncated: bool,
    pub build_ms: u64,
}

impl DataInfo {
    pub fn failed(error: impl Into<String>) -> Self {
        DataInfo {
            ok: false, error: Some(error.into()), chars: 0, tokens: 0, vocab_size: 0, merges: 0, compression: 0.0,
            preview: vec![], sources: 0, bytes: 0, truncated: false, build_ms: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StatusPayload {
    /// idle | preparing | running | stopped | done | error
    pub state: &'static str,
    pub message: String,
    pub step: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StepPoint {
    pub step: usize,
    pub loss: f32,
    pub lr: f32,
    pub tok_per_sec: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalPoint {
    pub step: usize,
    pub val_loss: f32,
    pub perplexity: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SamplePayload {
    pub step: usize,
    pub prompt: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenToken {
    pub i: usize,
    pub id: u32,
    pub text: String,
}
