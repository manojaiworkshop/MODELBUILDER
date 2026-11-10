//! Laya (Hugging Face `convaiinnovations/laya`): a pretrained decision model to start System One
//! training from, instead of random weights.
//!
//! Laya is a ModernBERT-large encoder (28 layers, 1024 wide, alternating full / 128-token
//! sliding-window attention) with a 2-layer transformer head. A question is **one sequence**:
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
//! ```
//!
//! and each option is scored at its own `[MASK]` token, then softmaxed over the question's options.
//! (This differs from the from-scratch model, which answers many questions in a single pass.)
//!
//! Everything here is written against burn directly and loads the `.safetensors` file as is:
//! tensors are read one at a time (f16 on disk → f32), so the 842 MB file is never held whole.
//! Training can freeze the lower encoder layers (they then run without recording a graph, which
//! is what makes a 395M-parameter model trainable in a few GB).

use crate::sysone::schema::{Answer, Question};
use burn::module::{Module, Param, ParamId};
use burn::prelude::*;
use burn::tensor::activation::{gelu, log_softmax, relu, softmax};
use burn::tensor::module::embedding;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const LN_EPS: f64 = 1e-5;
const INVALID: f32 = -1e9;
pub const QTYPE_CHOICE: i64 = 0;
pub const QTYPE_SCORE: i64 = 1;
pub const QTYPE_NOUL: i64 = 2;
/// Tokens kept per option text (Laya's own cap).
const MAX_OPTION_TOKENS: usize = 48;

pub const REPO: &str = "convaiinnovations/laya";
pub const WEIGHTS_BYTES_HINT: u64 = 842_609_210;

// ------------------------------------------------------------------ files

/// Where the Laya files are kept: `$MODELBUILDER_CACHE/laya` or `~/.cache/modelbuilder/laya`.
pub fn cache_dir() -> PathBuf {
    let base = std::env::var_os("MODELBUILDER_CACHE").map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(".")).join(".cache").join("modelbuilder")
    });
    base.join("laya")
}

/// Files fetched from the Hub: (path in repo, path under the cache dir).
pub const SMALL_FILES: [&str; 3] = ["tokenizer/tokenizer.json", "encoder/config.json", "rl_agent_config.json"];
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// Written once every file is complete, so a half-downloaded cache is never mistaken for a usable one.
pub const COMPLETE_MARKER: &str = "complete";

pub fn is_downloaded(dir: &Path) -> bool {
    dir.join(COMPLETE_MARKER).exists() && dir.join(WEIGHTS_FILE).exists() && SMALL_FILES.iter().all(|f| dir.join(f).exists())
}

// ------------------------------------------------------------------ config

#[derive(Debug, Clone)]
pub struct LayaConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub local_window: usize,
    pub global_every: usize,
    pub theta_global: f32,
    pub theta_local: f32,
    pub head_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    /// Per (type, option-count bucket) temperatures fitted by the authors, and per-type fallbacks.
    pub temperature: [f32; 3],
}

fn num(v: &Value, k: &str) -> Result<usize, String> {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| format!("config is missing '{k}'"))
}

impl LayaConfig {
    pub fn load(dir: &Path) -> Result<LayaConfig, String> {
        let read = |p: &str| -> Result<Value, String> {
            let t = std::fs::read_to_string(dir.join(p)).map_err(|e| format!("reading {p}: {e}"))?;
            serde_json::from_str(&t).map_err(|e| format!("{p}: {e}"))
        };
        let enc = read("encoder/config.json")?;
        let rl = read("rl_agent_config.json")?;
        let rope = |kind: &str, d: f32| enc.pointer(&format!("/rope_parameters/{kind}/rope_theta")).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(d);
        let mut temperature = [1.0f32; 3];
        if let Some(a) = rl.get("temperature").and_then(|t| t.as_array()) {
            for (i, x) in a.iter().take(3).enumerate() {
                temperature[i] = x.as_f64().unwrap_or(1.0) as f32;
            }
        }
        Ok(LayaConfig {
            vocab: num(&enc, "vocab_size")?,
            hidden: num(&enc, "hidden_size")?,
            layers: num(&enc, "num_hidden_layers")?,
            heads: num(&enc, "num_attention_heads")?,
            ffn: num(&enc, "intermediate_size")?,
            local_window: num(&enc, "local_attention")?,
            global_every: num(&enc, "global_attn_every_n_layers")?,
            theta_global: rope("full_attention", 160_000.0),
            theta_local: rope("sliding_attention", 10_000.0),
            head_layers: num(&rl, "head_layers")?,
            max_len: num(&rl, "max_len")?,
            head_max_len: num(&rl, "head_max_len")?,
            temperature,
        })
    }

    fn is_global(&self, layer: usize) -> bool {
        layer % self.global_every == 0
    }
}

// ------------------------------------------------------------------ safetensors reader

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

pub struct Weights {
    file: File,
    base: u64,
    entries: HashMap<String, Entry>,
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal: renormalise
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights, String> {
        let file = File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
        let mut len = [0u8; 8];
        file.read_exact_at(&mut len, 0).map_err(|e| format!("reading header: {e}"))?;
        let n = u64::from_le_bytes(len);
        if n == 0 || n > 64 << 20 {
            return Err("not a safetensors file".into());
        }
        let mut raw = vec![0u8; n as usize];
        file.read_exact_at(&mut raw, 8).map_err(|e| format!("reading header: {e}"))?;
        let header: Value = serde_json::from_slice(&raw).map_err(|e| format!("bad safetensors header: {e}"))?;
        let mut entries = HashMap::new();
        for (name, v) in header.as_object().ok_or("bad safetensors header")? {
            if name == "__metadata__" {
                continue;
            }
            let offs = v.get("data_offsets").and_then(|o| o.as_array()).ok_or("bad tensor entry")?;
            entries.insert(
                name.clone(),
                Entry {
                    dtype: v.get("dtype").and_then(|d| d.as_str()).unwrap_or("").to_string(),
                    shape: v.get("shape").and_then(|s| s.as_array()).map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default(),
                    start: offs.first().and_then(|x| x.as_u64()).unwrap_or(0),
                    end: offs.get(1).and_then(|x| x.as_u64()).unwrap_or(0),
                },
            );
        }
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        let max_end = entries.values().map(|e| e.end).max().unwrap_or(0);
        if 8 + n + max_end > size {
            return Err("the weights file is truncated — download it again".into());
        }
        Ok(Weights { file, base: 8 + n, entries })
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// One tensor as f32, with the shape the file records.
    pub fn read(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let e = self.entries.get(name).ok_or_else(|| format!("the weights file has no tensor '{name}'"))?;
        let mut raw = vec![0u8; (e.end - e.start) as usize];
        self.file.read_exact_at(&mut raw, self.base + e.start).map_err(|e| format!("reading {name}: {e}"))?;
        let data: Vec<f32> = match e.dtype.as_str() {
            "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            other => return Err(format!("{name}: unsupported dtype {other}")),
        };
        Ok((data, e.shape.clone()))
    }
}

// ------------------------------------------------------------------ building blocks

/// `Param` from a host tensor: trainable (tracked) or frozen (plain, never given a gradient).
fn param<B: Backend, const D: usize>(data: Vec<f32>, shape: [usize; D], device: &B::Device, trainable: bool) -> Param<Tensor<B, D>> {
    let t = Tensor::<B, D>::from_data(TensorData::new(data, shape), device);
    if trainable {
        Param::from_tensor(t)
    } else {
        Param::initialized(ParamId::new(), t)
    }
}

struct Loader<'a> {
    w: &'a Weights,
}

impl Loader<'_> {
    fn vec<B: Backend>(&self, name: &str, device: &B::Device, trainable: bool) -> Result<Param<Tensor<B, 1>>, String> {
        let (d, s) = self.w.read(name)?;
        if s.len() != 1 {
            return Err(format!("{name}: expected a vector, got shape {s:?}"));
        }
        Ok(param(d, [s[0]], device, trainable))
    }

    fn mat<B: Backend>(&self, name: &str, device: &B::Device, trainable: bool) -> Result<Param<Tensor<B, 2>>, String> {
        let (d, s) = self.w.read(name)?;
        if s.len() != 2 {
            return Err(format!("{name}: expected a matrix, got shape {s:?}"));
        }
        Ok(param(d, [s[0], s[1]], device, trainable))
    }

    /// A torch `Linear` weight `[out, in]`, stored transposed (`[in, out]`) for `x.matmul(w)`.
    fn linear<B: Backend>(&self, prefix: &str, bias: bool, device: &B::Device, trainable: bool) -> Result<Lin<B>, String> {
        let name = format!("{prefix}.weight");
        let (d, s) = self.w.read(&name)?;
        if s.len() != 2 {
            return Err(format!("{name}: expected a matrix, got shape {s:?}"));
        }
        let (o, i) = (s[0], s[1]);
        let mut t = vec![0f32; d.len()];
        for r in 0..o {
            for c in 0..i {
                t[c * o + r] = d[r * i + c];
            }
        }
        let b = if bias { Some(self.vec::<B>(&format!("{prefix}.bias"), device, trainable)?) } else { None };
        Ok(Lin { w: param(t, [i, o], device, trainable), b })
    }

    fn norm<B: Backend>(&self, prefix: &str, bias: bool, device: &B::Device, trainable: bool) -> Result<Ln<B>, String> {
        let b = if bias { Some(self.vec::<B>(&format!("{prefix}.bias"), device, trainable)?) } else { None };
        Ok(Ln { g: self.vec::<B>(&format!("{prefix}.weight"), device, trainable)?, b })
    }
}

#[derive(Module, Debug)]
pub struct Lin<B: Backend> {
    w: Param<Tensor<B, 2>>,
    b: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> Lin<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = x.matmul(self.w.val().unsqueeze::<3>());
        match &self.b {
            Some(b) => y + b.val().unsqueeze::<3>(),
            None => y,
        }
    }
}

#[derive(Module, Debug)]
pub struct Ln<B: Backend> {
    g: Param<Tensor<B, 1>>,
    b: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> Ln<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = x.clone().mean_dim(2);
        let xc = x - mean;
        let var = xc.clone().powf_scalar(2.0).mean_dim(2);
        let y = xc / (var + LN_EPS).sqrt() * self.g.val().unsqueeze::<3>();
        match &self.b {
            Some(b) => y + b.val().unsqueeze::<3>(),
            None => y,
        }
    }
}

/// `[b, h, s, hd]` -> rotated with the "split halves" convention ModernBERT uses.
fn rope<B: Backend>(x: Tensor<B, 4>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, hd] = x.dims();
    let half = hd / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.clone().slice([0..b, 0..h, 0..s, half..hd]);
    let rot = Tensor::cat(vec![x2.neg(), x1], 3);
    x * cos.clone() + rot * sin.clone()
}

fn heads<B: Backend>(t: Tensor<B, 3>, h: usize) -> Tensor<B, 4> {
    let [b, s, d] = t.dims();
    t.reshape([b, s, h, d / h]).swap_dims(1, 2)
}

fn attend<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, blocked: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
    let [b, h, s, hd] = q.dims();
    let scores = q.matmul(k.swap_dims(2, 3)).div_scalar((hd as f32).sqrt());
    let scores = scores.mask_fill(blocked.expand([b, h, s, s]), INVALID);
    softmax(scores, 3).matmul(v).swap_dims(1, 2).reshape([b, s, h * hd])
}

#[derive(Module, Debug)]
struct EncLayer<B: Backend> {
    attn_norm: Option<Ln<B>>,
    wqkv: Lin<B>,
    wo: Lin<B>,
    mlp_norm: Ln<B>,
    wi: Lin<B>,
    mlp_wo: Lin<B>,
    n_heads: usize,
    ffn: usize,
    local: bool,
}

impl<B: Backend> EncLayer<B> {
    fn forward(&self, x: Tensor<B, 3>, rot: &(Tensor<B, 4>, Tensor<B, 4>), blocked: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let [_, _, d] = x.dims();
        let a_in = match &self.attn_norm {
            Some(n) => n.forward(x.clone()),
            None => x.clone(),
        };
        let [b, s, _] = a_in.dims();
        let qkv = self.wqkv.forward(a_in);
        let q = heads(qkv.clone().slice([0..b, 0..s, 0..d]), self.n_heads);
        let k = heads(qkv.clone().slice([0..b, 0..s, d..2 * d]), self.n_heads);
        let v = heads(qkv.slice([0..b, 0..s, 2 * d..3 * d]), self.n_heads);
        let (q, k) = (rope(q, &rot.0, &rot.1), rope(k, &rot.0, &rot.1));
        let x = x + self.wo.forward(attend(q, k, v, blocked));

        let h = self.wi.forward(self.mlp_norm.forward(x.clone()));
        let input = h.clone().slice([0..b, 0..s, 0..self.ffn]);
        let gate = h.slice([0..b, 0..s, self.ffn..2 * self.ffn]);
        x + self.mlp_wo.forward(gelu(input) * gate)
    }
}

/// torch `TransformerEncoderLayer(norm_first=True, activation=relu)`.
#[derive(Module, Debug)]
struct HeadLayer<B: Backend> {
    norm1: Ln<B>,
    in_proj: Lin<B>,
    out_proj: Lin<B>,
    norm2: Ln<B>,
    linear1: Lin<B>,
    linear2: Lin<B>,
    n_heads: usize,
}

impl<B: Backend> HeadLayer<B> {
    fn forward(&self, x: Tensor<B, 3>, blocked: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let qkv = self.in_proj.forward(self.norm1.forward(x.clone()));
        let q = heads(qkv.clone().slice([0..b, 0..s, 0..d]), self.n_heads);
        let k = heads(qkv.clone().slice([0..b, 0..s, d..2 * d]), self.n_heads);
        let v = heads(qkv.slice([0..b, 0..s, 2 * d..3 * d]), self.n_heads);
        let x = x + self.out_proj.forward(attend(q, k, v, blocked));
        x.clone() + self.linear2.forward(relu(self.linear1.forward(self.norm2.forward(x))))
    }
}

// ------------------------------------------------------------------ the model

/// A padded batch of question sequences.
pub struct LayaBatch {
    pub b: usize,
    pub s: usize,
    pub k: usize,
    pub ids: Vec<i64>,
    /// 1 = real token.
    pub att: Vec<u8>,
    pub marker_pos: Vec<i64>,
    pub marker_ok: Vec<u8>,
    pub qtype: Vec<i64>,
    /// `[b * k]`, zero where there is no option.
    pub target: Vec<f32>,
}

pub struct BatchTensors<B: Backend> {
    ids: Tensor<B, 2, Int>,
    /// true = blocked, `[b, 1, s, s]`
    global: Tensor<B, 4, Bool>,
    local: Tensor<B, 4, Bool>,
    qtype: Tensor<B, 1, Int>,
    marker_pos: Tensor<B, 2, Int>,
    marker_ok: Tensor<B, 2, Bool>,
    pub target: Tensor<B, 2>,
}

impl LayaBatch {
    pub fn tensors<B: Backend>(&self, window: usize, device: &B::Device) -> BatchTensors<B> {
        let (b, s, k) = (self.b, self.s, self.k);
        let half = (window / 2) as i64;
        let (mut g, mut l) = (vec![false; b * s * s], vec![false; b * s * s]);
        for bi in 0..b {
            for j in 0..s {
                let pad = self.att[bi * s + j] == 0;
                for i in 0..s {
                    let idx = (bi * s + i) * s + j;
                    g[idx] = pad;
                    l[idx] = pad || (i as i64 - j as i64).abs() > half;
                }
            }
        }
        BatchTensors {
            ids: Tensor::from_data(TensorData::new(self.ids.clone(), [b, s]), device),
            global: Tensor::from_data(TensorData::new(g, [b, 1, s, s]), device),
            local: Tensor::from_data(TensorData::new(l, [b, 1, s, s]), device),
            qtype: Tensor::from_data(TensorData::new(self.qtype.clone(), [b]), device),
            marker_pos: Tensor::from_data(TensorData::new(self.marker_pos.clone(), [b, k]), device),
            marker_ok: Tensor::from_data(TensorData::new(self.marker_ok.iter().map(|&x| x != 0).collect::<Vec<bool>>(), [b, k]), device),
            target: Tensor::from_data(TensorData::new(self.target.clone(), [b, k]), device),
        }
    }
}

#[derive(Module, Debug)]
pub struct LayaModel<B: Backend> {
    tok_emb: Param<Tensor<B, 2>>,
    emb_norm: Ln<B>,
    layers: Vec<EncLayer<B>>,
    final_norm: Ln<B>,
    type_emb: Param<Tensor<B, 2>>,
    head: Vec<HeadLayer<B>>,
    scorer_norm: Ln<B>,
    scorer1: Lin<B>,
    scorer2: Lin<B>,
    hidden: usize,
    n_heads: usize,
    window: usize,
    theta_global: usize,
    theta_local: usize,
}

/// Position tables `[1, 1, s, hd]` for one rope base.
fn rope_tables<B: Backend>(theta: f32, hd: usize, s: usize, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = hd / 2;
    let (mut c, mut sn) = (Vec::with_capacity(s * hd), Vec::with_capacity(s * hd));
    for p in 0..s {
        let mut row_c = vec![0f32; hd];
        let mut row_s = vec![0f32; hd];
        for i in 0..half {
            let inv = (theta as f64).powf(-(2.0 * i as f64) / hd as f64);
            let a = (p as f64 * inv) as f32;
            row_c[i] = a.cos();
            row_c[i + half] = a.cos();
            row_s[i] = a.sin();
            row_s[i + half] = a.sin();
        }
        c.extend(row_c);
        sn.extend(row_s);
    }
    (Tensor::from_data(TensorData::new(c, [1, 1, s, hd]), device), Tensor::from_data(TensorData::new(sn, [1, 1, s, hd]), device))
}

impl<B: Backend> LayaModel<B> {
    /// Loads the pretrained weights. Everything below the last `train_layers` encoder layers (and
    /// the embeddings) is frozen; the head, scorer and type embeddings always train.
    pub fn load(dir: &Path, cfg: &LayaConfig, train_layers: usize, device: &B::Device) -> Result<LayaModel<B>, String> {
        let w = Weights::open(&dir.join(WEIGHTS_FILE))?;
        let ld = Loader { w: &w };
        let first_trainable = cfg.layers.saturating_sub(train_layers);
        let tok = ld.mat::<B>("encoder.embeddings.tok_embeddings.weight", device, false)?;
        if tok.val().dims() != [cfg.vocab, cfg.hidden] {
            return Err("the weights don't match the config".into());
        }
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let t = i >= first_trainable;
            let p = format!("encoder.layers.{i}");
            layers.push(EncLayer {
                attn_norm: if w.has(&format!("{p}.attn_norm.weight")) { Some(ld.norm::<B>(&format!("{p}.attn_norm"), false, device, t)?) } else { None },
                wqkv: ld.linear(&format!("{p}.attn.Wqkv"), false, device, t)?,
                wo: ld.linear(&format!("{p}.attn.Wo"), false, device, t)?,
                mlp_norm: ld.norm(&format!("{p}.mlp_norm"), false, device, t)?,
                wi: ld.linear(&format!("{p}.mlp.Wi"), false, device, t)?,
                mlp_wo: ld.linear(&format!("{p}.mlp.Wo"), false, device, t)?,
                n_heads: cfg.heads,
                ffn: cfg.ffn,
                local: !cfg.is_global(i),
            });
        }
        let mut head = Vec::new();
        for i in 0..cfg.head_layers {
            let p = format!("head.layers.{i}");
            let (iw, is) = w.read(&format!("{p}.self_attn.in_proj_weight"))?;
            let (rows, cols) = (is[0], is[1]);
            let mut t = vec![0f32; iw.len()];
            for r in 0..rows {
                for c in 0..cols {
                    t[c * rows + r] = iw[r * cols + c];
                }
            }
            let in_proj = Lin { w: param(t, [cols, rows], device, true), b: Some(ld.vec::<B>(&format!("{p}.self_attn.in_proj_bias"), device, true)?) };
            head.push(HeadLayer {
                norm1: ld.norm(&format!("{p}.norm1"), true, device, true)?,
                in_proj,
                out_proj: ld.linear(&format!("{p}.self_attn.out_proj"), true, device, true)?,
                norm2: ld.norm(&format!("{p}.norm2"), true, device, true)?,
                linear1: ld.linear(&format!("{p}.linear1"), true, device, true)?,
                linear2: ld.linear(&format!("{p}.linear2"), true, device, true)?,
                n_heads: (cfg.hidden / 64).max(1),
            });
        }
        Ok(LayaModel {
            tok_emb: tok,
            emb_norm: ld.norm("encoder.embeddings.norm", false, device, false)?,
            layers,
            final_norm: ld.norm("encoder.final_norm", false, device, true)?,
            type_emb: ld.mat::<B>("type_emb.weight", device, true)?,
            head,
            scorer_norm: ld.norm("scorer.0", true, device, true)?,
            scorer1: ld.linear("scorer.1", true, device, true)?,
            scorer2: ld.linear("scorer.3", true, device, true)?,
            hidden: cfg.hidden,
            n_heads: cfg.heads,
            window: cfg.local_window,
            theta_global: cfg.theta_global as usize,
            theta_local: cfg.theta_local as usize,
        })
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// `[b, k]` option logits (invalid slots at a large negative value).
    pub fn forward(&self, t: &BatchTensors<B>) -> Tensor<B, 2> {
        let [b, s] = t.ids.dims();
        let device = t.ids.device();
        let hd = self.hidden / self.n_heads;
        let rot_g = rope_tables::<B>(self.theta_global as f32, hd, s, &device);
        let rot_l = rope_tables::<B>(self.theta_local as f32, hd, s, &device);

        let mut x = embedding(self.tok_emb.val(), t.ids.clone());
        x = self.emb_norm.forward(x);
        for layer in &self.layers {
            x = if layer.local { layer.forward(x, &rot_l, t.local.clone()) } else { layer.forward(x, &rot_g, t.global.clone()) };
        }
        let mut h = self.final_norm.forward(x);
        h = h + self.type_emb.val().select(0, t.qtype.clone()).unsqueeze_dim::<3>(1);
        for l in &self.head {
            h = l.forward(h, t.global.clone());
        }
        let k = t.marker_pos.dims()[1];
        let idx = t.marker_pos.clone().unsqueeze_dim::<3>(2).repeat_dim(2, self.hidden);
        let m = h.gather(1, idx);
        let z = self.scorer2.forward(gelu(self.scorer1.forward(self.scorer_norm.forward(m)))).reshape([b, k]);
        z.mask_fill(t.marker_ok.clone().bool_not(), INVALID)
    }
}

/// Soft-target cross-entropy over each row's valid options, averaged over rows.
pub fn loss<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let logp = log_softmax(logits, 1);
    (target * logp).sum_dim(1).neg().mean()
}

// ------------------------------------------------------------------ tokenizing questions

pub struct LayaTok {
    tok: tokenizers::Tokenizer,
    cls: i64,
    sep: i64,
    pub pad: i64,
    mask: i64,
    mask_text: String,
}

pub struct Encoded {
    pub ids: Vec<i64>,
    pub markers: Vec<usize>,
    pub qtype: i64,
}

impl LayaTok {
    pub fn load(dir: &Path) -> Result<LayaTok, String> {
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer/tokenizer.json")).map_err(|e| format!("tokenizer: {e}"))?;
        let id = |s: &str| tok.token_to_id(s).map(|x| x as i64).ok_or_else(|| format!("the tokenizer has no {s} token"));
        Ok(LayaTok { cls: id("[CLS]")?, sep: id("[SEP]")?, pad: id("[PAD]")?, mask: id("[MASK]")?, mask_text: "[MASK]".into(), tok })
    }

    fn ids(&self, text: &str) -> Vec<i64> {
        self.tok.encode(text.replace(&self.mask_text, " "), false).map(|e| e.get_ids().iter().map(|&x| x as i64).collect()).unwrap_or_default()
    }

    /// The option texts of a question in Laya's rendering, and its type.
    pub fn render(q: &Question) -> (i64, &'static str, Vec<String>) {
        match q {
            Question::Noul { .. } => (QTYPE_NOUL, "noul", vec!["false: no, the statement does not hold".into(), "true: yes, the statement holds".into()]),
            Question::Choice { options, .. } => (QTYPE_CHOICE, "choice", options.clone()),
            Question::Score { levels, .. } => (QTYPE_SCORE, "score", levels.iter().enumerate().map(|(i, l)| format!("level {i}: {l}")).collect()),
        }
    }

    /// `[CLS] <type> question: ... [SEP] [MASK] opt ... [SEP] state [SEP]`, cut to `max_len` (the
    /// state is what gets cut). `None` when the options alone don't fit in `head_max_len`.
    pub fn encode(&self, state: &str, q: &Question, max_len: usize, head_max_len: usize) -> Option<Encoded> {
        let (qtype, name, opts) = Self::render(q);
        let mut head = self.ids(&format!("{name} question: {}", q.text()));
        let mut opt_ids: Vec<Vec<i64>> = opts
            .iter()
            .map(|o| {
                let mut v = vec![self.mask];
                v.extend(self.ids(&format!(" {o}")).into_iter().take(MAX_OPTION_TOKENS));
                v
            })
            .collect();
        let mut budget = head_max_len as i64 - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
        if budget < 16 {
            let per = 4.max((head_max_len.saturating_sub(16)) / opt_ids.len().max(1));
            for o in opt_ids.iter_mut() {
                o.truncate(per);
            }
            budget = head_max_len as i64 - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
        }
        head.truncate(8.max(budget.max(0) as usize));
        let mut ids = vec![self.cls];
        ids.extend(head);
        ids.push(self.sep);
        let mut markers = Vec::new();
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend(o);
        }
        ids.push(self.sep);
        let room = max_len.saturating_sub(ids.len() + 1);
        let st = self.ids(state);
        ids.extend(st.into_iter().take(room));
        ids.push(self.sep);
        ids.truncate(max_len);
        let markers: Vec<usize> = markers.into_iter().filter(|&m| m < max_len).collect();
        if markers.len() != opts.len() {
            return None;
        }
        Some(Encoded { ids, markers, qtype })
    }
}

/// Pads `items` (with their target distributions) into one batch.
pub fn make_batch(items: &[(&Encoded, &[f32])], pad: i64) -> LayaBatch {
    let b = items.len();
    let s = items.iter().map(|(e, _)| e.ids.len()).max().unwrap_or(1);
    let k = items.iter().map(|(e, _)| e.markers.len()).max().unwrap_or(1).max(1);
    let mut out = LayaBatch { b, s, k, ids: vec![pad; b * s], att: vec![0; b * s], marker_pos: vec![0; b * k], marker_ok: vec![0; b * k], qtype: Vec::with_capacity(b), target: vec![0.0; b * k] };
    for (bi, (e, t)) in items.iter().enumerate() {
        for (i, &id) in e.ids.iter().enumerate() {
            out.ids[bi * s + i] = id;
            out.att[bi * s + i] = 1;
        }
        for (j, &m) in e.markers.iter().enumerate() {
            out.marker_pos[bi * k + j] = m as i64;
            out.marker_ok[bi * k + j] = 1;
            out.target[bi * k + j] = t.get(j).copied().unwrap_or(0.0);
        }
        out.qtype.push(e.qtype);
    }
    out
}

// ------------------------------------------------------------------ serving

pub struct TrainedLaya<B: Backend> {
    pub model: LayaModel<B>,
    pub tok: LayaTok,
    pub max_len: usize,
    pub head_max_len: usize,
    /// Fitted on held-out questions after training: choice, score, noul.
    pub temperature: [f32; 3],
    pub step: usize,
}

pub fn confidence(p: &[f32]) -> f32 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = -p.iter().map(|&x| if x > 1e-12 { x * x.ln() } else { 0.0 }).sum::<f32>();
    (1.0 - ent / (k as f32).ln()).clamp(0.0, 1.0)
}

fn softmax_t(z: &[f32], t: f32) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|x| ((x - m) / t).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

pub fn answer_from_logits(q: &Question, z: &[f32], temperature: &[f32; 3]) -> Answer {
    let (qt, _, _) = LayaTok::render(q);
    let p = softmax_t(z, temperature[qt as usize].max(0.05));
    let confidence = confidence(&p);
    match q {
        Question::Noul { .. } => Answer::Noul { p: p[1], confidence },
        Question::Choice { .. } => {
            let best = p.iter().enumerate().fold((0, -1.0), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;
            Answer::Choice { probs: p, best, confidence }
        }
        Question::Score { .. } => {
            let n = p.len();
            let score = p.iter().enumerate().map(|(l, x)| x * l as f32 / (n.max(2) - 1) as f32).sum();
            Answer::Score { probs: p, score, confidence }
        }
    }
}

/// Raw logits for every question about `state`, batched (order preserved).
pub fn logits_for<B: Backend>(model: &LayaModel<B>, tok: &LayaTok, device: &B::Device, state: &str, questions: &[Question], max_len: usize, head_max_len: usize) -> Result<Vec<Vec<f32>>, String> {
    let mut enc = Vec::with_capacity(questions.len());
    for (i, q) in questions.iter().enumerate() {
        enc.push(tok.encode(state, q, max_len, head_max_len).ok_or_else(|| format!("question {i}: its options don't fit in {head_max_len} tokens"))?);
    }
    let mut order: Vec<usize> = (0..enc.len()).collect();
    order.sort_by_key(|&i| enc[i].ids.len());
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); enc.len()];
    for chunk in order.chunks(16) {
        let zeros: Vec<Vec<f32>> = chunk.iter().map(|&i| vec![0.0; enc[i].markers.len()]).collect();
        let items: Vec<(&Encoded, &[f32])> = chunk.iter().zip(&zeros).map(|(&i, z)| (&enc[i], z.as_slice())).collect();
        let batch = make_batch(&items, tok.pad);
        let z = model.forward(&batch.tensors::<B>(model.window(), device)).into_data().convert::<f32>().to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
        for (r, &i) in chunk.iter().enumerate() {
            out[i] = z[r * batch.k..r * batch.k + enc[i].markers.len()].to_vec();
        }
    }
    Ok(out)
}

impl<B: Backend> TrainedLaya<B> {
    pub fn decide(&self, device: &B::Device, state: &str, questions: &[Question]) -> Result<Vec<Answer>, String> {
        if questions.is_empty() {
            return Err("at least one question is required".into());
        }
        let z = logits_for(&self.model, &self.tok, device, state, questions, self.max_len, self.head_max_len)?;
        Ok(questions.iter().zip(&z).map(|(q, z)| answer_from_logits(q, z, &self.temperature)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    /// Needs the downloaded weights; prints ids + raw logits to compare against the reference
    /// implementation: `cargo test --release laya_reference -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn laya_reference() {
        let dir = cache_dir();
        assert!(dir.join(WEIGHTS_FILE).exists(), "download Laya first");
        let cfg = LayaConfig::load(&dir).unwrap();
        let tok = LayaTok::load(&dir).unwrap();
        let model = LayaModel::<NdArray>::load(&dir, &cfg, 0, &Default::default()).unwrap();
        let state = "The server room is on fire and the alarm is ringing. Staff are still inside.";
        let qs = vec![
            Question::Choice { text: "What should we do first?".into(), options: vec!["evacuate".into(), "ignore it".into(), "order lunch".into()] },
            Question::Noul { text: "Is this an emergency?".into() },
            Question::Score { text: "How urgent is this?".into(), levels: vec!["low".into(), "medium".into(), "high".into()] },
        ];
        for q in &qs {
            let e = tok.encode(state, q, cfg.max_len, cfg.head_max_len).unwrap();
            println!("IDS {:?}", e.ids);
        }
        let z = logits_for(&model, &tok, &Default::default(), state, &qs, cfg.max_len, cfg.head_max_len).unwrap();
        for z in &z {
            println!("LOGITS {:?}", z);
        }
    }

    #[test]
    fn f16_conversion_matches_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x0001) - 5.9604645e-8).abs() < 1e-14); // smallest subnormal
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!((f16_to_f32(0x3555) - 0.33325195).abs() < 1e-6);
    }
}
