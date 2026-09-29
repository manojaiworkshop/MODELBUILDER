//! Runs a model on the probe sentence and packages everything the right-hand
//! panel shows; also text generation (shared by "sample during training" and
//! the Generate box).

use crate::events::{InspectPayload, MapPoint, NextToken, TokenInfo};
use crate::model::Model;
use crate::spec::Analysis;
use crate::tokenizer::{Bpe, Tokenized, EOS_ID, PAD_ID, UNK_ID, BOS_ID};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub type IB = NdArray;
pub const MAX_PROBE_TOKENS: usize = 32;
const MAP_POINTS: usize = 160;

fn ids_tensor(ids: &[u32], device: &NdArrayDevice) -> Tensor<IB, 2, Int> {
    let v: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
    Tensor::from_data(TensorData::new(v, [1, ids.len()]), device)
}

pub fn probe_ids(bpe: &Bpe, probe: &str, max_len: usize) -> Vec<u32> {
    let mut ids = bpe.encode(probe);
    ids.truncate(max_len.min(MAX_PROBE_TOKENS));
    if ids.is_empty() {
        ids = bpe.encode(" ");
    }
    ids
}

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn is_special(id: usize) -> bool {
    matches!(id as u32, PAD_ID | UNK_ID | BOS_ID | EOS_ID)
}

/// Top two principal components (power iteration) of the rows of `x` (n x d), scaled into [-1, 1].
fn pca2(x: &[f32], n: usize, d: usize) -> Vec<(f32, f32)> {
    if n < 3 || d < 2 {
        return vec![(0.0, 0.0); n];
    }
    let mut mean = vec![0f64; d];
    for r in 0..n {
        for c in 0..d {
            mean[c] += x[r * d + c] as f64 / n as f64;
        }
    }
    let xc: Vec<f64> = (0..n * d).map(|i| x[i] as f64 - mean[i % d]).collect();
    let component = |prev: Option<&Vec<f64>>| -> Vec<f64> {
        let mut v: Vec<f64> = (0..d).map(|i| ((i * 2654435761usize) % 1000) as f64 / 1000.0 - 0.5).collect();
        for _ in 0..40 {
            if let Some(p) = prev {
                let dot: f64 = v.iter().zip(p).map(|(a, b)| a * b).sum();
                v.iter_mut().zip(p).for_each(|(a, b)| *a -= dot * b);
            }
            // w = X v ; v = X^T w
            let w: Vec<f64> = (0..n).map(|r| (0..d).map(|c| xc[r * d + c] * v[c]).sum()).collect();
            let mut nv = vec![0f64; d];
            for r in 0..n {
                for c in 0..d {
                    nv[c] += xc[r * d + c] * w[r];
                }
            }
            let norm = nv.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-12);
            v = nv.into_iter().map(|a| a / norm).collect();
        }
        // sign convention so the picture doesn't flip between updates
        if let Some((i, _)) = v.iter().enumerate().max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()) {
            if v[i] < 0.0 {
                v.iter_mut().for_each(|a| *a = -*a);
            }
        }
        v
    };
    let c1 = component(None);
    let c2 = component(Some(&c1));
    let proj: Vec<(f64, f64)> = (0..n)
        .map(|r| {
            let a: f64 = (0..d).map(|c| xc[r * d + c] * c1[c]).sum();
            let b: f64 = (0..d).map(|c| xc[r * d + c] * c2[c]).sum();
            (a, b)
        })
        .collect();
    let scale = proj.iter().fold(1e-9f64, |m, p| m.max(p.0.abs()).max(p.1.abs()));
    proj.into_iter().map(|(a, b)| ((a / scale) as f32, (b / scale) as f32)).collect()
}

pub fn embedding_map(model: &Model<IB>, tok: &Tokenized) -> Vec<MapPoint> {
    let (w, vocab, d) = model.embedding_matrix();
    let chosen: Vec<u32> = tok.frequent.iter().cloned().filter(|&i| (i as usize) < vocab && !is_special(i as usize)).take(MAP_POINTS).collect();
    let mut rows = Vec::with_capacity(chosen.len() * d);
    for &id in &chosen {
        rows.extend_from_slice(&w[id as usize * d..(id as usize + 1) * d]);
    }
    let pts = pca2(&rows, chosen.len(), d);
    chosen
        .iter()
        .zip(pts)
        .map(|(&id, (x, y))| MapPoint { id, text: tok.bpe.token_text(id), x: crate::trace::r3(x), y: crate::trace::r3(y), count: tok.counts.get(id as usize).copied().unwrap_or(0) as usize })
        .collect()
}

/// Forward the probe sentence through `model` and collect the full snapshot.
pub fn inspect_model(
    model: &Model<IB>, tok: &Tokenized, probe: &str, analysis: &Analysis, seq_len: usize,
    trained_step: Option<usize>, with_map: bool,
) -> InspectPayload {
    let device = Default::default();
    let ids = probe_ids(&tok.bpe, probe, model.max_seq_len.min(seq_len));
    let n = ids.len();
    let (logits, trace) = model.run(ids_tensor(&ids, &device), true);
    let mut trace = trace.expect("trace requested");

    // next-token loss inside the probe (position i predicts token i+1)
    let v = logits.dims()[2];
    let flat: Vec<f32> = logits.into_data().convert::<f32>().to_vec::<f32>().unwrap_or_default();
    if n >= 2 {
        let loss: f32 = (0..n - 1)
            .map(|i| {
                let p = softmax_vec(&flat[i * v..(i + 1) * v]);
                -p[ids[i + 1] as usize].max(1e-12).ln()
            })
            .sum::<f32>()
            / (n - 1) as f32;
        trace.probe_loss = Some(loss);
    }

    let probs = softmax_vec(&trace.last_logits);
    // Only tokens that occur in the training data are candidates: the others were never trained,
    // so listing them would just show noise.
    let seen: std::collections::HashSet<u32> = tok.frequent.iter().copied().collect();
    let mut order: Vec<usize> = (0..probs.len()).filter(|&i| !is_special(i) && seen.contains(&(i as u32))).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let next_tokens = order
        .into_iter()
        .take(6)
        .map(|i| NextToken { id: i as u32, text: tok.bpe.token_text(i as u32), p: crate::trace::r3(probs[i]) })
        .collect();

    InspectPayload {
        ok: true,
        error: None,
        issues: analysis.issues.clone(),
        shapes: analysis.shapes.clone(),
        total_params: analysis.total_params,
        vocab_size: tok.bpe.vocab_size(),
        seq_len,
        trained: trained_step.is_some(),
        step: trained_step,
        tokens: ids.iter().map(|&id| TokenInfo { id, text: tok.bpe.token_text(id) }).collect(),
        layers: trace.layers,
        next_tokens,
        probe_loss: trace.probe_loss,
        map: if with_map { embedding_map(model, tok) } else { vec![] },
    }
}

/// Samples up to `max_new` tokens after `prompt`. Only tokens in `allowed` (those seen in the training
/// data) can be produced: the logits of never-trained tokens are noise, and would show up as garbage bytes.
/// An empty `allowed` means "any token". `on_token(i, id, text_delta)` fires per token;
/// bytes of a multi-byte character split across tokens are held back until complete.
pub fn generate(
    model: &Model<IB>, bpe: &Bpe, allowed: &[u32], prompt: &str, max_new: usize, temperature: f32, top_k: usize, seed: u64,
    mut on_token: impl FnMut(usize, u32, &str) -> bool,
) -> String {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ok = vec![allowed.is_empty(); model.vocab_size()];
    for &t in allowed {
        if let Some(slot) = ok.get_mut(t as usize) {
            *slot = true;
        }
    }
    let mut ids = bpe.encode(prompt);
    if ids.is_empty() {
        ids = bpe.encode("\n");
    }
    let (mut out, mut pending) = (String::new(), Vec::<u8>::new());
    for i in 0..max_new {
        let start = ids.len().saturating_sub(model.max_seq_len);
        let ctx = &ids[start..];
        let logits = model.forward(ids_tensor(ctx, &device));
        let [_, s, v] = logits.dims();
        let mut last: Vec<f32> = logits.slice([0..1, s - 1..s, 0..v]).into_data().convert::<f32>().to_vec::<f32>().unwrap_or_default();
        for (j, l) in last.iter_mut().enumerate() {
            if is_special(j) || !ok[j] {
                *l = f32::NEG_INFINITY;
            }
        }
        let next = if temperature <= 0.0 {
            last.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap_or(0)
        } else {
            let mut cand: Vec<usize> = (0..v).filter(|&j| last[j].is_finite()).collect();
            cand.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
            if top_k > 0 {
                cand.truncate(top_k);
            }
            let scaled: Vec<f32> = cand.iter().map(|&j| last[j] / temperature).collect();
            let p = softmax_vec(&scaled);
            let mut r: f32 = rng.gen();
            let mut pick = *cand.last().unwrap();
            for (k, &pk) in p.iter().enumerate() {
                if r < pk {
                    pick = cand[k];
                    break;
                }
                r -= pk;
            }
            pick
        };
        ids.push(next as u32);
        pending.extend(bpe.token_bytes(next as u32));
        // Emit everything that is complete UTF-8. An incomplete tail (a multi-byte char split across
        // tokens) is held back; bytes that can never be valid are flushed as U+FFFD immediately.
        let take = match std::str::from_utf8(&pending) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to() + e.error_len().unwrap_or(0),
        };
        let delta = String::from_utf8_lossy(&pending[..take]).into_owned();
        pending.drain(..take);
        out.push_str(&delta);
        if !on_token(i, next as u32, &delta) {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::build;
    use crate::spec::ModelSpec;

    fn tiny() -> (Model<IB>, Tokenized) {
        let spec: ModelSpec = serde_json::from_str(
            r#"{"blocks":[{"id":"e","type":"embedding","d_model":16},{"id":"a","type":"attention","n_heads":2,"n_kv_heads":2},{"id":"h","type":"head"}]}"#,
        )
        .unwrap();
        let tok = Tokenized::from_text("hello world hello there", 5);
        let m = build::<IB>(&spec, tok.bpe.vocab_size(), 16, &Default::default()).unwrap();
        (m, tok)
    }

    #[test]
    fn generation_only_uses_allowed_tokens_even_at_high_temperature() {
        let (m, tok) = tiny();
        let mut seen = std::collections::HashSet::new();
        // an untrained model has near-uniform logits, so without the mask it would hit unseen tokens
        for seed in 0..5 {
            generate(&m, &tok.bpe, &tok.frequent, "hel", 40, 3.0, 0, seed, |_, id, _| {
                seen.insert(id);
                true
            });
        }
        assert!(seen.iter().all(|id| tok.frequent.contains(id)), "sampled a token that never occurs in the data");
    }

    #[test]
    fn pca_is_finite_and_bounded() {
        let x: Vec<f32> = (0..40 * 8).map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0).collect();
        for (a, b) in pca2(&x, 40, 8) {
            assert!(a.is_finite() && b.is_finite() && a.abs() <= 1.001 && b.abs() <= 1.001);
        }
    }
}
