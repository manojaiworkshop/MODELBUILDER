//! Packs one `Sample` (state + K questions) into a single token sequence:
//!
//! ```text
//! [BOS] state tokens ... | [Q_NOUL] q text | [Q_CHOICE] q text [OPT] a [OPT] b | ...
//!   seg 0                    seg 1                seg 2
//! ```
//!
//! `seg` ids drive the structured attention mask (`mask.rs`): state tokens
//! only see the state; each question sees the state and itself. The
//! *readout positions* are the question marker (one per question) and one
//! `[OPT]` marker per option; the model reads its answers off those
//! positions' hidden states.
//!
//! Ported from `transformer_rs/src/sysone/encode.rs`.

use crate::sysone::schema::{Question, Target};
use crate::sysone::tokens::{Marker, Tokenizer, BOS_ID, PAD_ID};
use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct EncodeConfig {
    pub max_seq_len: usize,
    pub max_state_tokens: usize,
    pub max_question_tokens: usize,
    pub max_option_tokens: usize,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        EncodeConfig { max_seq_len: 512, max_state_tokens: 256, max_question_tokens: 32, max_option_tokens: 8 }
    }
}

impl EncodeConfig {
    /// Derives sub-budgets from one user-facing "max sequence length" (roughly half for state
    /// text, the rest for however many typed questions are packed in), so the model's own
    /// `max_seq_len` and the packer's budget can never drift apart.
    pub fn scaled(max_seq_len: usize) -> Self {
        EncodeConfig {
            max_seq_len,
            max_state_tokens: (max_seq_len / 2).max(8),
            max_question_tokens: 32.min(max_seq_len / 4).max(4),
            max_option_tokens: 8.min(max_seq_len / 8).max(2),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Noul = 1,
    Choice = 2,
    Score = 3,
}

#[derive(Debug, Clone)]
pub struct EncQuestion {
    pub kind: Kind,
    /// Sequence index of this question's marker token.
    pub q_pos: usize,
    /// Sequence index of each option's `[OPT]` marker (empty for Noul).
    pub opt_pos: Vec<usize>,
    pub target: Option<Target>,
}

#[derive(Debug, Clone)]
pub struct Example {
    pub ids: Vec<u32>,
    /// 0 = state (incl. BOS), i+1 = question i.
    pub seg: Vec<u32>,
    pub questions: Vec<EncQuestion>,
}

impl Example {
    /// Returns `None` if the questions alone cannot fit in `max_seq_len`
    /// (callers drop the sample; state is truncated to fit, questions are not).
    pub fn encode(tok: &Tokenizer, cfg: &EncodeConfig, state: &str, questions: &[Question], targets: Option<&[Target]>) -> Option<Example> {
        struct QParts {
            kind: Kind,
            marker: u32,
            text: Vec<u32>,
            opts: Vec<Vec<u32>>,
        }
        let parts: Vec<QParts> = questions
            .iter()
            .map(|q| {
                let (kind, marker) = match q {
                    Question::Noul { .. } => (Kind::Noul, tok.marker(Marker::QNoul)),
                    Question::Choice { .. } => (Kind::Choice, tok.marker(Marker::QChoice)),
                    Question::Score { .. } => (Kind::Score, tok.marker(Marker::QScore)),
                };
                QParts {
                    kind,
                    marker,
                    text: tok.encode_truncated(q.text(), cfg.max_question_tokens),
                    opts: q.options().iter().map(|o| tok.encode_truncated(o, cfg.max_option_tokens)).collect(),
                }
            })
            .collect();

        let q_total: usize = parts.iter().map(|p| 1 + p.text.len() + p.opts.iter().map(|o| 1 + o.len()).sum::<usize>()).sum();
        if q_total + 2 > cfg.max_seq_len {
            return None;
        }
        let state_budget = cfg.max_state_tokens.min(cfg.max_seq_len - 1 - q_total);
        let state_ids = tok.encode_truncated(state, state_budget);

        let mut ids = vec![BOS_ID];
        ids.extend(&state_ids);
        let mut seg = vec![0u32; ids.len()];
        let mut enc_qs = Vec::with_capacity(parts.len());

        for (i, p) in parts.iter().enumerate() {
            let seg_id = (i + 1) as u32;
            let q_pos = ids.len();
            ids.push(p.marker);
            ids.extend(&p.text);
            let mut opt_pos = Vec::with_capacity(p.opts.len());
            for o in &p.opts {
                opt_pos.push(ids.len());
                ids.push(tok.marker(Marker::Opt));
                ids.extend(o);
            }
            seg.resize(ids.len(), seg_id);
            enc_qs.push(EncQuestion { kind: p.kind, q_pos, opt_pos, target: targets.map(|t| t[i].clone()) });
        }

        Some(Example { ids, seg, questions: enc_qs })
    }
}

/// Padded, flattened batch on the host. `to_tensors` moves it to a device.
#[derive(Debug, Clone)]
pub struct Batch {
    pub b: usize,
    pub s: usize,
    pub q: usize,
    pub o: usize,
    ids: Vec<i64>,
    seg: Vec<i64>,
    pad: Vec<bool>,
    q_sel: Vec<f32>,
    o_sel: Vec<f32>,
    pub kinds: Vec<u8>,
    w_noul: Vec<f32>,
    w_opt: Vec<f32>,
    o_valid: Vec<f32>,
    noul_t: Vec<f32>,
    opt_t: Vec<f32>,
}

pub fn make_batch(examples: &[Example]) -> Batch {
    let b = examples.len();
    let s = examples.iter().map(|e| e.ids.len()).max().unwrap_or(1);
    let q = examples.iter().map(|e| e.questions.len()).max().unwrap_or(1).max(1);
    let o = examples.iter().flat_map(|e| e.questions.iter().map(|q| q.opt_pos.len())).max().unwrap_or(0).max(1);

    let mut ids = vec![PAD_ID as i64; b * s];
    let mut seg = vec![0i64; b * s];
    let mut pad = vec![true; b * s];
    let mut q_sel = vec![0f32; b * q * s];
    let mut o_sel = vec![0f32; b * q * o * s];
    let mut kinds = vec![0u8; b * q];
    let mut w_noul = vec![0f32; b * q];
    let mut w_opt = vec![0f32; b * q];
    let mut o_valid = vec![0f32; b * q * o];
    let mut noul_t = vec![0f32; b * q];
    let mut opt_t = vec![0f32; b * q * o];

    for (bi, ex) in examples.iter().enumerate() {
        for (i, (&id, &sg)) in ex.ids.iter().zip(&ex.seg).enumerate() {
            ids[bi * s + i] = id as i64;
            seg[bi * s + i] = sg as i64;
            pad[bi * s + i] = false;
        }
        for (qi, eq) in ex.questions.iter().enumerate() {
            let bq = bi * q + qi;
            kinds[bq] = eq.kind as u8;
            q_sel[bq * s + eq.q_pos] = 1.0;
            match eq.kind {
                Kind::Noul => {
                    w_noul[bq] = 1.0;
                    if let Some(Target::Noul { p }) = &eq.target {
                        noul_t[bq] = *p;
                    }
                }
                Kind::Choice | Kind::Score => {
                    w_opt[bq] = 1.0;
                    for (oi, &pos) in eq.opt_pos.iter().enumerate() {
                        o_valid[bq * o + oi] = 1.0;
                        o_sel[(bq * o + oi) * s + pos] = 1.0;
                    }
                    if let Some(Target::Choice { probs } | Target::Score { probs }) = &eq.target {
                        for (oi, &p) in probs.iter().enumerate() {
                            opt_t[bq * o + oi] = p;
                        }
                    }
                }
            }
        }
    }

    Batch { b, s, q, o, ids, seg, pad, q_sel, o_sel, kinds, w_noul, w_opt, o_valid, noul_t, opt_t }
}

/// Device-side tensors for one batch. Shapes: `B` batch, `S` sequence,
/// `Q` question slots, `O` option slots.
pub struct BatchTensors<B: Backend> {
    pub ids: Tensor<B, 2, Int>,
    pub seg: Tensor<B, 2, Int>,
    pub pad: Tensor<B, 2, Bool>,
    pub q_sel: Tensor<B, 3>,
    pub o_sel: Tensor<B, 3>,
    pub w_noul: Tensor<B, 2>,
    pub w_opt: Tensor<B, 2>,
    pub o_valid: Tensor<B, 3>,
    pub noul_t: Tensor<B, 2>,
    pub opt_t: Tensor<B, 3>,
}

impl Batch {
    pub fn to_tensors<B: Backend>(&self, device: &B::Device) -> BatchTensors<B> {
        let (b, s, q, o) = (self.b, self.s, self.q, self.o);
        BatchTensors {
            ids: Tensor::from_data(TensorData::new(self.ids.clone(), [b, s]), device),
            seg: Tensor::from_data(TensorData::new(self.seg.clone(), [b, s]), device),
            pad: Tensor::from_data(TensorData::new(self.pad.clone(), [b, s]), device),
            q_sel: Tensor::from_data(TensorData::new(self.q_sel.clone(), [b, q, s]), device),
            o_sel: Tensor::from_data(TensorData::new(self.o_sel.clone(), [b, q * o, s]), device),
            w_noul: Tensor::from_data(TensorData::new(self.w_noul.clone(), [b, q]), device),
            w_opt: Tensor::from_data(TensorData::new(self.w_opt.clone(), [b, q]), device),
            o_valid: Tensor::from_data(TensorData::new(self.o_valid.clone(), [b, q, o]), device),
            noul_t: Tensor::from_data(TensorData::new(self.noul_t.clone(), [b, q]), device),
            opt_t: Tensor::from_data(TensorData::new(self.opt_t.clone(), [b, q, o]), device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Bpe;

    fn tok() -> Tokenizer {
        Tokenizer::new(Bpe::train("deploy failed urgent infra web team", 10))
    }

    fn qs() -> Vec<Question> {
        vec![
            Question::Noul { text: "urgent?".into() },
            Question::Choice { text: "team?".into(), options: vec!["infra".into(), "web".into(), "db".into()] },
        ]
    }

    #[test]
    fn layout_positions_and_segments_line_up() {
        let t = tok();
        let ex = Example::encode(&t, &EncodeConfig::default(), "deploy failed", &qs(), None).unwrap();
        assert_eq!(ex.ids[0], BOS_ID);
        assert_eq!(ex.ids.len(), ex.seg.len());
        let q0 = &ex.questions[0];
        let q1 = &ex.questions[1];
        assert_eq!(ex.ids[q0.q_pos], t.marker(Marker::QNoul));
        assert_eq!(ex.ids[q1.q_pos], t.marker(Marker::QChoice));
        assert_eq!(q1.opt_pos.len(), 3);
        for &p in &q1.opt_pos {
            assert_eq!(ex.ids[p], t.marker(Marker::Opt));
            assert_eq!(ex.seg[p], 2);
        }
        assert_eq!(ex.seg[q0.q_pos], 1);
        assert_eq!(ex.seg[0], 0);
    }

    #[test]
    fn long_state_is_truncated_to_fit_questions() {
        let t = tok();
        let cfg = EncodeConfig { max_seq_len: 40, ..Default::default() };
        let long = "deploy failed ".repeat(50);
        let ex = Example::encode(&t, &cfg, &long, &qs(), None).unwrap();
        assert!(ex.ids.len() <= 40);
        assert_eq!(ex.questions.len(), 2);
    }

    #[test]
    fn drops_when_questions_alone_do_not_fit() {
        let t = tok();
        let cfg = EncodeConfig { max_seq_len: 6, ..Default::default() };
        assert!(Example::encode(&t, &cfg, "x", &qs(), None).is_none());
    }

    #[test]
    fn batch_pads_and_selects_the_right_positions() {
        let t = tok();
        let a = Example::encode(&t, &EncodeConfig::default(), "deploy", &qs()[..1], None).unwrap();
        let b = Example::encode(&t, &EncodeConfig::default(), "deploy failed urgent", &qs(), None).unwrap();
        let batch = make_batch(&[a.clone(), b.clone()]);
        assert_eq!((batch.b, batch.q, batch.o), (2, 2, 3));
        assert_eq!(batch.s, b.ids.len().max(a.ids.len()));
        assert_eq!(batch.kinds, vec![1, 0, 1, 2]);
        let q_pos = b.questions[1].q_pos;
        let row = &batch.q_sel[(2 + 1) * batch.s..(2 + 2) * batch.s];
        assert_eq!(row.iter().sum::<f32>(), 1.0);
        assert_eq!(row[q_pos], 1.0);
        assert!(batch.pad[a.ids.len()..batch.s].iter().all(|&p| p) || a.ids.len() == batch.s);
    }
}
