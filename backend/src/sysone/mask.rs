//! Structured attention mask — the piece that makes "many questions, one
//! forward pass" work.
//!
//! Sequence segments: 0 = state (incl. BOS), i+1 = question i. A query token
//! at segment `si` may attend to key `j` at segment `sj` iff `j` is not
//! padding AND (`sj == 0` OR `sj == si`). So:
//! - state tokens see only the state (bidirectional inside it) — the state
//!   encoding is therefore independent of which questions are asked;
//! - each question sees the whole state plus itself, never other questions,
//!   so answers do not leak between questions.
//!
//! Ported from `transformer_rs/src/sysone/mask.rs`.

use burn::prelude::*;

/// Returns `[batch, n_heads, seq, seq]`, true = blocked.
pub fn structured_mask<B: Backend>(seg: Tensor<B, 2, Int>, pad: Tensor<B, 2, Bool>, n_heads: usize) -> Tensor<B, 4, Bool> {
    let [b, s] = seg.dims();
    let seg_q = seg.clone().reshape([b, s, 1]).expand([b, s, s]);
    let seg_k = seg.reshape([b, 1, s]).expand([b, s, s]);

    let same = seg_q.equal(seg_k.clone()).int();
    let not_same = same.mul_scalar(-1).add_scalar(1);
    let key_is_question = seg_k.greater_elem(0).int();
    let cross_question = key_is_question * not_same;

    let key_pad = pad.reshape([b, 1, s]).expand([b, s, s]).int();
    let blocked: Tensor<B, 3, Bool> = (cross_question + key_pad).greater_elem(0);
    blocked.reshape([b, 1, s, s]).repeat_dim(1, n_heads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn state_and_questions_see_what_they_should() {
        let device = Default::default();
        let seg = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0i64, 0, 1, 1, 2], [1, 5]), &device);
        let pad = Tensor::<B, 2, Bool>::from_data(TensorData::new(vec![false; 5], [1, 5]), &device);
        let m = structured_mask::<B>(seg, pad, 1);
        let v: Vec<bool> = m.into_data().to_vec::<bool>().unwrap();
        let blocked = |i: usize, j: usize| v[i * 5 + j];

        assert!(!blocked(0, 0) && !blocked(0, 1));
        assert!(blocked(0, 2) && blocked(0, 3) && blocked(0, 4));
        assert!(!blocked(2, 0) && !blocked(2, 1) && !blocked(2, 2) && !blocked(2, 3));
        assert!(blocked(2, 4));
        assert!(!blocked(4, 0) && !blocked(4, 1) && !blocked(4, 4));
        assert!(blocked(4, 2) && blocked(4, 3));
    }

    #[test]
    fn padding_keys_are_blocked_and_no_row_is_fully_blocked() {
        let device = Default::default();
        let seg = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0i64, 0, 1, 0], [1, 4]), &device);
        let pad = Tensor::<B, 2, Bool>::from_data(TensorData::new(vec![false, false, false, true], [1, 4]), &device);
        let v: Vec<bool> = structured_mask::<B>(seg, pad, 2).into_data().to_vec::<bool>().unwrap();
        for i in 0..4 {
            assert!(v[i * 4 + 3], "padding key must be blocked for query {i}");
            assert!((0..4).any(|j| !v[i * 4 + j]), "query {i} has no visible key (NaN risk)");
        }
    }
}
