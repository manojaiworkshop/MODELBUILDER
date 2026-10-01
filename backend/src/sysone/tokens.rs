//! Special marker tokens layered on top of the byte-level BPE.
//!
//! The BPE owns ids `0..bpe.vocab_size()`. Markers are appended *after* it, so
//! the existing tokenizer is reused unchanged.
//!
//! Ported from `transformer_rs/src/sysone/tokens.rs`.

use crate::tokenizer::Bpe;

pub use crate::tokenizer::{BOS_ID, PAD_ID};

/// Marker kinds, in id order starting at `bpe.vocab_size()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marker {
    QNoul = 0,
    QChoice = 1,
    QScore = 2,
    Opt = 3,
}
pub const NUM_MARKERS: usize = 4;

pub struct Tokenizer {
    pub bpe: Bpe,
}

impl Tokenizer {
    pub fn new(bpe: Bpe) -> Self {
        Tokenizer { bpe }
    }

    /// Total embedding-table size the model needs.
    pub fn vocab_size(&self) -> usize {
        self.bpe.vocab_size() + NUM_MARKERS
    }

    pub fn marker(&self, m: Marker) -> u32 {
        (self.bpe.vocab_size() + m as usize) as u32
    }

    /// Encodes and truncates to at most `max` tokens.
    pub fn encode_truncated(&self, text: &str, max: usize) -> Vec<u32> {
        let mut ids = self.bpe.encode(text);
        ids.truncate(max);
        ids
    }
}
