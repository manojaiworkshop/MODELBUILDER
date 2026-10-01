//! System One ("JEV model") — a Jev-style decision model: a bidirectional
//! transformer encoder that reads a state plus typed questions and answers
//! all of them with calibrated probabilities in one forward pass, no text
//! generation. Ported from `transformer_rs/src/sysone/`.
//!
//! This is a genuinely different model shape from the causal, block-composed
//! transformer the rest of this app builds (no LM head, non-causal structured
//! attention, three small decision heads), so it lives in its own module tree
//! and training pipeline rather than being squeezed into the block system.

pub mod api;
pub mod encode;
pub mod generate;
pub mod hfimport;
pub mod loss;
pub mod mask;
pub mod metrics;
pub mod model;
pub mod schema;
pub mod tokens;
pub mod train;
