//! The public contract for System One ("JEV model"): a Jev-style decision
//! model that reads a state plus typed questions and returns calibrated
//! probabilities in one forward pass — no text generation.
//!
//! - **Noul**: a yes/no statement; answer is P(true).
//! - **Choice**: pick among named options; answer is a distribution.
//! - **Score**: rate on ordered levels (low..high); answer is a
//!   distribution over levels plus its expected value in `[0, 1]`.
//!
//! Every `Answer` also carries a `confidence` in `[0, 1]`: the model's
//! learned estimate that its top decision is correct.
//!
//! Ported from `transformer_rs/src/sysone/schema.rs`.

use serde::{Deserialize, Serialize};

/// Hard cap on options per Choice/Score question.
pub const MAX_OPTIONS: usize = 255;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    Noul { text: String },
    Choice { text: String, options: Vec<String> },
    Score { text: String, levels: Vec<String> },
}

impl Question {
    pub fn text(&self) -> &str {
        match self {
            Question::Noul { text } | Question::Choice { text, .. } | Question::Score { text, .. } => text,
        }
    }

    /// Option (or level) labels; empty for Noul.
    pub fn options(&self) -> &[String] {
        match self {
            Question::Noul { .. } => &[],
            Question::Choice { options, .. } => options,
            Question::Score { levels, .. } => levels,
        }
    }
}

/// Ground truth for one question. Probabilities, not hard labels, so
/// teacher-LLM soft labels and hard labels share one format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Target {
    Noul { p: f32 },
    Choice { probs: Vec<f32> },
    Score { probs: Vec<f32> },
}

impl Target {
    pub fn choice_hard(n: usize, gold: usize) -> Target {
        let mut probs = vec![0.0; n];
        probs[gold] = 1.0;
        Target::Choice { probs }
    }
}

/// The tool call a tool-calling sample was derived from: the full ground truth, including
/// arguments (free text, nested values) that the typed questions can't express.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Qualified name, `<server>__<tool>`.
    pub tool: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sample {
    /// Session-local id for list/remove in the UI; 0 for freshly-parsed samples
    /// that have not been assigned one yet.
    #[serde(default)]
    pub id: u32,
    pub state: String,
    pub questions: Vec<Question>,
    pub targets: Vec<Target>,
    /// Which generator produced this (for per-source metrics / provenance).
    #[serde(default)]
    pub source: String,
    /// Set for tool-calling samples (see `toolgen.rs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call: Option<ToolCall>,
}

impl Sample {
    /// Checks that targets line up with questions (kind and cardinality).
    pub fn validate(&self) -> Result<(), String> {
        if self.questions.len() != self.targets.len() {
            return Err(format!("{} questions but {} targets", self.questions.len(), self.targets.len()));
        }
        for (i, (q, t)) in self.questions.iter().zip(&self.targets).enumerate() {
            match (q, t) {
                (Question::Noul { .. }, Target::Noul { p }) => {
                    if !(0.0..=1.0).contains(p) {
                        return Err(format!("question {i}: p out of range"));
                    }
                }
                (Question::Choice { options, .. }, Target::Choice { probs })
                | (Question::Score { levels: options, .. }, Target::Score { probs }) => {
                    if !(2..=MAX_OPTIONS).contains(&options.len()) {
                        return Err(format!("question {i}: needs 2..={MAX_OPTIONS} options, got {}", options.len()));
                    }
                    if probs.len() != options.len() {
                        return Err(format!("question {i}: {} probs for {} options", probs.len(), options.len()));
                    }
                    let sum: f32 = probs.iter().sum();
                    if (sum - 1.0).abs() >= 1e-3 {
                        return Err(format!("question {i}: probs sum to {sum}"));
                    }
                }
                _ => return Err(format!("question {i}: target kind does not match question kind")),
            }
        }
        Ok(())
    }
}

/// Model output for one question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    Noul { p: f32, confidence: f32 },
    Choice { probs: Vec<f32>, best: usize, confidence: f32 },
    Score { probs: Vec<f32>, score: f32, confidence: f32 },
}

impl Answer {
    pub fn confidence(&self) -> f32 {
        match self {
            Answer::Noul { confidence, .. } | Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => *confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Sample {
        Sample {
            id: 1,
            state: "deploy failed twice".into(),
            questions: vec![
                Question::Noul { text: "urgent?".into() },
                Question::Choice { text: "team?".into(), options: vec!["infra".into(), "web".into()] },
            ],
            targets: vec![Target::Noul { p: 1.0 }, Target::choice_hard(2, 0)],
            source: "test".into(),
            call: None,
        }
    }

    #[test]
    fn validates_and_round_trips_json() {
        let s = sample();
        s.validate().unwrap();
        let back: Sample = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn rejects_mismatched_target() {
        let mut s = sample();
        s.targets[0] = Target::choice_hard(2, 0);
        assert!(s.validate().is_err());
        let mut s = sample();
        s.targets[1] = Target::Choice { probs: vec![0.5, 0.2] };
        assert!(s.validate().is_err());
    }
}
