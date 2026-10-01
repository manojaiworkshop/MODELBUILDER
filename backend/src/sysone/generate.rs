//! Teacher-LLM data generation for System One: the model server writes
//! `Sample` JSON (state + typed questions + calibrated soft-probability
//! targets) grounded in a document page — the same format
//! `sysone::schema::Sample` trains on, so generated data needs no
//! conversion.

use crate::learn::{self, LlmConfig};
use crate::sysone::schema::Sample;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const MAX_PAGE_CHARS: usize = 8_000;
pub const MAX_PER_PAGE: usize = 10;
pub const MAX_SAMPLES: usize = 20_000;
/// Most samples asked for in one synthetic-generation model call (mirrors `MAX_PER_PAGE`).
pub const MAX_PER_BATCH: usize = 10;
/// Most samples a single synthetic-generation request may ask for in total.
pub const MAX_SYNTH_TOTAL: usize = 500;

/// Roughly what fraction of the generated questions should be each type. Values are weights, not
/// required to sum to exactly 100 — they're normalized before being described to the model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionMix {
    pub noul: u32,
    pub choice: u32,
    pub score: u32,
}

impl QuestionMix {
    /// Percentages that sum to 100 (as close as integer rounding allows), for the prompt.
    fn percentages(&self) -> (u32, u32, u32) {
        let total = (self.noul + self.choice + self.score).max(1);
        let noul = self.noul * 100 / total;
        let choice = self.choice * 100 / total;
        (noul, choice, 100 - noul - choice)
    }
}

fn system_prompt(n: usize) -> String {
    format!(
        "You write training data for a \"decision model\": it reads a short state (situation \
description) and answers typed questions with CALIBRATED probabilities, not just a single \
best guess.\n\
Respond with ONLY a single JSON object, no markdown fences and no text around it, shaped exactly like:\n\
{{\"samples\": [{{\n  \"state\": string,\n  \"questions\": [ {{\"type\": \"noul\", \"text\": string}} \
| {{\"type\": \"choice\", \"text\": string, \"options\": [string, ...]}} | {{\"type\": \"score\", \"text\": string, \"levels\": [string, ...]}} ],\n\
  \"targets\": [ {{\"type\": \"noul\", \"p\": number}} | {{\"type\": \"choice\", \"probs\": [number, ...]}} | {{\"type\": \"score\", \"probs\": [number, ...]}} ]\n\
}}]}}\n\n\
Rules:\n\
- Write exactly {n} samples. Write fewer only if the page does not support {n} distinct situations.\n\
- \"state\" is a short, self-contained situation grounded in the page — a ticket, a sentence, a \
scenario — never \"according to the page\" or similar; it must make sense with no other context.\n\
- Each sample has 1-3 questions. Mix question types across samples (noul/choice/score), and vary \
the phrasing and the situations — do not reuse the same questions verbatim.\n\
- \"noul\" is a yes/no statement (e.g. \"this needs attention today\"), answered with p = P(true).\n\
- \"choice\" options and \"score\" levels: 2-5 short labels each; levels are ordered low to high \
(e.g. [\"low\",\"medium\",\"high\"]).\n\
- targets.probs must have exactly one entry per option/level and sum to 1. Use CALIBRATED \
probabilities honestly reflecting your uncertainty — e.g. [0.6, 0.4] for a close call, not always \
a confident one-hot [1, 0]. p for noul should likewise reflect genuine uncertainty when the \
situation is ambiguous.\n\
- questions[i] and targets[i] must have the same \"type\" and line up by position.\n\
- Use the language of the page."
    )
}

fn user_prompt(doc: &str, page: usize, text: &str, n: usize) -> String {
    let clipped: String = text.chars().take(MAX_PAGE_CHARS).collect();
    format!("Document: {doc}\nPage: {page}\n\nPage text:\n\"\"\"\n{clipped}\n\"\"\"\n\nWrite {n} samples grounded in this page.")
}

/// Lenient extraction: `{"samples": [...]}` or a bare array. Each element is parsed strictly as a
/// `Sample` and validated; malformed entries are dropped rather than failing the whole page.
pub fn parse_samples(raw: &str, source: &str) -> Vec<Sample> {
    let array = || {
        let (a, b) = (raw.find('[')?, raw.rfind(']')?);
        serde_json::from_str::<serde_json::Value>(raw.get(a..=b)?).ok()
    };
    let object = || learn::balanced_json_object(raw).and_then(|o| serde_json::from_str::<serde_json::Value>(o).ok());
    let array_first = matches!((raw.find('['), raw.find('{')), (Some(a), Some(o)) if a < o);
    let value = if array_first { array().or_else(object) } else { object().or_else(array) };
    let Some(v) = value else { return Vec::new() };
    let arr = match &v {
        serde_json::Value::Array(a) => Some(a),
        other => other.get("samples").and_then(|p| p.as_array()),
    };
    arr.into_iter()
        .flatten()
        .filter_map(|item| {
            let mut s: Sample = serde_json::from_value(item.clone()).ok()?;
            s.source = source.to_string();
            s.validate().ok()?;
            Some(s)
        })
        .take(MAX_SAMPLES)
        .collect()
}

/// Asks the model for `n` samples about one page. One retry for transient failures.
pub async fn generate_page(client: &reqwest::Client, llm: &LlmConfig, doc: &str, page: usize, text: &str, n: usize) -> Result<Vec<Sample>, String> {
    let messages = || {
        vec![
            serde_json::json!({"role": "system", "content": system_prompt(n)}),
            serde_json::json!({"role": "user", "content": user_prompt(doc, page, text, n)}),
        ]
    };
    let source = format!("{doc} p.{page}");
    let mut last = String::new();
    for attempt in 0..2 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(800)).await;
        }
        match learn::complete(client, llm, messages(), Duration::from_secs(120)).await {
            Ok(content) => {
                let mut samples = parse_samples(&content, &source);
                if samples.is_empty() {
                    last = "The model didn't return any usable samples.".into();
                    continue;
                }
                samples.truncate(n);
                return Ok(samples);
            }
            Err(e) if e.fatal => return Err(e.msg),
            Err(e) => last = e.msg,
        }
    }
    Err(last)
}

// ---------------------------------------------------------------- synthetic (no document)

fn synth_system_prompt(n: usize, mix: &QuestionMix) -> String {
    let (noul, choice, score) = mix.percentages();
    format!(
        "You write training data for a \"decision model\": it reads a short state (situation \
description) and answers typed questions with CALIBRATED probabilities, not just a single \
best guess.\n\
Respond with ONLY a single JSON object, no markdown fences and no text around it, shaped exactly like:\n\
{{\"samples\": [{{\n  \"state\": string,\n  \"questions\": [ {{\"type\": \"noul\", \"text\": string}} \
| {{\"type\": \"choice\", \"text\": string, \"options\": [string, ...]}} | {{\"type\": \"score\", \"text\": string, \"levels\": [string, ...]}} ],\n\
  \"targets\": [ {{\"type\": \"noul\", \"p\": number}} | {{\"type\": \"choice\", \"probs\": [number, ...]}} | {{\"type\": \"score\", \"probs\": [number, ...]}} ]\n\
}}]}}\n\n\
Rules:\n\
- Write exactly {n} samples, entirely invented (no source document) but realistic for the domain and \
context given by the user.\n\
- \"state\" is a short, self-contained situation — a ticket, a message, a scenario, a sentence — that \
makes sense with no other context.\n\
- Each sample has 1-3 questions. Vary the phrasing, the situations and the question wording across \
samples — do not reuse the same question verbatim more than a couple of times.\n\
- Across ALL the questions in this batch, aim for roughly this mix of types: {noul}% noul (yes/no), \
{choice}% choice, {score}% score. Treat this as a target ratio for the batch as a whole, not a rule \
for each individual sample.\n\
- \"noul\" is a yes/no statement (e.g. \"this needs attention today\"), answered with p = P(true).\n\
- \"choice\" options and \"score\" levels: 2-5 short labels each; levels are ordered low to high \
(e.g. [\"low\",\"medium\",\"high\"]).\n\
- targets.probs must have exactly one entry per option/level and sum to 1. Use CALIBRATED \
probabilities honestly reflecting your uncertainty — e.g. [0.6, 0.4] for a close call, not always \
a confident one-hot [1, 0]. p for noul should likewise reflect genuine uncertainty when the \
situation is ambiguous.\n\
- questions[i] and targets[i] must have the same \"type\" and line up by position."
    )
}

fn synth_user_prompt(domain: &str, context: &str, n: usize) -> String {
    format!("Domain: {domain}\n\nContext / seed description:\n\"\"\"\n{context}\n\"\"\"\n\nInvent {n} samples for this domain and context.")
}

/// Asks the model for `n` wholly synthetic samples (no source document), for a given domain and
/// free-form seed context, aiming for a target question-type mix. One retry for transient failures.
pub async fn generate_synthetic(client: &reqwest::Client, llm: &LlmConfig, domain: &str, context: &str, n: usize, mix: &QuestionMix) -> Result<Vec<Sample>, String> {
    let messages = || {
        vec![
            serde_json::json!({"role": "system", "content": synth_system_prompt(n, mix)}),
            serde_json::json!({"role": "user", "content": synth_user_prompt(domain, context, n)}),
        ]
    };
    let source = format!("synthetic: {domain}");
    let mut last = String::new();
    for attempt in 0..2 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(800)).await;
        }
        match learn::complete(client, llm, messages(), Duration::from_secs(120)).await {
            Ok(content) => {
                let mut samples = parse_samples(&content, &source);
                if samples.is_empty() {
                    last = "The model didn't return any usable samples.".into();
                    continue;
                }
                samples.truncate(n);
                return Ok(samples);
            }
            Err(e) if e.fatal => return Err(e.msg),
            Err(e) => last = e.msg,
        }
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_and_drops_malformed_samples() {
        let raw = r#"{"samples": [
            {"state": "Server down", "questions": [{"type":"noul","text":"urgent?"}], "targets": [{"type":"noul","p":0.9}]},
            {"state": "bad one", "questions": [{"type":"noul","text":"x?"}], "targets": [{"type":"choice","probs":[0.5,0.5]}]}
        ]}"#;
        let got = parse_samples(raw, "doc.pdf p.1");
        assert_eq!(got.len(), 1, "the type-mismatched sample must be dropped, not crash the batch");
        assert_eq!(got[0].state, "Server down");
        assert_eq!(got[0].source, "doc.pdf p.1");
    }

    #[test]
    fn parses_fenced_bare_array() {
        let raw = "Sure:\n```json\n[{\"state\":\"s\",\"questions\":[{\"type\":\"noul\",\"text\":\"q\"}],\"targets\":[{\"type\":\"noul\",\"p\":1.0}]}]\n```";
        let got = parse_samples(raw, "src");
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn empty_or_invalid_json_yields_no_samples() {
        assert!(parse_samples("not json at all", "src").is_empty());
        assert!(parse_samples("{}", "src").is_empty());
    }

    #[test]
    fn mix_percentages_normalize_to_100_even_when_inputs_do_not_sum_to_100() {
        let mix = QuestionMix { noul: 1, choice: 1, score: 2 };
        let (n, c, s) = mix.percentages();
        assert_eq!((n, c, s), (25, 25, 50));
        assert_eq!(n + c + s, 100);
        let prompt = synth_system_prompt(5, &mix);
        assert!(prompt.contains("25% noul") && prompt.contains("25% choice") && prompt.contains("50% score"), "{prompt}");
    }

    #[test]
    fn zero_mix_does_not_divide_by_zero() {
        let mix = QuestionMix { noul: 0, choice: 0, score: 0 };
        let (n, c, s) = mix.percentages();
        assert_eq!(n + c + s, 100);
    }
}
