//! Summarization dataset generation from an uploaded document.
//!
//! Same page-picking flow as `qa.rs` (and reuses its page-boundary helpers), but each selected
//! page gets exactly one summary instead of several question/answer pairs. Each pair keeps the
//! source text it summarizes, so training text can be built without touching disk again (see
//! `format_training_text`).

use crate::hfdata::{Style, SUMMARIZE_INSTRUCTION, SUMMARIZE_PROMPT};
use crate::learn::{self, LlmConfig};
use serde::Serialize;
use std::time::Duration;

/// Most characters of one page sent to the model, and kept on the pair for training.
const MAX_PAGE_CHARS: usize = 12_000;
pub const MAX_PAIRS: usize = 50_000;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryPair {
    pub id: u32,
    pub doc_id: String,
    pub doc: String,
    pub page: usize,
    /// Generated / imported style; also sets the training layout (see `training_block`).
    pub style: Style,
    /// System prompt for chat-style pairs (empty = none).
    pub system: String,
    /// The page text the summary was written from (clipped, same as what the model saw).
    pub text: String,
    pub summary: String,
}

fn style_rules(style: Style, sentences: usize) -> String {
    match style {
        Style::Hf => format!("- CNN/DailyMail style: write the summary as about {sentences} short \"highlights\", each one a single \
sentence on its own line (separated by newline characters inside the JSON string, no bullet symbols or numbering).\n"),
        Style::Instruction => format!("- Instruction style: write the summary as the response to the instruction \"Summarize the following text.\": about \
{sentences} sentences, direct, with no preamble.\n"),
        Style::Chat => format!("- Chat style: write the summary as an AI assistant's reply to \"Summarize the following text\": about \
{sentences} sentences, natural and direct, with no preamble such as \"Sure\" or \"Here is a summary\".\n"),
        _ => format!("- Write about {sentences} sentences covering only the main points of the page.\n"),
    }
}

fn system_prompt(sentences: usize, style: Style) -> String {
    let rules = style_rules(style, sentences);
    format!(
        "You write a training example that teaches a model to summarize one page of a document.\n\
Respond with ONLY a single JSON object, no markdown fences and no text around it, shaped exactly like:\n\
{{\"summary\": string}}\n\n\
Rules:\n\
{rules}\
- The summary must stand on its own: never write \"this page\", \"the text above\" or similar — \
write as if describing the content directly.\n\
- Stay faithful to the page. Never invent facts that are not on it.\n\
- Use the language of the page."
    )
}

fn user_prompt(doc: &str, page: usize, text: &str) -> String {
    let clipped: String = text.chars().take(MAX_PAGE_CHARS).collect();
    format!("Document: {doc}\nPage: {page}\n\nPage text:\n\"\"\"\n{clipped}\n\"\"\"\n\nWrite the summary.")
}

/// Lenient extraction: `{"summary": "..."}`, or — if the model skipped JSON entirely — the
/// trimmed reply itself, so a plain-text summary isn't thrown away.
pub fn parse_summary(raw: &str) -> Option<String> {
    if let Some(obj) = learn::balanced_json_object(raw) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(obj) {
            if let Some(s) = v.get("summary").and_then(|x| x.as_str()) {
                let s = s.trim();
                if !s.is_empty() {
                    return Some(s.chars().take(2000).collect());
                }
            }
        }
    }
    let t = raw.trim().strip_prefix("Summary:").unwrap_or(raw.trim()).trim();
    let t = t.trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    if t.is_empty() || t.len() > 4000 || t.starts_with('{') || t.starts_with('[') {
        None
    } else {
        Some(t.chars().take(2000).collect())
    }
}

/// Asks the model for one summary of a page. One retry for transient failures. Returns the
/// (clipped) source text alongside the summary, since the caller stores both.
pub async fn generate_page(client: &reqwest::Client, llm: &LlmConfig, doc: &str, page: usize, text: &str, sentences: usize, style: Style) -> Result<(String, String), String> {
    let clipped: String = text.chars().take(MAX_PAGE_CHARS).collect();
    let messages = || {
        vec![
            serde_json::json!({"role": "system", "content": system_prompt(sentences, style)}),
            serde_json::json!({"role": "user", "content": user_prompt(doc, page, &clipped)}),
        ]
    };
    let mut last = String::new();
    for attempt in 0..2 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(800)).await;
        }
        match learn::complete(client, llm, messages(), Duration::from_secs(120)).await {
            Ok(content) => match parse_summary(&content) {
                Some(summary) => return Ok((clipped, summary)),
                None => last = "The model didn't return a usable summary.".into(),
            },
            Err(e) if e.fatal => return Err(e.msg),
            Err(e) => last = e.msg,
        }
    }
    Err(last)
}

// ---------------------------------------------------------------- training text

/// One training block, laid out by the pair's style: simple `Text: … / Summary: …`,
/// hf (CNN/DailyMail) `Article: … / Highlights: …`, chat `<|user|>` / `<|assistant|>` turns.
pub fn training_block(p: &SummaryPair) -> String {
    match p.style {
        Style::Hf => format!("Article: {}\nHighlights: {}\n\n", p.text, p.summary),
        Style::Instruction => format!("### Instruction:\n{SUMMARIZE_INSTRUCTION}\n\n### Input:\n{}\n\n### Response:\n{}\n\n", p.text, p.summary),
        Style::Chat => {
            let sys = if p.system.is_empty() { String::new() } else { format!("<|system|>\n{}\n", p.system) };
            format!("{sys}<|user|>\n{SUMMARIZE_PROMPT}{}\n<|assistant|>\n{}\n\n", p.text, p.summary)
        }
        _ => format!("Text: {}\nSummary: {}\n\n", p.text, p.summary),
    }
}

/// The text a model is trained on: one block per pair, in that pair's style.
pub fn format_training_text(pairs: &[SummaryPair]) -> String {
    pairs.iter().map(training_block).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_and_bare_text_replies() {
        assert_eq!(parse_summary(r#"{"summary":"A short summary."}"#), Some("A short summary.".to_string()));
        assert_eq!(parse_summary("Sure:\n```json\n{\"summary\": \"Fenced.\"}\n```"), Some("Fenced.".to_string()));
        assert_eq!(parse_summary("Summary: Plain text reply."), Some("Plain text reply.".to_string()));
        assert_eq!(parse_summary("  Just prose, no label.  "), Some("Just prose, no label.".to_string()));
        assert!(parse_summary("").is_none());
        assert!(parse_summary(r#"{"summary": ""}"#).is_none());
    }

    #[test]
    fn training_layout_follows_the_pair_style() {
        let mut p = SummaryPair { id: 1, doc_id: "d".into(), doc: "a".into(), page: 1, style: Style::Hf, system: String::new(), text: "T.".into(), summary: "S.".into() };
        assert_eq!(training_block(&p), "Article: T.\nHighlights: S.\n\n");
        p.style = Style::Chat;
        assert_eq!(training_block(&p), "<|user|>\nSummarize the following text:\n\nT.\n<|assistant|>\nS.\n\n");
        p.style = Style::Instruction;
        assert_eq!(training_block(&p), "### Instruction:\nSummarize the following text.\n\n### Input:\nT.\n\n### Response:\nS.\n\n");
        assert!(system_prompt(3, Style::Hf).contains("highlights"));
        assert!(system_prompt(3, Style::Chat).contains("AI assistant"));
    }

    #[test]
    fn training_text_format() {
        let p = SummaryPair { id: 1, doc_id: "d1".into(), doc: "a.pdf".into(), page: 1, style: Style::Simple, system: String::new(), text: "Full page text.".into(), summary: "Short version.".into() };
        assert_eq!(format_training_text(&[p]), "Text: Full page text.\nSummary: Short version.\n\n");
    }

    #[tokio::test]
    async fn generate_page_retries_a_transient_failure() {
        use axum::{routing::post, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<serde_json::Value>| {
                let c = c.clone();
                async move {
                    let sys = body["messages"][0]["content"].as_str().unwrap_or_default().to_string();
                    if c.fetch_add(1, Ordering::SeqCst) == 0 {
                        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "busy"})));
                    }
                    assert!(sys.contains("about 3 sentences"), "{sys}");
                    let body = serde_json::json!({"summary": "It is a summary."});
                    (axum::http::StatusCode::OK, Json(serde_json::json!({"choices":[{"message":{"content": body.to_string()}}]})))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let llm = LlmConfig { base_url: format!("http://{addr}/v1"), api_key: "k".into(), model: "m".into() };
        let (text, summary) = generate_page(&reqwest::Client::new(), &llm, "doc.pdf", 3, "Some page text.", 3, Style::Simple).await.unwrap();
        assert_eq!(text, "Some page text.");
        assert_eq!(summary, "It is a summary.");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "one failure then one success");
    }
}
