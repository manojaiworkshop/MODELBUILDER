//! Question-answer dataset generation from an uploaded document.
//!
//! The extractor records where each page / slide starts in a document's text (see
//! `Sink::mark_page`). Formats without pages (plain text, Word, …) get "sections" of a few
//! thousand characters instead. The user picks pages; each page's text goes to the configured
//! chat model in its own request (many in flight at once, see `session::Session::spawn_qa`), and
//! the model returns question/answer pairs grounded in that page. The pairs can then be used as
//! the training text (`format_training_text`).

use crate::hfdata::Style;
use crate::learn::{self, LlmConfig};
use crate::tokenizer::{compact_ids, Bpe, ImportedTok, Tokenized};
use serde::Serialize;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

/// Target size of a synthetic section for documents without real pages.
const SECTION_BYTES: u64 = 3500;
/// Most synthetic sections offered per document.
const MAX_SECTIONS: usize = 2000;
/// Most characters of one page sent to the model.
pub const MAX_PAGE_CHARS: usize = 12_000;
/// Pages with less text than this are skipped (headers, blank pages).
pub const MIN_PAGE_CHARS: usize = 40;
pub const MAX_PER_PAGE: usize = 20;
pub const MAX_PAIRS: usize = 50_000;

/// Only the first characters go to the browser (enough to build a sampling prompt); the full
/// page text stays on the server for export and training.
fn context_head<S: serde::Serializer>(c: &std::sync::Arc<str>, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&c.chars().take(300).collect::<String>())
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QaPair {
    pub id: u32,
    pub doc_id: String,
    pub doc: String,
    pub page: usize,
    /// The style the pair was generated / imported in; it also sets how it is laid out for training.
    pub style: Style,
    /// System prompt for chat-style pairs (empty = none).
    pub system: String,
    pub question: String,
    pub answer: String,
    /// The page text the pair was written from (kept for export; shared by all pairs of a page,
    /// never sent to the browser). Empty for imported pairs that had no context.
    #[serde(rename = "contextHead", serialize_with = "context_head")]
    pub context: std::sync::Arc<str>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageInfo {
    /// 1-based page (or section) number.
    pub n: usize,
    pub chars: usize,
    pub preview: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageList {
    pub pages: Vec<PageInfo>,
    /// True when the document has no real pages and these are fixed-size sections.
    pub synthetic: bool,
    /// True when the document is longer than what is offered.
    pub truncated: bool,
}

/// Start offsets of ~`SECTION_BYTES` sections, cut at line / space boundaries.
fn synthetic_starts(path: &Path, total: u64) -> std::io::Result<Vec<u64>> {
    let mut f = File::open(path)?;
    let mut starts = vec![0u64];
    let mut at = 0u64;
    let mut buf = vec![0u8; 512];
    while at + SECTION_BYTES < total && starts.len() < MAX_SECTIONS {
        let probe = at + SECTION_BYTES;
        f.seek(SeekFrom::Start(probe))?;
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        // cut at the next line break (else space) after the target, and never inside a UTF-8 sequence
        let cut = buf[..n]
            .iter()
            .position(|&b| b == b'\n')
            .or_else(|| buf[..n].iter().position(|&b| b == b' '))
            .map(|p| p + 1)
            .unwrap_or_else(|| buf[..n].iter().position(|&b| (b & 0xC0) != 0x80).unwrap_or(0));
        at = probe + cut as u64;
        if at < total {
            starts.push(at);
        }
    }
    Ok(starts)
}

pub fn read_range(path: &Path, start: u64, end: u64) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(start))?;
    let mut buf = Vec::with_capacity((end - start).min(1 << 20) as usize);
    f.take(end - start).read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).trim().to_string())
}

/// The page boundaries of a document: `(start, end)` byte ranges, real ones when the extractor
/// recorded them, otherwise synthetic sections. Second value: synthetic?
pub fn page_ranges(path: &Path, marks: &[u64], total: u64) -> std::io::Result<(Vec<(u64, u64)>, bool)> {
    let (starts, synthetic) = if marks.len() >= 2 { (marks.to_vec(), false) } else { (synthetic_starts(path, total)?, true) };
    let ranges = starts.iter().enumerate().map(|(i, &s)| (s, starts.get(i + 1).copied().unwrap_or(total))).collect();
    Ok((ranges, synthetic))
}

pub fn list_pages(path: &Path, marks: &[u64], total: u64) -> std::io::Result<PageList> {
    let (ranges, synthetic) = page_ranges(path, marks, total)?;
    let mut pages = Vec::new();
    for (i, &(s, e)) in ranges.iter().enumerate() {
        // byte length is a fine size hint; near-empty pages (headers, blanks) aren't offered
        if e.saturating_sub(s) < MIN_PAGE_CHARS as u64 {
            continue;
        }
        let head = read_range(path, s, e.min(s + 400))?;
        if head.is_empty() {
            continue;
        }
        pages.push(PageInfo { n: i + 1, chars: (e - s) as usize, preview: head.chars().take(90).collect::<String>().replace('\n', " ") });
    }
    let truncated = synthetic && ranges.len() >= MAX_SECTIONS;
    Ok(PageList { pages, synthetic, truncated })
}

// ---------------------------------------------------------------- the model

fn style_rules(style: Style) -> &'static str {
    match style {
        Style::Hf => "- SQuAD style (extractive): every answer is a short span copied word for word from the page — \
a name, number, phrase or clause, usually 1-12 words, exactly as written and never paraphrased — so it can be \
found in the page text. Questions may rely on the page as context.\n",
        Style::Instruction => "- Instruction style: write each question as an instruction or question a user would give an assistant \
(\"Explain …\", \"List …\", \"What …\"), and each answer as the complete response in 1-4 sentences, without mentioning the page or \
document.\n",
        Style::Chat => "- Chat style: write each question the way a person would ask an AI assistant (a question or a short \
request), and each answer as the assistant's helpful, natural reply in 1-4 sentences, without mentioning the page or \
document.\n",
        _ => "- Answers are self-contained, 1-3 sentences, and stay faithful to the page.\n",
    }
}

fn system_prompt(n: usize, style: Style) -> String {
    let rules = style_rules(style);
    format!(
        "You write question-and-answer training data from one page of a document.\n\
Respond with ONLY a single JSON object, no markdown fences and no text around it, shaped exactly like:\n\
{{\"pairs\": [{{\"question\": string, \"answer\": string}}]}}\n\n\
Rules:\n\
- Write exactly {n} pairs. Write fewer only if the page does not contain {n} distinct facts.\n\
- Every question must be answerable from the page alone and must make sense on its own: never write \
\"according to the text\", \"in this passage\", \"on this page\" or refer to a figure by position.\n\
{rules}\
- Never invent facts that are not on the page.\n\
- Cover different facts and vary the question types (what / why / how / when / how many / define).\n\
- Use the language of the page. No numbering or prefixes such as \"Q:\"."
    )
}

fn user_prompt(doc: &str, page: usize, text: &str, n: usize) -> String {
    let clipped: String = text.chars().take(MAX_PAGE_CHARS).collect();
    format!("Document: {doc}\nPage: {page}\n\nPage text:\n\"\"\"\n{clipped}\n\"\"\"\n\nWrite {n} question-answer pairs.")
}

/// Lenient extraction: `{"pairs":[{"question","answer"}]}`, a bare array, or objects using the
/// keys `q` / `a`. Anything unusable is dropped rather than failing the whole page.
pub fn parse_pairs(raw: &str) -> Vec<(String, String)> {
    // a bare array starts with `[` before any `{`; otherwise look for the wrapping object
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
        other => other.get("pairs").or_else(|| other.get("qa")).or_else(|| other.get("items")).and_then(|p| p.as_array()),
    };
    let field = |item: &serde_json::Value, keys: &[&str]| keys.iter().find_map(|k| item.get(*k).and_then(|x| x.as_str())).map(|s| s.trim().to_string()).unwrap_or_default();
    arr.into_iter()
        .flatten()
        .map(|item| (field(item, &["question", "q"]), field(item, &["answer", "a"])))
        .filter(|(q, a)| q.len() >= 4 && !a.is_empty())
        .map(|(q, a)| (q.chars().take(600).collect(), a.chars().take(2000).collect()))
        .collect()
}

/// Asks the model for `n` pairs about one page. One retry for transient failures.
pub async fn generate_page(client: &reqwest::Client, llm: &LlmConfig, doc: &str, page: usize, text: &str, n: usize, style: Style) -> Result<Vec<(String, String)>, String> {
    let messages = || {
        vec![
            serde_json::json!({"role": "system", "content": system_prompt(n, style)}),
            serde_json::json!({"role": "user", "content": user_prompt(doc, page, text, n)}),
        ]
    };
    let mut last = String::new();
    for attempt in 0..2 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(800)).await;
        }
        match learn::complete(client, llm, messages(), Duration::from_secs(120)).await {
            Ok(content) => {
                let mut pairs = parse_pairs(&content);
                if pairs.is_empty() {
                    last = "The model didn't return any question-answer pairs.".into();
                    continue;
                }
                pairs.truncate(n);
                return Ok(pairs);
            }
            Err(e) if e.fatal => return Err(e.msg),
            Err(e) => last = e.msg,
        }
    }
    Err(last)
}

// ---------------------------------------------------------------- training text

/// One training block, laid out by the pair's style:
/// simple `Question: … / Answer: …`, hf (SQuAD) with a leading `Context: …`, chat with
/// `<|user|>` / `<|assistant|>` turns.
pub fn training_block(p: &QaPair) -> String {
    match p.style {
        Style::Hf if !p.context.is_empty() => format!("Context: {}\nQuestion: {}\nAnswer: {}\n\n", p.context, p.question, p.answer),
        Style::Instruction if p.context.is_empty() => format!("### Instruction:\n{}\n\n### Response:\n{}\n\n", p.question, p.answer),
        Style::Instruction => format!("### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}\n\n", p.question, p.context, p.answer),
        Style::Chat => {
            let sys = if p.system.is_empty() { String::new() } else { format!("<|system|>\n{}\n", p.system) };
            format!("{sys}<|user|>\n{}\n<|assistant|>\n{}\n\n", p.question, p.answer)
        }
        _ => format!("Question: {}\nAnswer: {}\n\n", p.question, p.answer),
    }
}

/// The text a model is trained on: one block per pair, in that pair's style.
pub fn format_training_text(pairs: &[QaPair]) -> String {
    pairs.iter().map(training_block).collect()
}

/// Tokenizes the generated text. With `bpe` (continuing a trained model) its vocabulary is kept, so
/// the weights still line up with the token ids; otherwise an imported `custom` tokenizer is used
/// (keeping only the ids that occur), or a fresh BPE with `merges` merges is learned.
pub fn tokenize(text: &str, bpe: Option<Bpe>, custom: Option<std::sync::Arc<ImportedTok>>, merges: usize) -> Result<Tokenized, String> {
    let (bpe, ids, counts) = match (bpe, custom) {
        (None, Some(tok)) => compact_ids(&tok, &tok.encode_raw(text))?,
        (prior, _) => {
            let bpe = prior.unwrap_or_else(|| Bpe::train(text, merges));
            let ids: Vec<u16> = bpe.encode(text).into_iter().map(|i| i as u16).collect();
            let mut counts = vec![0u64; bpe.vocab_size()];
            for &i in &ids {
                counts[i as usize] += 1;
            }
            (bpe, ids, counts)
        }
    };
    Ok(Tokenized::from_parts(bpe, ids, text.chars().count(), text.len() as u64, counts, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_layout_follows_the_pair_style() {
        let mut p = QaPair { id: 1, doc_id: "d".into(), doc: "a".into(), page: 1, style: Style::Simple, system: String::new(), question: "Q?".into(), answer: "A.".into(), context: "Ctx.".into() };
        assert_eq!(training_block(&p), "Question: Q?\nAnswer: A.\n\n");
        p.style = Style::Hf;
        assert_eq!(training_block(&p), "Context: Ctx.\nQuestion: Q?\nAnswer: A.\n\n");
        p.style = Style::Chat;
        assert_eq!(training_block(&p), "<|user|>\nQ?\n<|assistant|>\nA.\n\n");
        p.system = "Be brief.".into();
        assert_eq!(training_block(&p), "<|system|>\nBe brief.\n<|user|>\nQ?\n<|assistant|>\nA.\n\n");
        p.style = Style::Instruction;
        assert_eq!(training_block(&p), "### Instruction:\nQ?\n\n### Input:\nCtx.\n\n### Response:\nA.\n\n");
        p.context = "".into();
        assert_eq!(training_block(&p), "### Instruction:\nQ?\n\n### Response:\nA.\n\n");
        assert!(system_prompt(3, Style::Hf).contains("copied word for word"));
        assert!(system_prompt(3, Style::Chat).contains("AI assistant"));
        assert!(!system_prompt(3, Style::Simple).contains("word for word"));
    }

    #[test]
    fn parses_object_array_and_fenced_replies() {
        let a = parse_pairs(r#"{"pairs":[{"question":"What is X?","answer":"X is a thing."}]}"#);
        assert_eq!(a, vec![("What is X?".to_string(), "X is a thing.".to_string())]);
        let b = parse_pairs("Sure:\n```json\n[{\"q\":\"Why is the sky blue?\",\"a\":\"Rayleigh scattering.\"}]\n```");
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].1, "Rayleigh scattering.");
        assert!(parse_pairs("no json here").is_empty());
        assert!(parse_pairs(r#"{"pairs":[{"question":"ok?","answer":""}]}"#).is_empty());
    }

    #[test]
    fn real_and_synthetic_pages() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.txt");
        let text = "alpha beta gamma delta. ".repeat(400); // ~9.6 KB
        std::fs::write(&path, &text).unwrap();
        let total = text.len() as u64;
        // no marks -> synthetic sections that tile the file exactly
        let (r, syn) = page_ranges(&path, &[], total).unwrap();
        assert!(syn && r.len() >= 2);
        assert_eq!(r[0].0, 0);
        assert_eq!(r.last().unwrap().1, total);
        assert!(r.windows(2).all(|w| w[0].1 == w[1].0));
        // real marks are honoured
        let marks = [0u64, 100, 200];
        let (r, syn) = page_ranges(&path, &marks, total).unwrap();
        assert!(!syn);
        assert_eq!(r, vec![(0, 100), (100, 200), (200, total)]);
        assert_eq!(read_range(&path, 100, 200).unwrap().len(), text[100..200].trim().len());
        let list = list_pages(&path, &marks, total).unwrap();
        assert_eq!(list.pages.iter().map(|p| p.n).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn continuing_keeps_the_vocabulary_new_learns_one() {
        let text = "Question: What is a cat?\nAnswer: A small pet.\n\n".repeat(30);
        let fresh = tokenize(&text, None, None, 40).unwrap();
        assert!(fresh.bpe.vocab_size() > 260 && !fresh.ids.is_empty());
        let kept = tokenize("Question: something new\nAnswer: yes\n", Some(fresh.bpe.clone()), None, 999).unwrap();
        assert_eq!(kept.bpe.vocab_size(), fresh.bpe.vocab_size());
        assert!(kept.ids.iter().all(|&i| (i as usize) < fresh.bpe.vocab_size()));
    }

    #[tokio::test]
    async fn generate_page_retries_a_transient_failure_and_caps_the_pairs() {
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
                    assert!(sys.contains("exactly 2 pairs"), "{sys}");
                    let pairs = serde_json::json!({"pairs": [
                        {"question": "What is one?", "answer": "The first number."},
                        {"question": "What is two?", "answer": "The second number."},
                        {"question": "What is three?", "answer": "Extra, must be dropped."}]});
                    (axum::http::StatusCode::OK, Json(serde_json::json!({"choices":[{"message":{"content": pairs.to_string()}}]})))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let llm = LlmConfig { base_url: format!("http://{addr}/v1"), api_key: "k".into(), model: "m".into() };
        let got = generate_page(&reqwest::Client::new(), &llm, "doc.pdf", 3, "Numbers: one, two, three are numbers.", 2, Style::Simple).await.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 2, "one failure then one success");

        // a missing model name is fatal: no retry
        let bad = LlmConfig { model: "".into(), ..llm };
        assert!(generate_page(&reqwest::Client::new(), &bad, "d", 1, "text", 1, Style::Simple).await.unwrap_err().contains("model name"));
    }

    #[test]
    fn training_text_format() {
        let p = QaPair { id: 1, doc_id: "d1".into(), doc: "a.pdf".into(), page: 1, style: Style::Simple, system: String::new(), question: "Q1?".into(), answer: "A1.".into(), context: "".into() };
        assert_eq!(format_training_text(&[p]), "Question: Q1?\nAnswer: A1.\n\n");
    }
}
