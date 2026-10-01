//! Hugging Face-style JSONL export / import of the Q&A and summary datasets.
//!
//! Three styles, for both directions (one JSON object per line, loadable with
//! `datasets.load_dataset("json", data_files=...)`):
//!
//! | style    | Q&A                                              | summary                       |
//! |----------|--------------------------------------------------|-------------------------------|
//! | `hf`     | SQuAD: id, title, context, question, answers     | CNN/DailyMail: id, article, highlights |
//! | `simple` | context, question, answer                        | document, summary (XSum-like) |
//! | `instruction` | Alpaca: instruction (question), input (context), output (answer) | instruction ("Summarize…"), input (text), output |
//! | `chat`   | messages: [{role, content}, …], optional leading `system` | same |
//!
//! Files are JSONL or Parquet (Parquet is converted to/from JSONL with arrow, so both share one code path).
//!
//! Import also accepts `auto` (per record, whichever style fits), a JSON array, and the official
//! nested SQuAD file (`{"data": [{"paragraphs": [{"context", "qas": […]}]}]}`).

use crate::qa::QaPair;
use crate::summarize::SummaryPair;
use crate::sysone::schema::Sample;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const SUMMARIZE_PROMPT: &str = "Summarize the following text:\n\n";
/// The `instruction` of an instruction-style summary record.
pub const SUMMARIZE_INSTRUCTION: &str = "Summarize the following text.";
const MAX_TEXT_CHARS: usize = 12_000;
const MAX_ANSWER_CHARS: usize = 4_000;

/// How a dataset is written: the LLM's output style at generation, the training-text layout, and
/// the file schema on export / import. `Auto` only makes sense when importing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Style {
    Auto,
    Hf,
    #[default]
    Simple,
    Instruction,
    Chat,
}

impl Style {
    /// A concrete style for generation (`Auto` has no meaning there).
    pub fn concrete(self) -> Style {
        if self == Style::Auto { Style::Simple } else { self }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Qa,
    Summary,
    /// System One samples: unlike Q&A/summary there is no standard alternate schema to translate
    /// into (SQuAD, Alpaca, …), so this kind ignores `style` and is always plain JSONL of
    /// `Sample` itself — already a stable, documented shape (see `sysone::schema`).
    SysOne,
}

fn jsonl(rows: impl IntoIterator<Item = Value>) -> String {
    let mut out = String::new();
    for r in rows {
        out.push_str(&r.to_string());
        out.push('\n');
    }
    out
}

/// Character (not byte) offset of `needle` in `hay`, as SQuAD's `answer_start` uses; -1 if absent.
fn char_offset(hay: &str, needle: &str) -> i64 {
    if needle.is_empty() {
        return -1;
    }
    hay.find(needle).map(|b| hay[..b].chars().count() as i64).unwrap_or(-1)
}

/// `messages` for a chat record: an optional system turn, then user and assistant.
fn chat_messages(system: &str, user: String, assistant: &str) -> Value {
    let mut m = Vec::new();
    if !system.trim().is_empty() {
        m.push(json!({ "role": "system", "content": system }));
    }
    m.push(json!({ "role": "user", "content": user }));
    m.push(json!({ "role": "assistant", "content": assistant }));
    json!({ "messages": m })
}

fn qa_user_turn(context: &str, question: &str) -> String {
    if context.is_empty() {
        question.to_string()
    } else {
        format!("Context:\n{context}\n\nQuestion: {question}")
    }
}

/// `system_fallback` fills in for pairs that carry no system prompt of their own (chat style only).
pub fn export_qa(pairs: &[QaPair], style: Style, system_fallback: &str) -> Result<String, String> {
    let row = |p: &QaPair| -> Value {
        let ctx: &str = &p.context;
        match style {
            Style::Hf => json!({
                "id": format!("{}-p{}-{}", p.doc_id, p.page, p.id),
                "title": p.doc,
                "context": ctx,
                "question": p.question,
                "answers": { "text": [p.answer], "answer_start": [char_offset(ctx, &p.answer)] },
            }),
            Style::Simple => json!({ "context": ctx, "question": p.question, "answer": p.answer }),
            Style::Instruction => json!({ "instruction": p.question, "input": ctx, "output": p.answer }),
            _ => chat_messages(if p.system.is_empty() { system_fallback } else { &p.system }, qa_user_turn(ctx, &p.question), &p.answer),
        }
    };
    match style {
        Style::Auto => Err("Pick a concrete style to export (hf, simple or chat).".into()),
        _ => Ok(jsonl(pairs.iter().map(row))),
    }
}

pub fn export_summaries(pairs: &[SummaryPair], style: Style, system_fallback: &str) -> Result<String, String> {
    let row = |p: &SummaryPair| -> Value {
        match style {
            Style::Hf => json!({ "id": format!("{}-p{}-{}", p.doc_id, p.page, p.id), "article": p.text, "highlights": p.summary }),
            Style::Simple => json!({ "document": p.text, "summary": p.summary }),
            Style::Instruction => json!({ "instruction": SUMMARIZE_INSTRUCTION, "input": p.text, "output": p.summary }),
            _ => chat_messages(if p.system.is_empty() { system_fallback } else { &p.system }, format!("{SUMMARIZE_PROMPT}{}", p.text), &p.summary),
        }
    };
    match style {
        Style::Auto => Err("Pick a concrete style to export (hf, simple or chat).".into()),
        _ => Ok(jsonl(pairs.iter().map(row))),
    }
}

// ---------------------------------------------------------------- import

/// The rows of an uploaded file: JSONL, a JSON array, `{"data"|"rows"|"train": [...]}`, a single
/// object, or the official nested SQuAD layout (flattened to one row per question).
fn records(content: &str) -> Result<Vec<Value>, String> {
    let content = content.trim_start_matches('\u{feff}').trim();
    if content.is_empty() {
        return Err("The file is empty.".into());
    }
    // one JSON document (array / wrapper / single object) first, JSONL otherwise
    if let Ok(v) = serde_json::from_str::<Value>(content) {
        return Ok(match v {
            Value::Array(a) => a,
            Value::Object(mut o) => {
                let wrapped = ["data", "rows", "train"].iter().find_map(|k| o.get(*k).and_then(|x| x.as_array()).cloned());
                match wrapped {
                    Some(a) if a.iter().any(|x| x.get("paragraphs").is_some()) => flatten_squad(&a),
                    Some(a) => a,
                    None => vec![Value::Object(std::mem::take(&mut o))],
                }
            }
            other => return Err(format!("Expected JSON objects, found {}.", other)),
        });
    }
    let mut out = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str::<Value>(line).map_err(|e| format!("Line {} is not valid JSON ({e}).", i + 1))?);
    }
    Ok(out)
}

fn flatten_squad(articles: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for art in articles {
        let title = art.get("title").cloned().unwrap_or(Value::Null);
        for para in art.get("paragraphs").and_then(|p| p.as_array()).into_iter().flatten() {
            let ctx = para.get("context").cloned().unwrap_or(Value::Null);
            for qa in para.get("qas").and_then(|q| q.as_array()).into_iter().flatten() {
                out.push(json!({
                    "id": qa.get("id"), "title": title, "context": ctx,
                    "question": qa.get("question"),
                    "answers": { "text": qa.get("answers").and_then(|a| a.as_array()).map(|a| a.iter().filter_map(|x| x.get("text").cloned()).collect::<Vec<_>>()).unwrap_or_default() },
                }));
            }
        }
    }
    out
}

fn text<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|x| x.as_str()).map(str::trim).filter(|s| !s.is_empty())
}

/// (system, user, assistant) of a `messages` record: the first system and user turns and the last
/// assistant turn (system is empty when absent).
fn chat_turns(v: &Value) -> Option<(&str, &str, &str)> {
    let msgs = v.get("messages")?.as_array()?;
    let role = |r: &'static str| msgs.iter().filter(move |m| m.get("role").and_then(|x| x.as_str()) == Some(r)).filter_map(|m| m.get("content").and_then(|c| c.as_str()));
    let system = role("system").next().unwrap_or("").trim();
    let user = role("user").next()?.trim();
    let asst = role("assistant").last()?.trim();
    (!user.is_empty() && !asst.is_empty()).then_some((system, user, asst))
}

/// First answer of a SQuAD `answers` field: `{"text": [..]}`, or a list of strings / `{text}`.
fn first_answer(v: &Value) -> Option<&str> {
    let a = v.get("answers")?;
    let first = match a {
        Value::Object(_) => a.get("text")?.as_array()?.first()?,
        Value::Array(list) => list.first()?,
        _ => return None,
    };
    first.as_str().or_else(|| first.get("text").and_then(|t| t.as_str())).map(str::trim).filter(|s| !s.is_empty())
}

/// One Q&A record: (context, question, answer, system).
type QaRow = (String, String, String, String);
/// One summary record: (text, summary, system).
type SumRow = (String, String, String);

fn qa_from(v: &Value, style: Style) -> Option<QaRow> {
    let mut system = "";
    let (ctx, q, a): (&str, &str, &str) = match style {
        Style::Chat => {
            let (sys, user, asst) = chat_turns(v)?;
            system = sys;
            match user.strip_prefix("Context:\n").and_then(|r| r.split_once("\n\nQuestion: ")) {
                Some((c, q)) => (c, q, asst),
                None => ("", user, asst),
            }
        }
        Style::Hf => (text(v, "context").unwrap_or(""), text(v, "question")?, first_answer(v)?),
        Style::Instruction => (text(v, "input").unwrap_or(""), text(v, "instruction")?, text(v, "output")?),
        _ => (text(v, "context").unwrap_or(""), text(v, "question")?, text(v, "answer").or_else(|| text(v, "response")).or_else(|| text(v, "output"))?),
    };
    Some((ctx.trim().to_string(), q.trim().to_string(), a.chars().take(MAX_ANSWER_CHARS).collect(), system.to_string()))
}

fn summary_from(v: &Value, style: Style) -> Option<SumRow> {
    let mut system = "";
    let (t, s): (&str, &str) = match style {
        Style::Chat => {
            let (sys, user, asst) = chat_turns(v)?;
            system = sys;
            (user.strip_prefix(SUMMARIZE_PROMPT).unwrap_or(user), asst)
        }
        Style::Hf => (text(v, "article")?, text(v, "highlights")?),
        Style::Instruction => (text(v, "input")?, text(v, "output")?),
        _ => (text(v, "document").or_else(|| text(v, "text"))?, text(v, "summary")?),
    };
    Some((t.chars().take(MAX_TEXT_CHARS).collect(), s.chars().take(MAX_ANSWER_CHARS).collect(), system.to_string()))
}

#[derive(Debug)]
pub struct Imported<T> {
    /// Each record with the style it was read as.
    pub items: Vec<(Style, T)>,
    pub skipped: usize,
}

fn import<T>(content: &str, style: Style, one: impl Fn(&Value, Style) -> Option<T>) -> Result<Imported<T>, String> {
    let rows = records(content)?;
    let styles: &[Style] = match style {
        Style::Auto => &[Style::Chat, Style::Hf, Style::Instruction, Style::Simple],
        Style::Hf => &[Style::Hf],
        Style::Simple => &[Style::Simple],
        Style::Instruction => &[Style::Instruction],
        Style::Chat => &[Style::Chat],
    };
    let mut items = Vec::new();
    let mut skipped = 0;
    for r in &rows {
        match styles.iter().find_map(|s| one(r, *s).map(|x| (*s, x))) {
            Some(x) => items.push(x),
            None => skipped += 1,
        }
    }
    if items.is_empty() {
        let keys = rows.first().and_then(|r| r.as_object()).map(|o| o.keys().cloned().collect::<Vec<_>>().join(", ")).unwrap_or_default();
        let hint = if keys.is_empty() { String::new() } else { format!(" (first record has: {keys})") };
        return Err(format!("No records matched the {style:?} style{hint}."));
    }
    Ok(Imported { items, skipped })
}

/// (context, question, answer, system) per record.
pub fn import_qa(content: &str, style: Style) -> Result<Imported<QaRow>, String> {
    import(content, style, qa_from)
}

/// (text, summary, system) per record.
pub fn import_summaries(content: &str, style: Style) -> Result<Imported<SumRow>, String> {
    import(content, style, summary_from)
}

/// Plain JSONL of `Sample` itself — its JSON shape (`state`, `questions`, `targets`, `source`) is
/// already the canonical format (see `sysone::schema`), so there is no separate "style" to pick.
pub fn export_sysone(samples: &[Sample]) -> String {
    jsonl(samples.iter().map(|s| serde_json::to_value(s).unwrap_or(Value::Null)))
}

/// Parses each record strictly as a `Sample` and validates it; anything that doesn't match (wrong
/// shape, mismatched question/target kinds, bad probabilities) is silently skipped rather than
/// failing the whole import — same convention as `import_qa` / `import_summaries`.
pub fn import_sysone(content: &str) -> Result<(Vec<Sample>, usize), String> {
    let rows = records(content)?;
    let mut items = Vec::new();
    let mut skipped = 0;
    for r in &rows {
        match serde_json::from_value::<Sample>(r.clone()).ok().filter(|s| s.validate().is_ok()) {
            Some(s) => items.push(s),
            None => skipped += 1,
        }
    }
    if items.is_empty() {
        let keys = rows.first().and_then(|r| r.as_object()).map(|o| o.keys().cloned().collect::<Vec<_>>().join(", ")).unwrap_or_default();
        let hint = if keys.is_empty() { String::new() } else { format!(" (first record has: {keys})") };
        return Err(format!("No valid System One samples found{hint}. Expected {{state, questions, targets}} objects."));
    }
    Ok((items, skipped))
}

// ---------------------------------------------------------------- Parquet

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileFormat {
    #[default]
    Jsonl,
    Parquet,
}

impl FileFormat {
    pub fn extension(self) -> &'static str {
        match self {
            FileFormat::Jsonl => "jsonl",
            FileFormat::Parquet => "parquet",
        }
    }
}

/// Parquet files start (and end) with this magic number.
pub fn is_parquet(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && bytes.starts_with(b"PAR1")
}

/// JSONL → Parquet: the schema is inferred from the records (nested `messages` / `answers`
/// become list / struct columns, like Hugging Face's own Parquet exports).
pub fn jsonl_to_parquet(jsonl: &str) -> Result<Vec<u8>, String> {
    use arrow_json::reader::{infer_json_schema_from_iterator, ReaderBuilder};
    let rows: Vec<Value> = jsonl.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>().map_err(|e| e.to_string())?;
    if rows.is_empty() {
        return Err("There is nothing to write.".into());
    }
    let schema = std::sync::Arc::new(infer_json_schema_from_iterator(rows.iter().map(|r| Ok::<_, arrow_schema::ArrowError>(r.clone()))).map_err(|e| e.to_string())?);
    let reader = ReaderBuilder::new(schema.clone()).with_batch_size(8192).build(std::io::Cursor::new(jsonl.as_bytes())).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    let mut w = parquet::arrow::ArrowWriter::try_new(&mut out, schema, None).map_err(|e| e.to_string())?;
    for batch in reader {
        w.write(&batch.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    w.close().map_err(|e| e.to_string())?;
    Ok(out)
}

/// Parquet → JSONL text (one object per row), ready for the JSONL importer.
pub fn parquet_to_jsonl(bytes: bytes::Bytes) -> Result<String, String> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(|e| format!("Not a readable Parquet file ({e})."))?.build().map_err(|e| e.to_string())?;
    let mut w = arrow_json::LineDelimitedWriter::new(Vec::new());
    for batch in reader {
        w.write(&batch.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    w.finish().map_err(|e| e.to_string())?;
    String::from_utf8(w.into_inner()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn qa_pair() -> QaPair {
        QaPair { id: 1, doc_id: "d1".into(), doc: "a.pdf".into(), page: 2, style: Style::Hf, system: String::new(), question: "What is é?".into(), answer: "a letter".into(), context: Arc::from("Here: é is a letter.") }
    }
    fn sum_pair() -> SummaryPair {
        SummaryPair { id: 1, doc_id: "d1".into(), doc: "a.pdf".into(), page: 2, style: Style::Hf, system: String::new(), text: "Long page.".into(), summary: "Short.".into() }
    }

    #[test]
    fn squad_export_has_character_offsets() {
        let out = export_qa(&[qa_pair()], Style::Hf, "").unwrap();
        let v: Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["answers"]["text"][0], "a letter");
        assert_eq!(v["answers"]["answer_start"][0], 11); // char index, "é" counts once
        assert_eq!(v["title"], "a.pdf");
        let mut p = qa_pair();
        p.answer = "not in context".into();
        let v: Value = serde_json::from_str(export_qa(&[p], Style::Hf, "").unwrap().trim()).unwrap();
        assert_eq!(v["answers"]["answer_start"][0], -1);
    }

    #[test]
    fn every_style_round_trips() {
        for style in [Style::Hf, Style::Simple, Style::Instruction, Style::Chat] {
            let got = import_qa(&export_qa(&[qa_pair()], style, "").unwrap(), style).unwrap();
            assert_eq!(got.items, vec![(style, ("Here: é is a letter.".to_string(), "What is é?".to_string(), "a letter".to_string(), String::new()))], "{style:?}");
            let auto = import_qa(&export_qa(&[qa_pair()], style, "").unwrap(), Style::Auto).unwrap();
            assert_eq!(auto.items[0].0, style, "auto detects {style:?}");

            let got = import_summaries(&export_summaries(&[sum_pair()], style, "").unwrap(), style).unwrap();
            assert_eq!(got.items, vec![(style, ("Long page.".to_string(), "Short.".to_string(), String::new()))], "{style:?}");
            assert_eq!(import_summaries(&export_summaries(&[sum_pair()], style, "").unwrap(), Style::Auto).unwrap().items.len(), 1);
        }
    }

    #[test]
    fn sysone_samples_round_trip_and_malformed_lines_are_skipped_not_fatal() {
        use crate::sysone::schema::{Question, Target};
        let good = Sample {
            id: 1,
            state: "Server is down".into(),
            questions: vec![Question::Noul { text: "urgent?".into() }],
            targets: vec![Target::Noul { p: 0.9 }],
            source: "test".into(),
        };
        let exported = export_sysone(&[good.clone()]);
        let (items, skipped) = import_sysone(&exported).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(skipped, 0);
        assert_eq!(items[0].state, "Server is down");
        assert_eq!(items[0].questions, good.questions);
        assert_eq!(items[0].targets, good.targets);

        // a line with mismatched question/target kinds (valid JSON, wrong schema) must be
        // skipped, not fail the batch — syntactically invalid JSON is still a hard error
        // per-line, same as Q&A/summary import (see `wrong_style_and_bad_files_are_explained`).
        let mixed = format!("{exported}{{\"state\":\"x\",\"questions\":[{{\"type\":\"noul\",\"text\":\"q\"}}],\"targets\":[{{\"type\":\"choice\",\"probs\":[0.5,0.5]}}]}}\n");
        let (items, skipped) = import_sysone(&mixed).unwrap();
        assert_eq!(items.len(), 1, "only the one valid sample should survive");
        assert_eq!(skipped, 1);
    }

    #[test]
    fn empty_or_all_invalid_sysone_input_is_an_error() {
        assert!(import_sysone("").is_err());
        assert!(import_sysone("not json\nalso not json").is_err());
    }

    #[test]
    fn imports_the_official_squad_file_and_json_arrays() {
        let squad = r#"{"version":"1.1","data":[{"title":"T","paragraphs":[{"context":"Paris is in France.","qas":[
            {"id":"a","question":"Where is Paris?","answers":[{"text":"France","answer_start":12}]},
            {"id":"b","question":"Unanswerable?","answers":[]}]}]}]}"#;
        let got = import_qa(squad, Style::Hf).unwrap();
        assert_eq!(got.items, vec![(Style::Hf, ("Paris is in France.".to_string(), "Where is Paris?".to_string(), "France".to_string(), String::new()))]);
        assert_eq!(got.skipped, 1, "the unanswerable question has no answer to train on");

        let arr = r#"[{"article":"Body text.","highlights":"Gist."},{"article":"x"}]"#;
        let got = import_summaries(arr, Style::Hf).unwrap();
        assert_eq!((got.items.len(), got.skipped), (1, 1));
    }

    #[test]
    fn wrong_style_and_bad_files_are_explained() {
        let e = import_qa(r#"{"article":"a","highlights":"b"}"#, Style::Hf).err().unwrap();
        assert!(e.contains("article"), "{e}");
        assert!(import_qa("", Style::Auto).is_err());
        assert!(import_qa("{\"question\":\"q\",\"answer\":\"a\"}\nnot json", Style::Simple).unwrap_err().contains("Line 2"));
        assert!(export_qa(&[], Style::Auto, "").is_err());
    }

    #[test]
    fn instruction_style_uses_alpaca_fields() {
        let v: Value = serde_json::from_str(export_qa(&[qa_pair()], Style::Instruction, "").unwrap().trim()).unwrap();
        assert_eq!((v["instruction"].as_str(), v["input"].as_str(), v["output"].as_str()), (Some("What is é?"), Some("Here: é is a letter."), Some("a letter")));
        let v: Value = serde_json::from_str(export_summaries(&[sum_pair()], Style::Instruction, "").unwrap().trim()).unwrap();
        assert_eq!((v["instruction"].as_str(), v["input"].as_str(), v["output"].as_str()), (Some(SUMMARIZE_INSTRUCTION), Some("Long page."), Some("Short.")));
    }

    #[test]
    fn system_prompt_is_written_read_and_falls_back() {
        let mut p = qa_pair();
        // no system anywhere: two turns
        let v: Value = serde_json::from_str(export_qa(&[p.clone()], Style::Chat, "").unwrap().trim()).unwrap();
        assert_eq!(v["messages"].as_array().unwrap().len(), 2);
        // export-time fallback fills an empty one; the pair's own wins over it
        let v: Value = serde_json::from_str(export_qa(&[p.clone()], Style::Chat, "Be brief.").unwrap().trim()).unwrap();
        assert_eq!((v["messages"][0]["role"].as_str(), v["messages"][0]["content"].as_str()), (Some("system"), Some("Be brief.")));
        p.system = "Own prompt.".into();
        let out = export_qa(&[p], Style::Chat, "Be brief.").unwrap();
        assert!(out.contains("Own prompt.") && !out.contains("Be brief."));
        // and it round-trips on import
        let got = import_qa(&out, Style::Chat).unwrap();
        assert_eq!(got.items[0].1 .3, "Own prompt.");
        let s = import_summaries(&export_summaries(&[sum_pair()], Style::Chat, "Sys.").unwrap(), Style::Auto).unwrap();
        assert_eq!(s.items[0].1 .2, "Sys.");
    }

    #[test]
    fn parquet_round_trips_every_style() {
        for style in [Style::Hf, Style::Simple, Style::Instruction, Style::Chat] {
            for jsonl in [export_qa(&[qa_pair(), qa_pair()], style, "Sys.").unwrap(), export_summaries(&[sum_pair()], style, "Sys.").unwrap()] {
                let bytes = jsonl_to_parquet(&jsonl).unwrap();
                assert!(is_parquet(&bytes));
                let back = parquet_to_jsonl(bytes::Bytes::from(bytes)).unwrap();
                let a: Vec<Value> = jsonl.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
                let b: Vec<Value> = back.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
                assert_eq!(a, b, "{style:?}");
            }
        }
        assert!(!is_parquet(b"{\"a\":1}"));
        assert!(parquet_to_jsonl(bytes::Bytes::from_static(b"PAR1 not really parquet")).is_err());
    }
}
