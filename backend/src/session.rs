//! One `Session` per connected browser tab: its uploaded documents, the tokenized dataset built
//! from them, and any trained weights. Also the process-wide `Shared` state (limits, registry).
//!
//! Data flow:  upload (HTTP, streamed to disk) -> extract to UTF-8 text (blocking pool, bounded)
//!             -> [debounced, cancellable] rebuild: sample -> train BPE -> parallel encode
//!             -> `data:info`. Progress is pushed as `data:progress`; the document list as `data:docs`.

use crate::corpus;
use crate::events::{DataInfo, TokenInfo, TokenizerInfo};
use crate::extract::{self, Kind};
use crate::learn::LlmConfig;
use crate::hfdata::Style;
use crate::tokenizer::ImportedTok;
use crate::qa::{self, QaPair};
use crate::summarize::{self, SummaryPair};
use crate::sysone::{self, schema::Sample, train::TrainedSysOne};
use crate::inspect::IB;
use crate::tokenizer::Tokenized;
use crate::train::TrainedSlot;
use serde::Serialize;
use socketioxide::extract::SocketRef;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// A request to generate Q&A pairs from some pages of one document.
pub struct QaJob {
    pub llm: LlmConfig,
    /// 1-based page (section) numbers.
    pub pages: Vec<usize>,
    pub per_page: usize,
    pub style: Style,
    pub system: String,
    pub concurrency: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct QaProgress {
    state: &'static str,
    done: usize,
    total: usize,
    pairs: usize,
    failed: usize,
    message: String,
}

/// A request to generate summaries from some pages of one document.
pub struct SummarizeJob {
    pub llm: LlmConfig,
    /// 1-based page (section) numbers.
    pub pages: Vec<usize>,
    pub sentences: usize,
    pub style: Style,
    pub system: String,
    pub concurrency: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SummarizeProgress {
    state: &'static str,
    done: usize,
    total: usize,
    summaries: usize,
    failed: usize,
    message: String,
}

/// A request to generate System One samples from some pages of one document.
pub struct SysOneGenJob {
    pub llm: LlmConfig,
    /// 1-based page (section) numbers.
    pub pages: Vec<usize>,
    pub per_page: usize,
    pub concurrency: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SysOneGenProgress {
    state: &'static str,
    done: usize,
    total: usize,
    samples: usize,
    failed: usize,
    message: String,
}

/// A request to generate wholly synthetic System One samples — no source document, just a domain
/// and free-form seed context, aimed at a target question-type mix.
pub struct SysOneSynthJob {
    pub llm: LlmConfig,
    pub domain: String,
    pub context: String,
    pub mix: sysone::generate::QuestionMix,
    pub total: usize,
    pub per_batch: usize,
    pub concurrency: usize,
}

/// A request to import rows of a public Hugging Face dataset, converted into Choice-question
/// System One samples (see `sysone::hfimport`).
pub struct SysOneHfImportJob {
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub text_col: String,
    pub label_col: String,
    pub options_per_q: usize,
    pub total: usize,
    pub question_text: String,
}

/// Process-wide limits and registries. All sizes are overridable by environment variables.
pub struct Shared {
    pub trainings: Arc<Semaphore>,
    pub extractions: Arc<Semaphore>,
    pub builds: Arc<Semaphore>,
    /// Concurrent Learn-tab chat requests (each is an outbound HTTP call to a user-configured server).
    pub learns: Arc<Semaphore>,
    /// Concurrent dataset-generation model calls (Q&A and summaries share one budget) across all sessions.
    pub qa_calls: Arc<Semaphore>,
    /// Shared, connection-pooling client for talking to that server.
    pub http: reqwest::Client,
    pub sessions: Mutex<HashMap<String, Arc<Session>>>,
    pub data_dir: PathBuf,
    /// Largest single upload.
    pub max_file_bytes: u64,
    /// Total bytes one session may keep on disk (uploads in flight + extracted text).
    pub max_session_bytes: u64,
    /// Cap on the text extracted from one document.
    pub max_doc_text: u64,
    /// Cap on tokens kept for training (u16 each).
    pub max_tokens: usize,
}

fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

impl Shared {
    pub fn from_env() -> Arc<Shared> {
        let data_dir = std::env::var("DATA_DIR").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join("modelbuilder"));
        // leftovers from a previous run are ours to clear
        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::create_dir_all(&data_dir);
        Arc::new(Shared {
            trainings: Arc::new(Semaphore::new(env_u64("MAX_TRAININGS", 2) as usize)),
            extractions: Arc::new(Semaphore::new(env_u64("MAX_EXTRACTIONS", 2) as usize)),
            builds: Arc::new(Semaphore::new(env_u64("MAX_BUILDS", 2) as usize)),
            learns: Arc::new(Semaphore::new(env_u64("MAX_LEARN_ASKS", 4) as usize)),
            qa_calls: Arc::new(Semaphore::new(env_u64("MAX_QA_CALLS", 16) as usize)),
            http: reqwest::Client::builder().build().expect("build http client"),
            sessions: Mutex::new(HashMap::new()),
            data_dir,
            max_file_bytes: env_u64("MAX_UPLOAD_BYTES", 1 << 30),
            max_session_bytes: env_u64("MAX_SESSION_BYTES", 4 << 30),
            max_doc_text: env_u64("MAX_DOC_TEXT_BYTES", 2 << 30),
            max_tokens: env_u64("MAX_TOKENS", 500_000_000) as usize,
        })
    }

    pub fn session(&self, sid: &str) -> Option<Arc<Session>> {
        self.sessions.lock().unwrap().get(sid).cloned()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum DocState {
    Extracting,
    Ready,
    Error,
}

#[derive(Clone)]
pub struct Doc {
    pub id: String,
    pub name: String,
    pub size: u64,
    pub state: DocState,
    pub kind: Option<Kind>,
    /// Bytes of extracted text.
    pub text_bytes: u64,
    pub note: Option<String>,
    pub error: Option<String>,
    pub enabled: bool,
    pub text_path: Option<PathBuf>,
    /// Start offset of each page / slide in the extracted text (empty for formats without pages).
    pub pages: Vec<u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DocView {
    pub id: String,
    pub name: String,
    pub size: u64,
    pub state: DocState,
    pub kind: Option<&'static str>,
    pub text_bytes: u64,
    pub note: Option<String>,
    pub error: Option<String>,
    pub enabled: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DocsPayload {
    docs: Vec<DocView>,
    total_size: u64,
}

#[derive(Serialize)]
struct ProgressPayload {
    stage: &'static str,
    pct: f32,
    message: String,
}

#[derive(Default)]
pub struct DataState {
    pub docs: Vec<Doc>,
    pub pasted: String,
    pub use_pasted: bool,
    pub merges: usize,
    /// An imported tokenizer.json, used instead of learning the built-in BPE (see `tokenizer::ImportedTok`).
    pub custom_tok: Option<Arc<ImportedTok>>,
    pub tok: Option<Arc<Tokenized>>,
    /// Generated question-answer pairs (see qa.rs).
    pub qa: Vec<QaPair>,
    /// Generated summaries (see summarize.rs).
    pub summaries: Vec<SummaryPair>,
    /// Generated System One samples (see sysone::generate).
    pub sysone_samples: Vec<Sample>,
    next_id: u32,
    next_qa_id: u32,
    next_summary_id: u32,
    next_sysone_id: u32,
}

impl DataState {
    pub fn new_id(&mut self) -> String {
        self.next_id += 1;
        format!("d{}", self.next_id)
    }

    pub fn new_qa_id(&mut self) -> u32 {
        self.next_qa_id += 1;
        self.next_qa_id
    }

    pub fn new_summary_id(&mut self) -> u32 {
        self.next_summary_id += 1;
        self.next_summary_id
    }

    pub fn new_sysone_id(&mut self) -> u32 {
        self.next_sysone_id += 1;
        self.next_sysone_id
    }
}

pub struct Session {
    pub id: String,
    pub dir: PathBuf,
    pub socket: SocketRef,
    pub shared: Arc<Shared>,
    pub data: tokio::sync::Mutex<DataState>,
    pub trained: TrainedSlot,
    pub probe: Arc<Mutex<String>>,
    pub train_cancel: Mutex<Option<Arc<AtomicBool>>>,
    pub gen_cancel: Mutex<Option<Arc<AtomicBool>>>,
    pub qa_cancel: Mutex<Option<Arc<AtomicBool>>>,
    pub summarize_cancel: Mutex<Option<Arc<AtomicBool>>>,
    /// System One sample generation (LLM), separate from the Q&A/Summary ones above.
    pub sysone_cancel: Mutex<Option<Arc<AtomicBool>>>,
    /// System One training — a separate pipeline from the causal model's, so it gets its own
    /// cancel flag and trained-weights slot rather than sharing `train_cancel` / `trained`.
    pub sysone_train_cancel: Mutex<Option<Arc<AtomicBool>>>,
    pub sysone_trained: Mutex<Option<Arc<TrainedSysOne<IB>>>>,
    /// Bumped on every change that needs a new dataset; a rebuild only finishes if it is still current.
    pub gen: AtomicU64,
    build_cancel: Mutex<Arc<AtomicBool>>,
    /// Bytes of uploads + extracted text currently on disk.
    pub disk_used: AtomicU64,
}

impl Session {
    pub fn new(socket: SocketRef, shared: Arc<Shared>) -> Arc<Session> {
        let id: String = socket.id.to_string().chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect();
        let dir = shared.data_dir.join(&id);
        let _ = std::fs::create_dir_all(dir.join("up"));
        let _ = std::fs::create_dir_all(dir.join("txt"));
        let s = Arc::new(Session {
            id: id.clone(),
            dir,
            socket,
            shared: shared.clone(),
            data: Default::default(),
            trained: Arc::new(Mutex::new(None)),
            probe: Arc::new(Mutex::new(String::new())),
            train_cancel: Mutex::new(None),
            gen_cancel: Mutex::new(None),
            qa_cancel: Mutex::new(None),
            summarize_cancel: Mutex::new(None),
            sysone_cancel: Mutex::new(None),
            sysone_train_cancel: Mutex::new(None),
            sysone_trained: Mutex::new(None),
            gen: AtomicU64::new(0),
            build_cancel: Mutex::new(Arc::new(AtomicBool::new(false))),
            disk_used: AtomicU64::new(0),
        });
        shared.sessions.lock().unwrap().insert(id, s.clone());
        s
    }

    /// Stop everything and delete this session's files.
    pub fn close(&self) {
        for slot in [&self.train_cancel, &self.gen_cancel, &self.qa_cancel, &self.summarize_cancel, &self.sysone_cancel, &self.sysone_train_cancel] {
            if let Some(c) = slot.lock().unwrap().as_ref() {
                c.store(true, Ordering::Relaxed);
            }
        }
        self.build_cancel.lock().unwrap().store(true, Ordering::Relaxed);
        self.gen.fetch_add(1, Ordering::SeqCst);
        self.shared.sessions.lock().unwrap().remove(&self.id);
        let _ = std::fs::remove_dir_all(&self.dir);
    }

    pub async fn emit_docs(&self) {
        let d = self.data.lock().await;
        let docs: Vec<DocView> = d
            .docs
            .iter()
            .map(|x| DocView {
                id: x.id.clone(),
                name: x.name.clone(),
                size: x.size,
                state: x.state,
                kind: x.kind.map(|k| k.label()),
                text_bytes: x.text_bytes,
                note: x.note.clone(),
                error: x.error.clone(),
                enabled: x.enabled,
            })
            .collect();
        let total_size = docs.iter().map(|x| x.size).sum();
        let _ = self.socket.emit("data:docs", &DocsPayload { docs, total_size });
    }

    fn progress(&self, stage: &'static str, pct: f32, message: String) {
        let _ = self.socket.emit("data:progress", &ProgressPayload { stage, pct, message });
    }

    pub fn emit_info(&self, info: &DataInfo) {
        let _ = self.socket.emit("data:info", info);
    }

    /// Tells the browser which tokenizer the dataset is (about to be) built with.
    pub async fn emit_tokenizer(&self) {
        let info = match &self.data.lock().await.custom_tok {
            Some(t) => TokenizerInfo { kind: "imported", name: t.name.clone(), model: t.model.clone(), full_vocab: t.full_vocab },
            None => TokenizerInfo { kind: "default", name: String::new(), model: String::new(), full_vocab: 0 },
        };
        let _ = self.socket.emit("data:tokenizer", &info);
    }

    /// Extract a freshly uploaded file, then rebuild the dataset.
    pub fn spawn_extraction(self: &Arc<Self>, id: String, raw: PathBuf, name: String) {
        let s = self.clone();
        tokio::spawn(async move {
            let Ok(_permit) = s.shared.extractions.clone().acquire_owned().await else { return };
            let out = s.dir.join("txt").join(format!("{id}.txt"));
            let scratch = s.dir.join("scratch").join(&id);
            let (cap, o2, s2) = (s.shared.max_doc_text, out.clone(), scratch.clone());
            let res = tokio::task::spawn_blocking(move || {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| extract::extract_to(&raw, &name, &o2, &s2, cap)));
                let _ = std::fs::remove_file(&raw);
                let _ = std::fs::remove_dir_all(&s2);
                r.unwrap_or_else(|_| Err("The extractor crashed on this file.".to_string()))
            })
            .await
            .unwrap_or_else(|e| Err(format!("extraction failed: {e}")));

            {
                let mut d = s.data.lock().await;
                if let Some(doc) = d.docs.iter_mut().find(|x| x.id == id) {
                    match res {
                        Ok(o) => {
                            s.disk_used.fetch_add(o.bytes, Ordering::Relaxed);
                            doc.state = DocState::Ready;
                            doc.kind = Some(o.kind);
                            doc.text_bytes = o.bytes;
                            doc.note = o.note;
                            doc.text_path = Some(out);
                            doc.pages = o.pages;
                        }
                        Err(e) => {
                            let _ = std::fs::remove_file(&out);
                            doc.state = DocState::Error;
                            doc.error = Some(e);
                        }
                    }
                }
            }
            s.emit_docs().await;
            s.schedule_rebuild();
        });
    }

    pub async fn emit_qa_list(&self) {
        let d = self.data.lock().await;
        let _ = self.socket.emit("qa:list", &d.qa);
    }

    /// Generate Q&A pairs for the chosen pages: one model request per page, up to
    /// `job.concurrency` at once (and never more than the server-wide limit). Progress is pushed
    /// as `qa:progress`; new pairs as `qa:added`. Cancelled through `cancel`.
    pub fn spawn_qa(self: &Arc<Self>, job: QaJob, cancel: Arc<AtomicBool>, doc: Doc) {
        let s = self.clone();
        tokio::spawn(async move {
            let total = job.pages.len();
            let emit = |state: &'static str, done: usize, pairs: usize, failed: usize, message: String| {
                let _ = s.socket.emit("qa:progress", &QaProgress { state, done, total, pairs, failed, message });
            };
            let (Some(path), total_bytes) = (doc.text_path.clone(), doc.text_bytes) else {
                emit("error", 0, 0, 0, "That document has no readable text.".into());
                *s.qa_cancel.lock().unwrap() = None;
                return;
            };
            let ranges = match qa::page_ranges(&path, &doc.pages, total_bytes) {
                Ok((r, _)) => Arc::new(r),
                Err(e) => {
                    emit("error", 0, 0, 0, format!("Could not read the document: {e}"));
                    *s.qa_cancel.lock().unwrap() = None;
                    return;
                }
            };
            emit("running", 0, 0, 0, String::new());

            let local = Arc::new(Semaphore::new(job.concurrency.max(1)));
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Result<(std::sync::Arc<str>, Vec<(String, String)>), String>)>();
            for &n in &job.pages {
                let (local, global, client, llm, cancel, tx) = (local.clone(), s.shared.qa_calls.clone(), s.shared.http.clone(), job.llm.clone(), cancel.clone(), tx.clone());
                let (ranges, path, name, per_page) = (ranges.clone(), path.clone(), doc.name.clone(), job.per_page);
                let style = job.style;
                tokio::spawn(async move {
                    let Ok(_l) = local.acquire_owned().await else { return };
                    let Ok(_g) = global.acquire_owned().await else { return };
                    if cancel.load(Ordering::Relaxed) {
                        let _ = tx.send((n, Err("cancelled".into())));
                        return;
                    }
                    let res = match ranges.get(n.wrapping_sub(1)) {
                        None => Err(format!("page {n} does not exist")),
                        Some(&(a, b)) => match qa::read_range(&path, a, b) {
                            Err(e) => Err(format!("could not read page {n}: {e}")),
                            Ok(text) if text.chars().count() < qa::MIN_PAGE_CHARS => Err(format!("page {n} has almost no text")),
                            Ok(text) => qa::generate_page(&client, &llm, &name, n, &text, per_page, style).await.map(|pairs| (std::sync::Arc::<str>::from(text.chars().take(qa::MAX_PAGE_CHARS).collect::<String>()), pairs)),
                        },
                    };
                    let _ = tx.send((n, res));
                });
            }
            drop(tx);

            let (mut done, mut pairs, mut failed) = (0usize, 0usize, 0usize);
            let mut first_err: Option<String> = None;
            let mut fatal = false;
            while let Some((n, res)) = rx.recv().await {
                match res {
                    Ok((context, list)) => {
                        let mut added = Vec::new();
                        {
                            let mut d = s.data.lock().await;
                            for (question, answer) in list {
                                let key = question.to_lowercase();
                                if d.qa.len() >= qa::MAX_PAIRS || d.qa.iter().any(|p| p.doc_id == doc.id && p.style == job.style && p.question.to_lowercase() == key) {
                                    continue;
                                }
                                let p = QaPair { id: d.new_qa_id(), doc_id: doc.id.clone(), doc: doc.name.clone(), page: n, style: job.style, system: job.system.clone(), question, answer, context: context.clone() };
                                d.qa.push(p.clone());
                                added.push(p);
                            }
                        }
                        pairs += added.len();
                        let _ = s.socket.emit("qa:added", &added);
                    }
                    Err(e) if e == "cancelled" => {}
                    Err(e) => {
                        failed += 1;
                        // a bad key / URL fails every page the same way: stop early
                        if e.contains("API key") || e.contains("Settings") || e.contains("(404)") {
                            fatal = true;
                            cancel.store(true, Ordering::Relaxed);
                        }
                        first_err.get_or_insert(format!("page {n}: {e}"));
                    }
                }
                done += 1;
                emit("running", done, pairs, failed, first_err.clone().unwrap_or_default());
            }
            let stopped = cancel.load(Ordering::Relaxed) && !fatal;
            {
                let mut d = s.data.lock().await;
                d.qa.sort_by(|a, b| (&a.doc_id, a.page, a.id).cmp(&(&b.doc_id, b.page, b.id)));
            }
            s.emit_qa_list().await;
            *s.qa_cancel.lock().unwrap() = None;
            let state = if fatal { "error" } else if stopped { "stopped" } else { "done" };
            emit(state, done, pairs, failed, first_err.unwrap_or_default());
        });
    }

    pub async fn emit_summary_list(&self) {
        let d = self.data.lock().await;
        let _ = self.socket.emit("summarize:list", &d.summaries);
    }

    /// Generate summaries for the chosen pages: one model request per page, up to
    /// `job.concurrency` at once (and never more than the server-wide limit). Progress is pushed
    /// as `summarize:progress`; new summaries as `summarize:added`. Cancelled through `cancel`.
    /// Re-summarizing a page replaces its earlier summary rather than adding another.
    pub fn spawn_summarize(self: &Arc<Self>, job: SummarizeJob, cancel: Arc<AtomicBool>, doc: Doc) {
        let s = self.clone();
        tokio::spawn(async move {
            let total = job.pages.len();
            let emit = |state: &'static str, done: usize, summaries: usize, failed: usize, message: String| {
                let _ = s.socket.emit("summarize:progress", &SummarizeProgress { state, done, total, summaries, failed, message });
            };
            let (Some(path), total_bytes) = (doc.text_path.clone(), doc.text_bytes) else {
                emit("error", 0, 0, 0, "That document has no readable text.".into());
                *s.summarize_cancel.lock().unwrap() = None;
                return;
            };
            let ranges = match qa::page_ranges(&path, &doc.pages, total_bytes) {
                Ok((r, _)) => Arc::new(r),
                Err(e) => {
                    emit("error", 0, 0, 0, format!("Could not read the document: {e}"));
                    *s.summarize_cancel.lock().unwrap() = None;
                    return;
                }
            };
            emit("running", 0, 0, 0, String::new());

            let local = Arc::new(Semaphore::new(job.concurrency.max(1)));
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Result<(String, String), String>)>();
            for &n in &job.pages {
                let (local, global, client, llm, cancel, tx) = (local.clone(), s.shared.qa_calls.clone(), s.shared.http.clone(), job.llm.clone(), cancel.clone(), tx.clone());
                let (ranges, path, name, sentences) = (ranges.clone(), path.clone(), doc.name.clone(), job.sentences);
                let style = job.style;
                tokio::spawn(async move {
                    let Ok(_l) = local.acquire_owned().await else { return };
                    let Ok(_g) = global.acquire_owned().await else { return };
                    if cancel.load(Ordering::Relaxed) {
                        let _ = tx.send((n, Err("cancelled".into())));
                        return;
                    }
                    let res = match ranges.get(n.wrapping_sub(1)) {
                        None => Err(format!("page {n} does not exist")),
                        Some(&(a, b)) => match qa::read_range(&path, a, b) {
                            Err(e) => Err(format!("could not read page {n}: {e}")),
                            Ok(text) if text.chars().count() < qa::MIN_PAGE_CHARS => Err(format!("page {n} has almost no text")),
                            Ok(text) => summarize::generate_page(&client, &llm, &name, n, &text, sentences, style).await,
                        },
                    };
                    let _ = tx.send((n, res));
                });
            }
            drop(tx);

            let (mut done, mut made, mut failed) = (0usize, 0usize, 0usize);
            let mut first_err: Option<String> = None;
            let mut fatal = false;
            while let Some((n, res)) = rx.recv().await {
                match res {
                    Ok((text, summary)) => {
                        let added = {
                            let mut d = s.data.lock().await;
                            if d.summaries.len() >= summarize::MAX_PAIRS {
                                None
                            } else {
                                d.summaries.retain(|p| !(p.doc_id == doc.id && p.page == n && p.style == job.style));
                                let p = SummaryPair { id: d.new_summary_id(), doc_id: doc.id.clone(), doc: doc.name.clone(), page: n, style: job.style, system: job.system.clone(), text, summary };
                                d.summaries.push(p.clone());
                                Some(p)
                            }
                        };
                        if let Some(p) = added {
                            made += 1;
                            // a fixed-size array here would be encoded as a bare (spread) arg, not
                            // a one-element JSON array (socketioxide treats compile-time-length
                            // sequences as tuple-like) — a Vec always stays wrapped as one array.
                            let _ = s.socket.emit("summarize:added", &vec![p]);
                        }
                    }
                    Err(e) if e == "cancelled" => {}
                    Err(e) => {
                        failed += 1;
                        if e.contains("API key") || e.contains("Settings") || e.contains("(404)") {
                            fatal = true;
                            cancel.store(true, Ordering::Relaxed);
                        }
                        first_err.get_or_insert(format!("page {n}: {e}"));
                    }
                }
                done += 1;
                emit("running", done, made, failed, first_err.clone().unwrap_or_default());
            }
            let stopped = cancel.load(Ordering::Relaxed) && !fatal;
            {
                let mut d = s.data.lock().await;
                d.summaries.sort_by(|a, b| (&a.doc_id, a.page, a.id).cmp(&(&b.doc_id, b.page, b.id)));
            }
            s.emit_summary_list().await;
            *s.summarize_cancel.lock().unwrap() = None;
            let state = if fatal { "error" } else if stopped { "stopped" } else { "done" };
            emit(state, done, made, failed, first_err.unwrap_or_default());
        });
    }

    pub async fn emit_sysone_list(&self) {
        let d = self.data.lock().await;
        let _ = self.socket.emit("sysone:list", &d.sysone_samples);
    }

    /// Generate System One samples for the chosen pages: one model request per page (each can
    /// return several samples), up to `job.concurrency` at once. Progress is pushed as
    /// `sysone:progress`; new samples as `sysone:added`. Cancelled through `cancel`.
    pub fn spawn_sysone_generate(self: &Arc<Self>, job: SysOneGenJob, cancel: Arc<AtomicBool>, doc: Doc) {
        let s = self.clone();
        tokio::spawn(async move {
            let total = job.pages.len();
            let emit = |state: &'static str, done: usize, samples: usize, failed: usize, message: String| {
                let _ = s.socket.emit("sysone:progress", &SysOneGenProgress { state, done, total, samples, failed, message });
            };
            let (Some(path), total_bytes) = (doc.text_path.clone(), doc.text_bytes) else {
                emit("error", 0, 0, 0, "That document has no readable text.".into());
                *s.sysone_cancel.lock().unwrap() = None;
                return;
            };
            let ranges = match qa::page_ranges(&path, &doc.pages, total_bytes) {
                Ok((r, _)) => Arc::new(r),
                Err(e) => {
                    emit("error", 0, 0, 0, format!("Could not read the document: {e}"));
                    *s.sysone_cancel.lock().unwrap() = None;
                    return;
                }
            };
            emit("running", 0, 0, 0, String::new());

            let local = Arc::new(Semaphore::new(job.concurrency.max(1)));
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Result<Vec<Sample>, String>)>();
            for &n in &job.pages {
                let (local, global, client, llm, cancel, tx) = (local.clone(), s.shared.qa_calls.clone(), s.shared.http.clone(), job.llm.clone(), cancel.clone(), tx.clone());
                let (ranges, path, name, per_page) = (ranges.clone(), path.clone(), doc.name.clone(), job.per_page);
                tokio::spawn(async move {
                    let Ok(_l) = local.acquire_owned().await else { return };
                    let Ok(_g) = global.acquire_owned().await else { return };
                    if cancel.load(Ordering::Relaxed) {
                        let _ = tx.send((n, Err("cancelled".into())));
                        return;
                    }
                    let res = match ranges.get(n.wrapping_sub(1)) {
                        None => Err(format!("page {n} does not exist")),
                        Some(&(a, b)) => match qa::read_range(&path, a, b) {
                            Err(e) => Err(format!("could not read page {n}: {e}")),
                            Ok(text) if text.chars().count() < qa::MIN_PAGE_CHARS => Err(format!("page {n} has almost no text")),
                            Ok(text) => sysone::generate::generate_page(&client, &llm, &name, n, &text, per_page).await,
                        },
                    };
                    let _ = tx.send((n, res));
                });
            }
            drop(tx);

            let (mut done, mut made, mut failed) = (0usize, 0usize, 0usize);
            let mut first_err: Option<String> = None;
            let mut fatal = false;
            while let Some((n, res)) = rx.recv().await {
                match res {
                    Ok(list) => {
                        let mut added = Vec::new();
                        {
                            let mut d = s.data.lock().await;
                            for mut sample in list {
                                if d.sysone_samples.len() >= sysone::generate::MAX_SAMPLES {
                                    break;
                                }
                                sample.id = d.new_sysone_id();
                                d.sysone_samples.push(sample.clone());
                                added.push(sample);
                            }
                        }
                        made += added.len();
                        let _ = s.socket.emit("sysone:added", &added);
                    }
                    Err(e) if e == "cancelled" => {}
                    Err(e) => {
                        failed += 1;
                        if e.contains("API key") || e.contains("Settings") || e.contains("(404)") {
                            fatal = true;
                            cancel.store(true, Ordering::Relaxed);
                        }
                        first_err.get_or_insert(format!("page {n}: {e}"));
                    }
                }
                done += 1;
                emit("running", done, made, failed, first_err.clone().unwrap_or_default());
            }
            let stopped = cancel.load(Ordering::Relaxed) && !fatal;
            s.emit_sysone_list().await;
            *s.sysone_cancel.lock().unwrap() = None;
            let state = if fatal { "error" } else if stopped { "stopped" } else { "done" };
            emit(state, done, made, failed, first_err.unwrap_or_default());
        });
    }

    /// Generate wholly synthetic System One samples (no source document): `job.total` samples
    /// split into batches of `job.per_batch`, up to `job.concurrency` model calls at once.
    /// Progress and new samples are pushed on the same `sysone:progress` / `sysone:added` events
    /// as page-grounded generation — the two are just different ways to fill the same list.
    pub fn spawn_sysone_synth(self: &Arc<Self>, job: SysOneSynthJob, cancel: Arc<AtomicBool>) {
        let s = self.clone();
        tokio::spawn(async move {
            let per_batch = job.per_batch.max(1);
            let mut sizes: Vec<usize> = vec![per_batch; job.total / per_batch];
            if job.total % per_batch != 0 {
                sizes.push(job.total % per_batch);
            }
            let total = sizes.len();
            let emit = |state: &'static str, done: usize, samples: usize, failed: usize, message: String| {
                let _ = s.socket.emit("sysone:progress", &SysOneGenProgress { state, done, total, samples, failed, message });
            };
            emit("running", 0, 0, 0, String::new());

            let local = Arc::new(Semaphore::new(job.concurrency.max(1)));
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Result<Vec<Sample>, String>)>();
            for (i, &batch_n) in sizes.iter().enumerate() {
                let (local, global, client, llm, cancel, tx) = (local.clone(), s.shared.qa_calls.clone(), s.shared.http.clone(), job.llm.clone(), cancel.clone(), tx.clone());
                let (domain, context, mix) = (job.domain.clone(), job.context.clone(), job.mix);
                tokio::spawn(async move {
                    let Ok(_l) = local.acquire_owned().await else { return };
                    let Ok(_g) = global.acquire_owned().await else { return };
                    let res = if cancel.load(Ordering::Relaxed) {
                        Err("cancelled".into())
                    } else {
                        sysone::generate::generate_synthetic(&client, &llm, &domain, &context, batch_n, &mix).await
                    };
                    let _ = tx.send((i, res));
                });
            }
            drop(tx);

            let (mut done, mut made, mut failed) = (0usize, 0usize, 0usize);
            let mut first_err: Option<String> = None;
            let mut fatal = false;
            while let Some((i, res)) = rx.recv().await {
                match res {
                    Ok(list) => {
                        let mut added = Vec::new();
                        {
                            let mut d = s.data.lock().await;
                            for mut sample in list {
                                if d.sysone_samples.len() >= sysone::generate::MAX_SAMPLES {
                                    break;
                                }
                                sample.id = d.new_sysone_id();
                                d.sysone_samples.push(sample.clone());
                                added.push(sample);
                            }
                        }
                        made += added.len();
                        let _ = s.socket.emit("sysone:added", &added);
                    }
                    Err(e) if e == "cancelled" => {}
                    Err(e) => {
                        failed += 1;
                        if e.contains("API key") || e.contains("Settings") || e.contains("(404)") {
                            fatal = true;
                            cancel.store(true, Ordering::Relaxed);
                        }
                        first_err.get_or_insert(format!("batch {}: {e}", i + 1));
                    }
                }
                done += 1;
                emit("running", done, made, failed, first_err.clone().unwrap_or_default());
            }
            let stopped = cancel.load(Ordering::Relaxed) && !fatal;
            s.emit_sysone_list().await;
            *s.sysone_cancel.lock().unwrap() = None;
            let state = if fatal { "error" } else if stopped { "stopped" } else { "done" };
            emit(state, done, made, failed, first_err.unwrap_or_default());
        });
    }

    /// Fetches a slice of a public Hugging Face dataset and converts it into Choice-question
    /// samples (see `sysone::hfimport`). Same `sysone:progress` / `sysone:added` events as the
    /// other two generation modes, just with coarser (fetch / convert) progress steps since this
    /// is one sequential pipeline rather than many parallel model calls.
    pub fn spawn_sysone_hf_import(self: &Arc<Self>, job: SysOneHfImportJob, cancel: Arc<AtomicBool>) {
        let s = self.clone();
        tokio::spawn(async move {
            // +1 so the final convert/store step visibly moves the bar past "all rows fetched".
            let total_steps = job.total + 1;
            let emit = |state: &'static str, done: usize, samples: usize, message: String| {
                let _ = s.socket.emit("sysone:progress", &SysOneGenProgress { state, done, total: total_steps, samples, failed: 0, message });
            };
            let bail = |msg: String| {
                let _ = s.socket.emit("sysone:progress", &SysOneGenProgress { state: "error", done: 0, total: total_steps, samples: 0, failed: 1, message: msg });
            };
            emit("running", 0, 0, format!("Looking up {}…", job.dataset));

            let info = match sysone::hfimport::fetch_info(&s.shared.http, &job.dataset).await {
                Ok(i) => i,
                Err(e) => {
                    bail(e);
                    *s.sysone_cancel.lock().unwrap() = None;
                    return;
                }
            };
            let class_names = match info.columns.iter().find(|c| c.name == job.label_col).and_then(|c| c.class_names.clone()) {
                Some(names) if !names.is_empty() => names,
                _ => {
                    bail(format!("Column \"{}\" doesn't look like a classification label column.", job.label_col));
                    *s.sysone_cancel.lock().unwrap() = None;
                    return;
                }
            };
            if cancel.load(Ordering::Relaxed) {
                emit("stopped", 0, 0, String::new());
                *s.sysone_cancel.lock().unwrap() = None;
                return;
            }

            let rows = match sysone::hfimport::fetch_rows(&s.shared.http, &job.dataset, &job.config, &job.split, job.total, |done, total| {
                emit("running", done, 0, format!("Fetching rows… ({done}/{total})"));
            })
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    bail(e);
                    *s.sysone_cancel.lock().unwrap() = None;
                    return;
                }
            };
            if cancel.load(Ordering::Relaxed) {
                emit("stopped", rows.len(), 0, String::new());
                *s.sysone_cancel.lock().unwrap() = None;
                return;
            }

            let source = format!("hf: {}", job.dataset);
            let question_text = if job.question_text.trim().is_empty() { "which category best fits this?".to_string() } else { job.question_text.clone() };
            let converted = sysone::hfimport::rows_to_samples(&rows, &job.text_col, &job.label_col, &class_names, job.options_per_q, &question_text, &source, 42);

            let mut added = Vec::new();
            {
                let mut d = s.data.lock().await;
                for mut sample in converted {
                    if d.sysone_samples.len() >= sysone::generate::MAX_SAMPLES {
                        break;
                    }
                    sample.id = d.new_sysone_id();
                    d.sysone_samples.push(sample.clone());
                    added.push(sample);
                }
            }
            let made = added.len();
            let _ = s.socket.emit("sysone:added", &added);
            s.emit_sysone_list().await;
            *s.sysone_cancel.lock().unwrap() = None;
            let msg = if made == 0 { format!("Fetched {} rows but converted none — check the column choices.", rows.len()) } else { String::new() };
            emit("done", total_steps, made, msg);
        });
    }

    /// Trains a System One model on the session's generated samples. A separate pipeline from
    /// the causal model's `train:start` (different architecture, loss and batch shape), so it
    /// gets its own status/step/eval events (`sysone:train:*`) and trained-weights slot.
    pub fn spawn_sysone_train(self: &Arc<Self>, samples: Vec<Sample>, hyper: sysone::train::TrainHyper, cancel: Arc<AtomicBool>) {
        let s = self.clone();
        tokio::spawn(async move {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<sysone::train::Event>();
            let device = Default::default();
            let (s2, cancel2) = (s.clone(), cancel.clone());
            let handle = tokio::task::spawn_blocking(move || {
                type TB = burn::backend::Autodiff<IB>;
                sysone::train::run::<TB>(samples, hyper, device, cancel2, tx)
            });

            while let Some(ev) = rx.recv().await {
                let _ = match ev {
                    sysone::train::Event::Status(p) => s2.socket.emit("sysone:train:status", &p),
                    sysone::train::Event::Step(p) => s2.socket.emit("sysone:train:step", &p),
                    sysone::train::Event::Eval(p) => s2.socket.emit("sysone:train:eval", &p),
                };
            }
            if let Ok(Some(trained)) = handle.await {
                *s2.sysone_trained.lock().unwrap() = Some(Arc::new(trained));
            }
            *s2.sysone_train_cancel.lock().unwrap() = None;
        });
    }

    /// (Re)build the tokenized dataset from the current sources. Debounced, and supersedes any
    /// build already running. Waits while documents are still being extracted (their completion
    /// calls this again).
    pub fn schedule_rebuild(self: &Arc<Self>) {
        let my = self.gen.fetch_add(1, Ordering::SeqCst) + 1;
        let cancel = Arc::new(AtomicBool::new(false));
        std::mem::replace(&mut *self.build_cancel.lock().unwrap(), cancel.clone()).store(true, Ordering::Relaxed);
        let s = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(350)).await;
            if s.gen.load(Ordering::SeqCst) != my {
                return;
            }
            let (files, merges, custom) = {
                let d = s.data.lock().await;
                if d.docs.iter().any(|x| x.state == DocState::Extracting) {
                    return;
                }
                let mut files = Vec::new();
                if d.use_pasted && !d.pasted.trim().is_empty() {
                    let p = s.dir.join("pasted.txt");
                    if std::fs::write(&p, &d.pasted).is_ok() {
                        files.push(p);
                    }
                }
                files.extend(d.docs.iter().filter(|x| x.enabled && x.state == DocState::Ready).filter_map(|x| x.text_path.clone()));
                (files, d.merges, d.custom_tok.clone())
            };
            let sources = files.len();
            if files.is_empty() {
                s.data.lock().await.tok = None;
                s.progress("done", 1.0, String::new());
                s.emit_info(&DataInfo::failed("There is no text to train on yet. Paste some text or upload a file."));
                return;
            }
            let Ok(_permit) = s.shared.builds.clone().acquire_owned().await else { return };
            if s.gen.load(Ordering::SeqCst) != my {
                return;
            }
            let started = Instant::now();
            s.progress("tokenizer", 0.0, "Preparing…".into());
            let (sock, last) = (s.socket.clone(), AtomicU32::new(u32::MAX));
            let cfg = corpus::Cfg { merges, max_tokens: s.shared.max_tokens, custom, ..Default::default() };
            let res = tokio::task::spawn_blocking(move || {
                let report = |stage: &'static str, pct: f32, message: String| {
                    // at most one event per percent
                    let key = (pct * 100.0) as u32 + if stage == "encode" { 1000 } else { 0 };
                    if last.swap(key, Ordering::Relaxed) != key {
                        let _ = sock.emit("data:progress", &ProgressPayload { stage, pct, message });
                    }
                };
                corpus::build(&files, &cfg, &cancel, &report)
            })
            .await;
            if s.gen.load(Ordering::SeqCst) != my {
                return; // superseded while building
            }
            s.progress("done", 1.0, String::new());
            match res {
                Ok(Ok(t)) => {
                    let t = Arc::new(t);
                    let info = data_info(&t, merges, sources, started.elapsed());
                    s.data.lock().await.tok = Some(t);
                    s.emit_info(&info);
                }
                Ok(Err(e)) if e == "cancelled" => {}
                Ok(Err(e)) => s.emit_info(&DataInfo::failed(e)),
                Err(e) => s.emit_info(&DataInfo::failed(format!("tokenization crashed: {e}"))),
            }
        });
    }
}

pub fn data_info(tok: &Tokenized, merges: usize, sources: usize, took: Duration) -> DataInfo {
    DataInfo {
        ok: true,
        error: None,
        chars: tok.chars,
        tokens: tok.ids.len(),
        vocab_size: tok.bpe.vocab_size(),
        full_vocab: tok.bpe.imported().map(|c| c.tok.full_vocab).unwrap_or(0),
        merges: tok.bpe.n_merges().min(merges),
        compression: if tok.ids.is_empty() { 0.0 } else { tok.chars as f32 / tok.ids.len() as f32 },
        preview: tok.ids.iter().take(48).map(|&id| TokenInfo { id: id as u32, text: tok.bpe.token_text(id as u32) }).collect(),
        sources,
        bytes: tok.bytes,
        truncated: tok.truncated,
        build_ms: took.as_millis() as u64,
    }
}
