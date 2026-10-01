//! `PUT /api/upload?sid=<socket id>&name=<file name>` — streams a file (up to `MAX_UPLOAD_BYTES`,
//! 1 GiB by default) straight to disk, then hands it to the extractor. The body is never
//! buffered in memory, so a 1 GB upload costs a few megabytes of RAM.

use crate::session::{Doc, DocState, Shared};
use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

#[derive(Deserialize)]
pub struct UploadQuery {
    sid: String,
    name: String,
}

fn fail(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "ok": false, "error": msg.into() }))).into_response()
}

fn mb(b: u64) -> String {
    format!("{:.0} MB", b as f64 / (1u64 << 20) as f64)
}

/// Removes a half-written upload if the client disconnects or an error occurs.
struct Partial(Option<std::path::PathBuf>);
impl Drop for Partial {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = std::fs::remove_file(p);
        }
    }
}

pub async fn upload(State(shared): State<Arc<Shared>>, Query(q): Query<UploadQuery>, headers: axum::http::HeaderMap, body: Body) -> Response {
    let Some(session) = shared.session(&q.sid) else {
        return fail(StatusCode::NOT_FOUND, "Your session expired. Reload the page and try again.");
    };
    let name: String = q.name.rsplit(['/', '\\']).next().unwrap_or("file").chars().filter(|c| !c.is_control()).take(200).collect();
    let name = if name.trim().is_empty() { "file".to_string() } else { name };

    // reject early when the size is announced
    let declared = headers.get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok());
    if let Some(n) = declared {
        if n > shared.max_file_bytes {
            return fail(StatusCode::PAYLOAD_TOO_LARGE, format!("{name} is {}; the limit is {} per file.", mb(n), mb(shared.max_file_bytes)));
        }
        if session.disk_used.load(Ordering::Relaxed) + n > shared.max_session_bytes {
            return fail(StatusCode::INSUFFICIENT_STORAGE, format!("This session is at its {} storage limit. Remove some documents first.", mb(shared.max_session_bytes)));
        }
    }

    let id = session.data.lock().await.new_id();
    let path = session.dir.join("up").join(format!("{id}.upload"));
    let mut guard = Partial(Some(path.clone()));
    let file = match tokio::fs::File::create(&path).await {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("cannot store the upload: {e}")),
    };
    let mut w = tokio::io::BufWriter::with_capacity(1 << 20, file);
    let mut stream = body.into_data_stream();
    let mut written: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(_) => return fail(StatusCode::BAD_REQUEST, "The upload was interrupted."),
        };
        written += chunk.len() as u64;
        if written > shared.max_file_bytes {
            return fail(StatusCode::PAYLOAD_TOO_LARGE, format!("{name} is larger than the {} per-file limit.", mb(shared.max_file_bytes)));
        }
        if session.disk_used.load(Ordering::Relaxed) + written > shared.max_session_bytes {
            return fail(StatusCode::INSUFFICIENT_STORAGE, "This session's storage limit was reached.");
        }
        if let Err(e) = w.write_all(&chunk).await {
            return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("disk write failed: {e}"));
        }
    }
    if let Err(e) = w.flush().await {
        return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("disk write failed: {e}"));
    }
    drop(w);
    if written == 0 {
        return fail(StatusCode::BAD_REQUEST, format!("{name} is empty."));
    }
    guard.0 = None; // keep the file: the extractor owns (and deletes) it now

    {
        let mut d = session.data.lock().await;
        d.docs.push(Doc {
            id: id.clone(), name: name.clone(), size: written, state: DocState::Extracting, kind: None, text_bytes: 0,
            note: None, error: None, enabled: true, text_path: None, pages: Vec::new(),
        });
    }
    session.emit_docs().await;
    session.spawn_extraction(id.clone(), path, name);
    (StatusCode::OK, Json(json!({ "ok": true, "id": id, "size": written }))).into_response()
}

#[derive(Deserialize)]
pub struct ImportQuery {
    sid: String,
    name: String,
    kind: crate::hfdata::Kind,
    style: crate::hfdata::Style,
}

/// `PUT /api/import?sid=&name=&kind=qa|summary&style=auto|hf|simple|chat` — a Hugging Face-style
/// JSON / JSONL file whose records become Q&A pairs or summaries (see `hfdata`). Read into
/// memory (capped by `MAX_IMPORT_BYTES`, 64 MiB): records are small and get parsed anyway.
pub async fn import_dataset(State(shared): State<Arc<Shared>>, Query(q): Query<ImportQuery>, body: Body) -> Response {
    use crate::hfdata::{self, Kind};
    let Some(session) = shared.session(&q.sid) else {
        return fail(StatusCode::NOT_FOUND, "Your session expired. Reload the page and try again.");
    };
    let limit: usize = std::env::var("MAX_IMPORT_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(64 << 20);
    let bytes = match axum::body::to_bytes(body, limit).await {
        Ok(b) => b,
        Err(_) => return fail(StatusCode::PAYLOAD_TOO_LARGE, format!("Dataset files can be at most {}.", mb(limit as u64))),
    };
    let parquet = crate::hfdata::is_parquet(&bytes);
    let content = if parquet {
        match tokio::task::spawn_blocking({ let b = bytes.clone(); move || crate::hfdata::parquet_to_jsonl(b) }).await {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => return fail(StatusCode::UNPROCESSABLE_ENTITY, e),
            Err(_) => return fail(StatusCode::INTERNAL_SERVER_ERROR, "Reading the Parquet file failed."),
        }
    } else {
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let name: String = q.name.rsplit(['/', '\\']).next().unwrap_or("import").chars().filter(|c| !c.is_control()).take(200).collect();
    let doc_id = format!("import:{name}");
    let (kind, style) = (q.kind, q.style);

    // `QaRow` / `SumRow` are private type aliases in `hfdata`, so the tuple shapes are spelled out here.
    enum Parsed {
        Qa(Vec<(crate::hfdata::Style, (String, String, String, String))>),
        Summary(Vec<(crate::hfdata::Style, (String, String, String))>),
        SysOne(Vec<crate::sysone::schema::Sample>),
    }
    let parsed = tokio::task::spawn_blocking(move || match kind {
        Kind::Qa => hfdata::import_qa(&content, style).map(|i| (i.skipped, Parsed::Qa(i.items))),
        Kind::Summary => hfdata::import_summaries(&content, style).map(|i| (i.skipped, Parsed::Summary(i.items))),
        Kind::SysOne => hfdata::import_sysone(&content).map(|(items, skipped)| (skipped, Parsed::SysOne(items))),
    })
    .await;
    let (skipped, parsed) = match parsed {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return fail(StatusCode::UNPROCESSABLE_ENTITY, e),
        Err(_) => return fail(StatusCode::INTERNAL_SERVER_ERROR, "Reading the file failed."),
    };

    let mut added = 0usize;
    let mut capped = false;
    {
        let mut d = session.data.lock().await;
        match parsed {
            Parsed::Qa(items) => {
                for (style, (ctx, question, answer, system)) in items {
                    if d.qa.len() >= crate::qa::MAX_PAIRS {
                        capped = true;
                        break;
                    }
                    let id = d.new_qa_id();
                    d.qa.push(crate::qa::QaPair { id, doc_id: doc_id.clone(), doc: name.clone(), page: 0, style, system, question, answer, context: ctx.into() });
                    added += 1;
                }
            }
            Parsed::Summary(items) => {
                for (style, (text, summary, system)) in items {
                    if d.summaries.len() >= crate::summarize::MAX_PAIRS {
                        capped = true;
                        break;
                    }
                    let id = d.new_summary_id();
                    d.summaries.push(crate::summarize::SummaryPair { id, doc_id: doc_id.clone(), doc: name.clone(), page: 0, style, system, text, summary });
                    added += 1;
                }
            }
            Parsed::SysOne(items) => {
                for mut sample in items {
                    if d.sysone_samples.len() >= crate::sysone::generate::MAX_SAMPLES {
                        capped = true;
                        break;
                    }
                    sample.id = d.new_sysone_id();
                    if sample.source.trim().is_empty() {
                        sample.source = format!("import:{name}");
                    }
                    d.sysone_samples.push(sample);
                    added += 1;
                }
            }
        }
    }
    match kind {
        Kind::Qa => session.emit_qa_list().await,
        Kind::Summary => session.emit_summary_list().await,
        Kind::SysOne => session.emit_sysone_list().await,
    }
    (StatusCode::OK, Json(json!({ "ok": true, "added": added, "skipped": skipped, "capped": capped }))).into_response()
}

#[derive(Deserialize)]
pub struct ExportQuery {
    sid: String,
    kind: crate::hfdata::Kind,
    style: crate::hfdata::Style,
    #[serde(default)]
    format: crate::hfdata::FileFormat,
    /// Chat style: system prompt for pairs that don't have their own.
    #[serde(default)]
    system: String,
}

/// `GET /api/export?sid=&kind=qa|summary&style=hf|simple|instruction|chat&format=jsonl|parquet` —
/// the current dataset as a downloadable file (see `hfdata`).
pub async fn export_dataset(State(shared): State<Arc<Shared>>, Query(q): Query<ExportQuery>) -> Response {
    use crate::hfdata::{self, FileFormat, Kind};
    let Some(session) = shared.session(&q.sid) else {
        return fail(StatusCode::NOT_FOUND, "Your session expired. Reload the page and try again.");
    };
    let system: String = q.system.trim().chars().take(2000).collect();
    let (jsonl, count, prefix) = {
        let d = session.data.lock().await;
        match q.kind {
            Kind::Qa => (hfdata::export_qa(&d.qa, q.style, &system), d.qa.len(), "qa"),
            Kind::Summary => (hfdata::export_summaries(&d.summaries, q.style, &system), d.summaries.len(), "summary"),
            Kind::SysOne => (Ok(hfdata::export_sysone(&d.sysone_samples)), d.sysone_samples.len(), "sysone"),
        }
    };
    if count == 0 {
        return fail(StatusCode::NOT_FOUND, "There is nothing to export yet.");
    }
    let jsonl = match jsonl {
        Ok(t) => t,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let format = q.format;
    let body = match format {
        FileFormat::Jsonl => Ok(jsonl.into_bytes()),
        FileFormat::Parquet => match tokio::task::spawn_blocking(move || hfdata::jsonl_to_parquet(&jsonl)).await {
            Ok(r) => r,
            Err(_) => Err("Writing the Parquet file failed.".to_string()),
        },
    };
    match body {
        Ok(bytes) => {
            let name = format!("{prefix}-{:?}.{}", q.style, format.extension()).to_lowercase();
            let ctype = if format == FileFormat::Parquet { "application/vnd.apache.parquet" } else { "application/x-ndjson" };
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, ctype.to_string()), (header::CONTENT_DISPOSITION, format!("attachment; filename=\"{name}\"")), (header::HeaderName::from_static("x-filename"), name), (header::HeaderName::from_static("x-count"), count.to_string())],
                bytes,
            )
                .into_response()
        }
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[derive(Deserialize)]
pub struct ModelExportQuery {
    sid: String,
}

/// `GET /api/model/export?sid=` — the most recently trained model as a Hugging Face-style folder
/// (zipped), see `export`.
pub async fn export_model(State(shared): State<Arc<Shared>>, Query(q): Query<ModelExportQuery>) -> Response {
    let Some(session) = shared.session(&q.sid) else {
        return fail(StatusCode::NOT_FOUND, "Your session expired. Reload the page and try again.");
    };
    let trained = session.trained.lock().unwrap().clone();
    let Some(trained) = trained else {
        return fail(StatusCode::NOT_FOUND, "There is no trained model yet. Press Run first.");
    };
    if trained.step == 0 {
        return fail(StatusCode::CONFLICT, "The model hasn't been trained yet (step 0).");
    }
    match tokio::task::spawn_blocking(move || crate::export::bundle(&trained)).await {
        Ok(Ok(bytes)) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/zip"), (header::CONTENT_DISPOSITION, "attachment; filename=\"modelbuilder-model.zip\"")],
            bytes,
        )
            .into_response(),
        Ok(Err(e)) => fail(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(_) => fail(StatusCode::INTERNAL_SERVER_ERROR, "Building the model bundle failed."),
    }
}

#[derive(Deserialize)]
pub struct TokenizerQuery {
    sid: String,
    name: String,
}

/// `PUT /api/tokenizer?sid=&name=` — a Hugging Face `tokenizer.json` (raw body, up to
/// `MAX_TOKENIZER_BYTES`, 128 MiB). It replaces the built-in BPE for every dataset the session
/// builds (text, Q&A, summaries), and the dataset is rebuilt with it.
pub async fn upload_tokenizer(State(shared): State<Arc<Shared>>, Query(q): Query<TokenizerQuery>, body: Body) -> Response {
    let Some(session) = shared.session(&q.sid) else {
        return fail(StatusCode::NOT_FOUND, "Your session expired. Reload the page and try again.");
    };
    let limit: usize = std::env::var("MAX_TOKENIZER_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(128 << 20);
    let bytes = match axum::body::to_bytes(body, limit).await {
        Ok(b) => b.to_vec(),
        Err(_) => return fail(StatusCode::PAYLOAD_TOO_LARGE, format!("A tokenizer.json can be at most {}.", mb(limit as u64))),
    };
    let name: String = q.name.rsplit(['/', '\\']).next().unwrap_or("tokenizer.json").chars().filter(|c| !c.is_control()).take(200).collect();
    let parsed = tokio::task::spawn_blocking(move || crate::tokenizer::ImportedTok::parse(&name, bytes)).await;
    let tok = match parsed {
        Ok(Ok(t)) => std::sync::Arc::new(t),
        Ok(Err(e)) => return fail(StatusCode::UNPROCESSABLE_ENTITY, e),
        Err(_) => return fail(StatusCode::INTERNAL_SERVER_ERROR, "Reading the tokenizer failed."),
    };
    let reply = json!({ "ok": true, "name": tok.name, "model": tok.model, "fullVocab": tok.full_vocab });
    session.data.lock().await.custom_tok = Some(tok);
    session.emit_tokenizer().await;
    session.schedule_rebuild();
    (StatusCode::OK, Json(reply)).into_response()
}
