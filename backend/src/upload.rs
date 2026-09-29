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
            note: None, error: None, enabled: true, text_path: None,
        });
    }
    session.emit_docs().await;
    session.spawn_extraction(id.clone(), path, name);
    (StatusCode::OK, Json(json!({ "ok": true, "id": id, "size": written }))).into_response()
}
