//! ModelBuilder backend: an HTTP + Socket.IO server that assembles a transformer
//! from a stack of blocks, trains it live and streams what happens inside.

mod corpus;
mod events;
mod extract;
mod inspect;
mod learn;
mod model;
mod server;
mod session;
mod spec;
mod tokenizer;
mod trace;
mod train;
mod upload;

use axum::{extract::DefaultBodyLimit, routing::{get, put}, Json, Router};
use serde_json::json;
use socketioxide::SocketIo;
use std::time::Duration;
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(4000);
    let shared = session::Shared::from_env();
    let max_trainings = shared.trainings.available_permits();
    let (io_layer, io) = SocketIo::builder()
        .max_payload(server::MAX_TEXT_BYTES as u64 + 100_000)
        .max_buffer_size(4096)
        .ping_interval(Duration::from_secs(10))
        .ping_timeout(Duration::from_secs(30))
        .build_layer();
    io.ns("/", {
        let shared = shared.clone();
        async move |socket: socketioxide::extract::SocketRef| server::on_connect(socket, shared.clone())
    });

    // CORS_ORIGIN="http://localhost:3000,http://my.host:3000" restricts browsers; unset = any origin (local dev).
    let cors = match std::env::var("CORS_ORIGIN") {
        Ok(list) if !list.trim().is_empty() => {
            let origins: Vec<_> = list.split(',').filter_map(|o| o.trim().parse().ok()).collect();
            CorsLayer::new().allow_origin(origins).allow_methods(Any).allow_headers(Any)
        }
        _ => CorsLayer::permissive(),
    };

    let app = Router::new()
        .route("/health", get(move || async move { Json(json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION"), "maxTrainings": max_trainings })) }))
        // large bodies are streamed to disk by the handler, which enforces its own limits
        .route("/api/upload", put(upload::upload).layer(DefaultBodyLimit::disable()).with_state(shared.clone()))
        .route("/api/limits", get({
            let sh = shared.clone();
            move || async move { Json(json!({ "maxFileBytes": sh.max_file_bytes, "maxSessionBytes": sh.max_session_bytes, "maxTokens": sh.max_tokens })) }
        }))
        .layer(io_layer)
        .layer(cors);

    let addr = format!("{host}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot listen on {addr}: {e}");
            if e.kind() == std::io::ErrorKind::AddrInUse {
                eprintln!("Another backend is probably already running. Stop it (e.g. `fuser -k {port}/tcp`) or start this one on a different port with PORT=<n>.");
            }
            std::process::exit(1);
        }
    };
    tracing::info!("ModelBuilder backend listening on http://{addr} (socket.io at /socket.io/)");
    axum::serve(listener, app).with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; }).await.unwrap();
}
