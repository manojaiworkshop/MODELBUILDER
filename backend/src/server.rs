//! Socket.IO wiring. One `Session` (see session.rs) per connected browser tab.
//!
//! client -> server (all use acks so the UI always gets a reply):
//!   data:set        {text?, merges, usePasted}  -> {ok}      (dataset is rebuilt in the background)
//!   data:remove     {id}                        -> {ok}
//!   data:toggle     {id, enabled}               -> {ok}
//!   model:inspect   {spec, probe, seqLen}       -> InspectPayload
//!   train:start     {spec, hyper, probe}        -> {ok, error?}
//!   train:stop      -                           -> {ok}
//!   probe:set       {text}                      -> {ok}
//!   generate        {prompt, maxNew, ...}       -> {ok, text?, error?}   (+ streams `generate:token`)
//!   generate:stop   -                           -> {ok}
//!   learn:ask       {llm, block?, history, question} -> {ok, answer?: LearnAnswer, error?}
//!   learn:test      {llm}                       -> {ok, models?: [string], error?}
//! server -> client (pushed):
//!   data:docs, data:progress, data:info,
//!   train:status, train:steps, train:eval, train:inspect, train:sample, generate:token
//! Files themselves are uploaded over HTTP (see upload.rs), not through the socket.

use crate::events::*;
use crate::inspect::{generate, inspect_model, IB};
use crate::learn;
use crate::model::build;
use crate::session::{DocState, Session, Shared};
use crate::spec::analyze;
use crate::train::{self, Event, Job};
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::Backend;
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use socketioxide::extract::{AckSender, Data, SocketRef};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const MAX_TEXT_BYTES: usize = 2_000_000;
pub const MAX_MERGES: usize = 1000;

fn parse<T: DeserializeOwned>(v: Value) -> Result<T, String> {
    serde_json::from_value(v).map_err(|e| format!("invalid request: {e}"))
}

fn fail(msg: impl Into<String>) -> Value {
    json!({ "ok": false, "error": msg.into() })
}

pub fn on_connect(socket: SocketRef, shared: Arc<Shared>) {
    tracing::info!("client connected: {}", socket.id);
    let s = Session::new(socket.clone(), shared.clone());

    {
        let s = s.clone();
        tokio::spawn(async move { s.emit_docs().await });
    }

    // ------------------------------------------------------------ data:set
    {
        let s = s.clone();
        socket.on("data:set", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: DataSetReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.merges > MAX_MERGES {
                return ack.send(&fail(format!("merges must be <= {MAX_MERGES}"))).ok().unwrap_or(());
            }
            if req.text.as_ref().is_some_and(|t| t.len() > MAX_TEXT_BYTES) {
                return ack.send(&fail(format!("Pasted text is limited to {} MB — upload a file for larger data.", MAX_TEXT_BYTES / 1_000_000))).ok().unwrap_or(());
            }
            {
                let mut d = s.data.lock().await;
                let changed = req.text.as_ref().is_some_and(|t| *t != d.pasted) || req.merges != d.merges || req.use_pasted != d.use_pasted || d.tok.is_none();
                if let Some(t) = req.text {
                    d.pasted = t;
                }
                d.merges = req.merges;
                d.use_pasted = req.use_pasted;
                drop(d);
                if changed {
                    s.schedule_rebuild();
                }
            }
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("data:remove", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: DocIdReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            let removed = {
                let mut d = s.data.lock().await;
                d.docs.iter().position(|x| x.id == req.id).map(|i| d.docs.remove(i))
            };
            if let Some(doc) = removed {
                if let Some(p) = doc.text_path {
                    let _ = std::fs::remove_file(p);
                }
                s.disk_used.fetch_sub(doc.text_bytes.min(s.disk_used.load(Ordering::Relaxed)), Ordering::Relaxed);
                s.emit_docs().await;
                s.schedule_rebuild();
            }
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("data:toggle", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: DocIdReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            {
                let mut d = s.data.lock().await;
                if let Some(doc) = d.docs.iter_mut().find(|x| x.id == req.id && x.state == DocState::Ready) {
                    doc.enabled = req.enabled;
                }
            }
            s.emit_docs().await;
            s.schedule_rebuild();
            ack.send(&json!({ "ok": true })).ok();
        });
    }

    // ------------------------------------------------------- model:inspect
    {
        let s = s.clone();
        socket.on("model:inspect", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: InspectReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            let Some(cur) = s.data.lock().await.tok.clone() else {
                return ack.send(&InspectPayload::failed("No dataset loaded yet.", vec![], vec![], 0, 0, req.seq_len)).ok().unwrap_or(());
            };
            let trained = s.trained.lock().unwrap().clone();
            let payload = tokio::task::spawn_blocking(move || {
                let seq_len = req.seq_len.clamp(4, 256);
                let a = analyze(&req.spec, cur.bpe.vocab_size(), seq_len);
                if a.has_errors {
                    let msg = a.issues.iter().find(|i| i.level == "error").map(|i| i.message.clone()).unwrap_or_default();
                    return InspectPayload::failed(msg, a.issues, a.shapes, a.total_params, cur.bpe.vocab_size(), seq_len);
                }
                let fp = req.spec.fingerprint();
                if let Some(t) = trained.filter(|t| t.fingerprint == fp && t.seq_len == seq_len && t.tok.bpe.vocab_size() == cur.bpe.vocab_size()) {
                    return inspect_model(&t.model, &t.tok, &req.probe, &a, seq_len, Some(t.step), true);
                }
                let dev = NdArrayDevice::default();
                <IB as Backend>::seed(&dev, req.seed);
                match build::<IB>(&req.spec, cur.bpe.vocab_size(), seq_len, &dev) {
                    Ok(m) => inspect_model(&m, &cur, &req.probe, &a, seq_len, None, true),
                    Err(e) => InspectPayload::failed(e, a.issues, a.shapes, a.total_params, cur.bpe.vocab_size(), seq_len),
                }
            })
            .await;
            match payload {
                Ok(p) => ack.send(&p).ok(),
                Err(e) => ack.send(&fail(format!("inspect crashed: {e}"))).ok(),
            };
        });
    }

    // -------------------------------------------------------- train:start
    {
        let s = s.clone();
        let sock = socket.clone();
        let shared = shared.clone();
        socket.on("train:start", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: TrainStartReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if let Err(e) = req.hyper.validate() {
                return ack.send(&fail(e)).ok().unwrap_or(());
            }
            if s.train_cancel.lock().unwrap().is_some() {
                return ack.send(&fail("A training run is already in progress; stop it first.")).ok().unwrap_or(());
            }
            let Some(tok) = s.data.lock().await.tok.clone() else {
                return ack.send(&fail("No dataset loaded yet.")).ok().unwrap_or(());
            };
            let Ok(permit) = shared.trainings.clone().try_acquire_owned() else {
                return ack.send(&fail("The server is busy with other training runs. Try again in a moment.")).ok().unwrap_or(());
            };

            let cancel = Arc::new(AtomicBool::new(false));
            *s.train_cancel.lock().unwrap() = Some(cancel.clone());
            *s.probe.lock().unwrap() = req.probe.chars().take(500).collect();
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
            let job = Job { spec: req.spec, tok, hyper: req.hyper, probe: s.probe.clone(), cancel, tx, trained: s.trained.clone() };

            // forwarder: channel -> socket
            let (fs, fsock) = (s.clone(), sock.clone());
            tokio::spawn(async move {
                while let Some(ev) = rx.recv().await {
                    let _ = match ev {
                        Event::Status(p) => fsock.emit("train:status", &p),
                        Event::Steps(p) => fsock.emit("train:steps", &p),
                        Event::Eval(p) => fsock.emit("train:eval", &p),
                        Event::Inspect(p) => fsock.emit("train:inspect", &*p),
                        Event::Sample(p) => fsock.emit("train:sample", &p),
                    };
                }
                *fs.train_cancel.lock().unwrap() = None;
            });
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                // A panic inside the model must not leave the UI hanging in "running".
                let tx = job.tx.clone();
                let total = job.hyper.steps;
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| train::run(job))).is_err() {
                    let _ = tx.send(Event::Status(StatusPayload { state: "error", message: "The training thread crashed.".into(), step: 0, total }));
                }
            });
            ack.send(&json!({ "ok": true })).ok();
        });
    }

    {
        let s = s.clone();
        socket.on("train:stop", async move |ack: AckSender| {
            if let Some(c) = s.train_cancel.lock().unwrap().as_ref() {
                c.store(true, Ordering::Relaxed);
            }
            ack.send(&json!({ "ok": true })).ok();
        });
    }

    {
        let s = s.clone();
        socket.on("probe:set", async move |Data(v): Data<Value>, ack: AckSender| {
            let text = v.get("text").and_then(|t| t.as_str()).unwrap_or("").chars().take(500).collect::<String>();
            *s.probe.lock().unwrap() = text;
            ack.send(&json!({ "ok": true })).ok();
        });
    }

    // ------------------------------------------------------------ generate
    {
        let s = s.clone();
        let sock = socket.clone();
        socket.on("generate", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: GenerateReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.max_new == 0 || req.max_new > 1000 || req.prompt.len() > 2000 || !(0.0..=5.0).contains(&req.temperature) {
                return ack.send(&fail("prompt <= 2000 chars, 1..1000 new tokens, temperature 0..5")).ok().unwrap_or(());
            }
            let Some(t) = s.trained.lock().unwrap().clone() else {
                return ack.send(&fail("Train the model first — there are no weights to sample from yet.")).ok().unwrap_or(());
            };
            let cancel = Arc::new(AtomicBool::new(false));
            *s.gen_cancel.lock().unwrap() = Some(cancel.clone());
            let out = tokio::task::spawn_blocking(move || {
                generate(&t.model, &t.tok.bpe, &t.tok.frequent, &req.prompt, req.max_new, req.temperature, req.top_k, req.seed, |i, id, text| {
                    let _ = sock.emit("generate:token", &GenToken { i, id, text: text.to_string() });
                    !cancel.load(Ordering::Relaxed)
                })
            })
            .await;
            *s.gen_cancel.lock().unwrap() = None;
            match out {
                Ok(text) => ack.send(&json!({ "ok": true, "text": text })).ok(),
                Err(e) => ack.send(&fail(format!("generation crashed: {e}"))).ok(),
            };
        });
    }
    {
        let s = s.clone();
        socket.on("generate:stop", async move |ack: AckSender| {
            if let Some(c) = s.gen_cancel.lock().unwrap().as_ref() {
                c.store(true, Ordering::Relaxed);
            }
            ack.send(&json!({ "ok": true })).ok();
        });
    }

    // ---------------------------------------------------------------- learn
    {
        let shared = shared.clone();
        socket.on("learn:ask", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: learn::AskReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.question.trim().is_empty() {
                return ack.send(&fail("Type a question first.")).ok().unwrap_or(());
            }
            if req.question.len() > 4000 {
                return ack.send(&fail("That question is a bit long — keep it under 4000 characters.")).ok().unwrap_or(());
            }
            let Ok(_permit) = shared.learns.clone().try_acquire_owned() else {
                return ack.send(&fail("The model server is busy with other questions right now — try again in a moment.")).ok().unwrap_or(());
            };
            match learn::ask(&shared.http, &req).await {
                Ok(answer) => ack.send(&json!({ "ok": true, "answer": answer })).ok(),
                Err(e) => ack.send(&fail(e)).ok(),
            };
        });
    }
    {
        let shared = shared.clone();
        socket.on("learn:test", async move |Data(v): Data<Value>, ack: AckSender| {
            let cfg: learn::LlmConfig = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            match learn::list_models(&shared.http, &cfg).await {
                Ok(models) => ack.send(&json!({ "ok": true, "models": models })).ok(),
                Err(e) => ack.send(&fail(e)).ok(),
            };
        });
    }

    socket.on_disconnect(async move |socket: SocketRef| {
        tracing::info!("client disconnected: {}", socket.id);
        s.close(); // stops runs and deletes this session's uploaded files
    });
}
