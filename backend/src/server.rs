//! Socket.IO wiring. One `Session` (see session.rs) per connected browser tab.
//!
//! client -> server (all use acks so the UI always gets a reply):
//!   data:set        {text?, merges, usePasted}  -> {ok}      (dataset is rebuilt in the background)
//!   tokenizer:default -                         -> {ok}      (drop an imported tokenizer; dataset is rebuilt)
//!   PUT  /api/tokenizer?sid=&name=  raw tokenizer.json body -> {ok, name, model, fullVocab}  (dataset is rebuilt with it)
//!   data:remove     {id}                        -> {ok}
//!   data:toggle     {id, enabled}               -> {ok}
//!   model:inspect   {spec, probe, seqLen}       -> InspectPayload
//!   train:start     {spec, hyper, probe, source?: text|qa|summary, mode?: new|continue} -> {ok, error?}
//!   train:stop      -                           -> {ok}
//!   probe:set       {text}                      -> {ok}
//!   generate        {prompt, maxNew, ...}       -> {ok, text?, error?}   (+ streams `generate:token`)
//!   generate:stop   -                           -> {ok}
//!   qa:pages        {id}                        -> {ok, pages, synthetic, truncated}   (also used to list pages for summarize)
//!   qa:generate     {llm, id, pages, perPage, concurrency} -> {ok}   (progress: qa:progress, qa:added)
//!   qa:stop         -                           -> {ok}
//!   qa:remove       {ids?}                      -> {ok}   (no ids = all; then qa:list)
//!   summarize:generate {llm, id, pages, sentences, concurrency} -> {ok}   (progress: summarize:progress, summarize:added)
//!   summarize:stop     -                        -> {ok}
//!   summarize:remove   {ids?}                   -> {ok}   (no ids = all; then summarize:list)
//!   sysone:generate {llm, id, pages, perPage, concurrency} -> {ok}   (progress: sysone:progress, sysone:added; pages via qa:pages)
//!   sysone:generate:synthetic {llm, domain, context, mix, total, perBatch, concurrency} -> {ok}  (no source document; same progress/added events)
//!   sysone:hf:info   {dataset}  -> {ok, info?: HfDatasetInfo, error?}  (columns/splits of a public HF dataset, via datasets-server)
//!   sysone:hf:import {dataset, config, split, textCol, labelCol, optionsPerQ, total, questionText?} -> {ok}  (same progress/added events; stop via sysone:stop)
//!   sysone:generate:tools {llm, tools, catalogue?, perTool, distractors, concurrency} -> {ok}  (tool-calling samples, one task per tool; same progress/added events)
//!   sysone:toolcall   {state, tools}                -> {ok, result?: {tool, toolConfidence, candidates, specificity, arguments, decisions, missing, toolCalls}}  (last trained System One model: pick a tool and fill its arguments)
//!   sysone:stop         -                       -> {ok}
//!   sysone:remove       {ids?}                  -> {ok}   (no ids = all; then sysone:list)
//!   sysone:train:start  {hyper}                 -> {ok, error?}  (separate pipeline from train:start; pushes sysone:train:*)
//!   sysone:train:stop   -                       -> {ok}
//!   sysone:infer        {state, questions}      -> {ok, answers?, error?}  (the last trained System One model, one forward pass)
//!   GET  /api/export?sid=&kind=&style=&format=jsonl|parquet[&system=] -> file download (see hfdata.rs)
//!   PUT  /api/import?sid=&kind=&style=&name=  raw JSON / JSONL / Parquet body -> {ok, added, skipped}  (style may be auto; then qa:list / summarize:list)
//!   learn:ask       {llm, block?, history, question} -> {ok, answer?: LearnAnswer, error?}
//!   learn:test      {llm}                       -> {ok, models?: [string], error?}
//!   mcp:tools       {name, transport, command?, args?, env?, cwd?, url?, timeoutSecs?, connectTimeoutSecs?} -> {ok, tools?, serverName?, serverVersion?, elapsedMs?, error?}  (connect, list tools, disconnect)
//! server -> client (pushed):
//!   data:docs, data:progress, data:info, data:tokenizer, qa:list, qa:added, qa:progress,
//!   summarize:list, summarize:added, summarize:progress,
//!   sysone:list, sysone:added, sysone:progress, sysone:train:status, sysone:train:step, sysone:train:eval,
//!   train:status, train:steps, train:eval, train:inspect, train:sample, generate:token
//! Files themselves are uploaded over HTTP (see upload.rs), not through the socket.

use crate::events::*;
use crate::inspect::{generate, inspect_model, IB};
use crate::learn;
use crate::qa;
use crate::summarize;
use crate::sysone;
use crate::model::build;
use crate::session::{DocState, QaJob, Session, Shared, SummarizeJob, SysOneGenJob, SysOneHfImportJob, SysOneSynthJob, SysOneToolGenJob};
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

    // ------------------------------------------------------------ tokenizer:default
    {
        let s = s.clone();
        socket.on("tokenizer:default", async move |ack: AckSender| {
            let changed = s.data.lock().await.custom_tok.take().is_some();
            s.emit_tokenizer().await;
            if changed {
                s.schedule_rebuild();
            }
            ack.send(&json!({ "ok": true })).ok();
        });
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
                if let Some(t) = trained.filter(|t| t.fingerprint == fp && t.seq_len == seq_len && (t.own_vocab || t.tok.bpe.vocab_size() == cur.bpe.vocab_size())) {
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
            // Continuing needs the same architecture as the run being continued from — shared by
            // both the Q&A and Summary sources, which each keep their own vocabulary.
            let resolve_prior = |s: &Arc<Session>, req: &TrainStartReq| -> Result<Option<Arc<train::Trained>>, String> {
                if req.mode != TrainMode::Continue {
                    return Ok(None);
                }
                match s.trained.lock().unwrap().clone() {
                    None => Err("There is no trained model to continue from. Train once first, or choose “New model”.".into()),
                    Some(t) if t.fingerprint != req.spec.fingerprint() || t.seq_len != req.hyper.seq_len => {
                        Err("The blocks or the context length changed since that model was trained. Continuing needs the same architecture — restore it, or choose “New model”.".into())
                    }
                    Some(t) => Ok(Some(t)),
                }
            };
            let (tok, init, own_vocab) = match req.source {
                TrainSource::Text => {
                    if req.mode == TrainMode::Continue {
                        return ack
                            .send(&fail("Continuing a trained model works with the Q&A or Summary dataset. Pick one of those, or choose “New model”."))
                            .ok()
                            .unwrap_or(());
                    }
                    let Some(tok) = s.data.lock().await.tok.clone() else {
                        return ack.send(&fail("No dataset loaded yet.")).ok().unwrap_or(());
                    };
                    (tok, None, false)
                }
                TrainSource::Qa => {
                    let (pairs, merges) = {
                        let d = s.data.lock().await;
                        (d.qa.clone(), d.merges)
                    };
                    if pairs.is_empty() {
                        return ack.send(&fail("There are no question-answer pairs yet. Generate some in the Data tab first.")).ok().unwrap_or(());
                    }
                    let prior = match resolve_prior(&s, &req) {
                        Ok(p) => p,
                        Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
                    };
                    let text = qa::format_training_text(&pairs);
                    let bpe = prior.as_ref().map(|t| t.tok.bpe.clone());
                    let custom = s.data.lock().await.custom_tok.clone();
                    let tok = match tokio::task::spawn_blocking(move || qa::tokenize(&text, bpe, custom, merges)).await {
                        Ok(Ok(t)) => Arc::new(t),
                        Ok(Err(e)) => return ack.send(&fail(e)).ok().unwrap_or(()),
                        Err(e) => return ack.send(&fail(format!("tokenizing the Q&A pairs failed: {e}"))).ok().unwrap_or(()),
                    };
                    (tok, prior.map(|t| t.model.clone()), true)
                }
                TrainSource::Summary => {
                    let (pairs, merges) = {
                        let d = s.data.lock().await;
                        (d.summaries.clone(), d.merges)
                    };
                    if pairs.is_empty() {
                        return ack.send(&fail("There are no summaries yet. Generate some in the Data tab first.")).ok().unwrap_or(());
                    }
                    let prior = match resolve_prior(&s, &req) {
                        Ok(p) => p,
                        Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
                    };
                    let text = summarize::format_training_text(&pairs);
                    let bpe = prior.as_ref().map(|t| t.tok.bpe.clone());
                    let custom = s.data.lock().await.custom_tok.clone();
                    let tok = match tokio::task::spawn_blocking(move || qa::tokenize(&text, bpe, custom, merges)).await {
                        Ok(Ok(t)) => Arc::new(t),
                        Ok(Err(e)) => return ack.send(&fail(e)).ok().unwrap_or(()),
                        Err(e) => return ack.send(&fail(format!("tokenizing the summaries failed: {e}"))).ok().unwrap_or(()),
                    };
                    (tok, prior.map(|t| t.model.clone()), true)
                }
            };
            let provenance = {
                let d = s.data.lock().await;
                let uniq = |styles: Vec<crate::hfdata::Style>| {
                    let mut out: Vec<crate::hfdata::Style> = Vec::new();
                    for st in styles {
                        if !out.contains(&st) {
                            out.push(st);
                        }
                    }
                    out
                };
                match req.source {
                    TrainSource::Text => train::Provenance { source: "text".into(), examples: tok.chars, ..Default::default() },
                    TrainSource::Qa => train::Provenance {
                        source: "qa".into(),
                        styles: uniq(d.qa.iter().map(|p| p.style).collect()),
                        system: d.qa.iter().find(|p| !p.system.is_empty()).map(|p| p.system.clone()).unwrap_or_default(),
                        examples: d.qa.len(),
                    },
                    TrainSource::Summary => train::Provenance {
                        source: "summary".into(),
                        styles: uniq(d.summaries.iter().map(|p| p.style).collect()),
                        system: d.summaries.iter().find(|p| !p.system.is_empty()).map(|p| p.system.clone()).unwrap_or_default(),
                        examples: d.summaries.len(),
                    },
                }
            };
            let Ok(permit) = shared.trainings.clone().try_acquire_owned() else {
                return ack.send(&fail("The server is busy with other training runs. Try again in a moment.")).ok().unwrap_or(());
            };

            let cancel = Arc::new(AtomicBool::new(false));
            *s.train_cancel.lock().unwrap() = Some(cancel.clone());
            *s.probe.lock().unwrap() = req.probe.chars().take(500).collect();
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
            let job = Job { spec: req.spec, tok, hyper: req.hyper, probe: s.probe.clone(), cancel, tx, trained: s.trained.clone(), init, own_vocab, provenance };

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

    // ------------------------------------------------------------------- qa
    {
        let s = s.clone();
        socket.on("qa:pages", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: QaPagesReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            let doc = s.data.lock().await.docs.iter().find(|d| d.id == req.id && d.state == DocState::Ready).cloned();
            let Some(doc) = doc else {
                return ack.send(&fail("That document isn't ready (or was removed).")).ok().unwrap_or(());
            };
            let Some(path) = doc.text_path.clone() else {
                return ack.send(&fail("That document has no readable text.")).ok().unwrap_or(());
            };
            match tokio::task::spawn_blocking(move || qa::list_pages(&path, &doc.pages, doc.text_bytes)).await {
                Ok(Ok(l)) => ack.send(&json!({ "ok": true, "pages": l.pages, "synthetic": l.synthetic, "truncated": l.truncated })).ok(),
                Ok(Err(e)) => ack.send(&fail(format!("Could not read the document: {e}"))).ok(),
                Err(e) => ack.send(&fail(format!("listing pages crashed: {e}"))).ok(),
            };
        });
    }
    {
        let s = s.clone();
        socket.on("qa:generate", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: QaGenerateReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.pages.is_empty() || req.pages.len() > 5000 {
                return ack.send(&fail("Select between 1 and 5000 pages.")).ok().unwrap_or(());
            }
            if !(1..=qa::MAX_PER_PAGE).contains(&req.per_page) {
                return ack.send(&fail(format!("Questions per page must be 1..{}.", qa::MAX_PER_PAGE))).ok().unwrap_or(());
            }
            if !(1..=16).contains(&req.concurrency) {
                return ack.send(&fail("Parallel requests must be 1..16.")).ok().unwrap_or(());
            }
            let l = &req.llm;
            if l.base_url.trim().is_empty() || l.api_key.trim().is_empty() || l.model.trim().is_empty() {
                return ack.send(&fail("Set the model server (URL, API key and model) in Settings first.")).ok().unwrap_or(());
            }
            let doc = s.data.lock().await.docs.iter().find(|d| d.id == req.id && d.state == DocState::Ready).cloned();
            let Some(doc) = doc else {
                return ack.send(&fail("That document isn't ready (or was removed).")).ok().unwrap_or(());
            };
            let cancel = Arc::new(AtomicBool::new(false));
            {
                let mut slot = s.qa_cancel.lock().unwrap();
                if slot.is_some() {
                    return ack.send(&fail("A Q&A generation is already running; stop it first.")).ok().unwrap_or(());
                }
                *slot = Some(cancel.clone());
            }
            let mut pages = req.pages;
            pages.sort_unstable();
            pages.dedup();
            s.spawn_qa(QaJob { llm: req.llm, pages, per_page: req.per_page, style: req.style.concrete(), system: req.system.trim().chars().take(2000).collect(), concurrency: req.concurrency }, cancel, doc);
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("qa:stop", async move |ack: AckSender| {
            if let Some(c) = s.qa_cancel.lock().unwrap().as_ref() {
                c.store(true, Ordering::Relaxed);
            }
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("qa:remove", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: QaRemoveReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            {
                let mut d = s.data.lock().await;
                if req.ids.is_empty() {
                    d.qa.clear();
                } else {
                    d.qa.retain(|p| !req.ids.contains(&p.id));
                }
            }
            s.emit_qa_list().await;
            ack.send(&json!({ "ok": true })).ok();
        });
    }

    // -------------------------------------------------------------- summarize
    // Page listing is shared with Q&A (`qa:pages`) — it's just the document's page boundaries,
    // nothing specific to either dataset type.
    {
        let s = s.clone();
        socket.on("summarize:generate", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SummarizeGenerateReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.pages.is_empty() || req.pages.len() > 5000 {
                return ack.send(&fail("Select between 1 and 5000 pages.")).ok().unwrap_or(());
            }
            if !(1..=10).contains(&req.sentences) {
                return ack.send(&fail("Sentences per summary must be 1..10.")).ok().unwrap_or(());
            }
            if !(1..=16).contains(&req.concurrency) {
                return ack.send(&fail("Parallel requests must be 1..16.")).ok().unwrap_or(());
            }
            let l = &req.llm;
            if l.base_url.trim().is_empty() || l.api_key.trim().is_empty() || l.model.trim().is_empty() {
                return ack.send(&fail("Set the model server (URL, API key and model) in Settings first.")).ok().unwrap_or(());
            }
            let doc = s.data.lock().await.docs.iter().find(|d| d.id == req.id && d.state == DocState::Ready).cloned();
            let Some(doc) = doc else {
                return ack.send(&fail("That document isn't ready (or was removed).")).ok().unwrap_or(());
            };
            let cancel = Arc::new(AtomicBool::new(false));
            {
                let mut slot = s.summarize_cancel.lock().unwrap();
                if slot.is_some() {
                    return ack.send(&fail("A summary generation is already running; stop it first.")).ok().unwrap_or(());
                }
                *slot = Some(cancel.clone());
            }
            let mut pages = req.pages;
            pages.sort_unstable();
            pages.dedup();
            s.spawn_summarize(SummarizeJob { llm: req.llm, pages, sentences: req.sentences, style: req.style.concrete(), system: req.system.trim().chars().take(2000).collect(), concurrency: req.concurrency }, cancel, doc);
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("summarize:stop", async move |ack: AckSender| {
            if let Some(c) = s.summarize_cancel.lock().unwrap().as_ref() {
                c.store(true, Ordering::Relaxed);
            }
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("summarize:remove", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SummarizeRemoveReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            {
                let mut d = s.data.lock().await;
                if req.ids.is_empty() {
                    d.summaries.clear();
                } else {
                    d.summaries.retain(|p| !req.ids.contains(&p.id));
                }
            }
            s.emit_summary_list().await;
            ack.send(&json!({ "ok": true })).ok();
        });
    }

    // -------------------------------------------------------------- sysone
    // Page listing is shared with Q&A (`qa:pages`) — same document page boundaries.
    {
        let s = s.clone();
        socket.on("sysone:generate", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SysOneGenerateReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.pages.is_empty() || req.pages.len() > 5000 {
                return ack.send(&fail("Select between 1 and 5000 pages.")).ok().unwrap_or(());
            }
            if !(1..=sysone::generate::MAX_PER_PAGE).contains(&req.per_page) {
                return ack.send(&fail(format!("Samples per page must be 1..{}.", sysone::generate::MAX_PER_PAGE))).ok().unwrap_or(());
            }
            if !(1..=16).contains(&req.concurrency) {
                return ack.send(&fail("Parallel requests must be 1..16.")).ok().unwrap_or(());
            }
            let l = &req.llm;
            if l.base_url.trim().is_empty() || l.api_key.trim().is_empty() || l.model.trim().is_empty() {
                return ack.send(&fail("Set the model server (URL, API key and model) in Settings first.")).ok().unwrap_or(());
            }
            let doc = s.data.lock().await.docs.iter().find(|d| d.id == req.id && d.state == DocState::Ready).cloned();
            let Some(doc) = doc else {
                return ack.send(&fail("That document isn't ready (or was removed).")).ok().unwrap_or(());
            };
            let cancel = Arc::new(AtomicBool::new(false));
            {
                let mut slot = s.sysone_cancel.lock().unwrap();
                if slot.is_some() {
                    return ack.send(&fail("A System One generation is already running; stop it first.")).ok().unwrap_or(());
                }
                *slot = Some(cancel.clone());
            }
            let mut pages = req.pages;
            pages.sort_unstable();
            pages.dedup();
            s.spawn_sysone_generate(SysOneGenJob { llm: req.llm, pages, per_page: req.per_page, concurrency: req.concurrency }, cancel, doc);
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("sysone:generate:synthetic", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SysOneSynthReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.domain.trim().is_empty() {
                return ack.send(&fail("Enter a domain (e.g. \"networking\", \"marketing\").")).ok().unwrap_or(());
            }
            if !(1..=sysone::generate::MAX_SYNTH_TOTAL).contains(&req.total) {
                return ack.send(&fail(format!("Total samples must be 1..{}.", sysone::generate::MAX_SYNTH_TOTAL))).ok().unwrap_or(());
            }
            if !(1..=sysone::generate::MAX_PER_BATCH).contains(&req.per_batch) {
                return ack.send(&fail(format!("Samples per request must be 1..{}.", sysone::generate::MAX_PER_BATCH))).ok().unwrap_or(());
            }
            if !(1..=16).contains(&req.concurrency) {
                return ack.send(&fail("Parallel requests must be 1..16.")).ok().unwrap_or(());
            }
            let l = &req.llm;
            if l.base_url.trim().is_empty() || l.api_key.trim().is_empty() || l.model.trim().is_empty() {
                return ack.send(&fail("Set the model server (URL, API key and model) in Settings first.")).ok().unwrap_or(());
            }
            let cancel = Arc::new(AtomicBool::new(false));
            {
                let mut slot = s.sysone_cancel.lock().unwrap();
                if slot.is_some() {
                    return ack.send(&fail("A System One generation is already running; stop it first.")).ok().unwrap_or(());
                }
                *slot = Some(cancel.clone());
            }
            let domain = req.domain.trim().chars().take(200).collect::<String>();
            let context = req.context.trim().chars().take(4000).collect::<String>();
            s.spawn_sysone_synth(SysOneSynthJob { llm: req.llm, domain, context, mix: req.mix, total: req.total, per_batch: req.per_batch, concurrency: req.concurrency }, cancel);
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("sysone:generate:tools", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SysOneToolGenReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            use sysone::toolgen::{MAX_CATALOGUE, MAX_DISTRACTORS, MAX_PER_TOOL, MAX_TOOLS_PER_RUN};
            if req.tools.is_empty() || req.tools.len() > MAX_TOOLS_PER_RUN {
                return ack.send(&fail(format!("Pick between 1 and {MAX_TOOLS_PER_RUN} tools."))).ok().unwrap_or(());
            }
            if req.catalogue.len() > MAX_CATALOGUE {
                return ack.send(&fail(format!("At most {MAX_CATALOGUE} tools can be told apart in one dataset."))).ok().unwrap_or(());
            }
            if !(1..=MAX_PER_TOOL).contains(&req.per_tool) {
                return ack.send(&fail(format!("Samples per tool must be 1..{MAX_PER_TOOL}."))).ok().unwrap_or(());
            }
            if req.distractors > MAX_DISTRACTORS {
                return ack.send(&fail(format!("Other tools to offer must be 0..{MAX_DISTRACTORS}."))).ok().unwrap_or(());
            }
            if !(1..=8).contains(&req.concurrency) {
                return ack.send(&fail("Parallel tools must be 1..8.")).ok().unwrap_or(());
            }
            let l = &req.llm;
            if l.base_url.trim().is_empty() || l.api_key.trim().is_empty() || l.model.trim().is_empty() {
                return ack.send(&fail("Set the model server (URL, API key and model) in Settings first.")).ok().unwrap_or(());
            }
            let cancel = Arc::new(AtomicBool::new(false));
            {
                let mut slot = s.sysone_cancel.lock().unwrap();
                if slot.is_some() {
                    return ack.send(&fail("A System One generation is already running; stop it first.")).ok().unwrap_or(());
                }
                *slot = Some(cancel.clone());
            }
            let catalogue = if req.catalogue.is_empty() { req.tools.clone() } else { req.catalogue };
            s.spawn_sysone_toolgen(SysOneToolGenJob { llm: req.llm, tools: req.tools, catalogue, per_tool: req.per_tool, distractors: req.distractors, concurrency: req.concurrency }, cancel);
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("sysone:hf:info", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SysOneHfInfoReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.dataset.trim().is_empty() {
                return ack.send(&fail("Enter a Hugging Face dataset repo id, e.g. \"dair-ai/emotion\".")).ok().unwrap_or(());
            }
            match sysone::hfimport::fetch_info(&s.shared.http, req.dataset.trim()).await {
                Ok(info) => ack.send(&json!({ "ok": true, "info": info })).ok(),
                Err(e) => ack.send(&fail(e)).ok(),
            };
        });
    }
    {
        let s = s.clone();
        socket.on("sysone:hf:import", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SysOneHfImportReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.dataset.trim().is_empty() || req.config.trim().is_empty() || req.split.trim().is_empty() {
                return ack.send(&fail("Pick a dataset, config and split first.")).ok().unwrap_or(());
            }
            if req.text_col.trim().is_empty() || req.label_col.trim().is_empty() {
                return ack.send(&fail("Pick which column is the text and which is the label.")).ok().unwrap_or(());
            }
            if !(1..=sysone::hfimport::MAX_ROWS).contains(&req.total) {
                return ack.send(&fail(format!("Rows to import must be 1..{}.", sysone::hfimport::MAX_ROWS))).ok().unwrap_or(());
            }
            let cancel = Arc::new(AtomicBool::new(false));
            {
                let mut slot = s.sysone_cancel.lock().unwrap();
                if slot.is_some() {
                    return ack.send(&fail("A System One generation is already running; stop it first.")).ok().unwrap_or(());
                }
                *slot = Some(cancel.clone());
            }
            s.spawn_sysone_hf_import(
                SysOneHfImportJob {
                    dataset: req.dataset.trim().to_string(),
                    config: req.config,
                    split: req.split,
                    text_col: req.text_col,
                    label_col: req.label_col,
                    options_per_q: req.options_per_q,
                    total: req.total,
                    question_text: req.question_text,
                },
                cancel,
            );
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("sysone:stop", async move |ack: AckSender| {
            if let Some(c) = s.sysone_cancel.lock().unwrap().as_ref() {
                c.store(true, Ordering::Relaxed);
            }
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("sysone:remove", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SysOneRemoveReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            {
                let mut d = s.data.lock().await;
                if req.ids.is_empty() {
                    d.sysone_samples.clear();
                } else {
                    d.sysone_samples.retain(|p| !req.ids.contains(&p.id));
                }
            }
            s.emit_sysone_list().await;
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        let shared = shared.clone();
        socket.on("sysone:train:start", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SysOneTrainStartReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if let Err(e) = req.hyper.validate() {
                return ack.send(&fail(e)).ok().unwrap_or(());
            }
            if s.sysone_train_cancel.lock().unwrap().is_some() {
                return ack.send(&fail("A System One training run is already in progress; stop it first.")).ok().unwrap_or(());
            }
            if req.hyper.base == "laya" && !sysone::laya::is_downloaded(&sysone::laya::cache_dir()) {
                return ack.send(&fail("Import Laya first (Training tab → Start from → Laya).")).ok().unwrap_or(());
            }
            let samples = s.data.lock().await.sysone_samples.clone();
            if samples.is_empty() {
                return ack.send(&fail("There are no System One samples yet. Generate some in the Data tab first.")).ok().unwrap_or(());
            }
            let Ok(_permit) = shared.trainings.clone().try_acquire_owned() else {
                return ack.send(&fail("The server is busy with other training runs. Try again in a moment.")).ok().unwrap_or(());
            };
            let cancel = Arc::new(AtomicBool::new(false));
            *s.sysone_train_cancel.lock().unwrap() = Some(cancel.clone());
            s.spawn_sysone_train(samples, req.hyper, cancel);
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("sysone:train:stop", async move |ack: AckSender| {
            if let Some(c) = s.sysone_train_cancel.lock().unwrap().as_ref() {
                c.store(true, Ordering::Relaxed);
            }
            ack.send(&json!({ "ok": true })).ok();
        });
    }
    {
        let s = s.clone();
        socket.on("sysone:infer", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SysOneInferReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            let Some(trained) = s.sysone_trained.lock().unwrap().clone() else {
                return ack.send(&fail("No trained System One model yet — train one first.")).ok().unwrap_or(());
            };
            let result = tokio::task::spawn_blocking(move || {
                trained.decide(&req.state, &req.questions)
            })
            .await;
            match result {
                Ok(Ok(answers)) => ack.send(&json!({ "ok": true, "answers": answers })).ok(),
                Ok(Err(e)) => ack.send(&fail(e)).ok(),
                Err(e) => ack.send(&fail(format!("inference crashed: {e}"))).ok(),
            };
        });
    }

    {
        let s = s.clone();
        socket.on("sysone:toolcall", async move |Data(v): Data<Value>, ack: AckSender| {
            let req: SysOneToolCallReq = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if req.state.trim().is_empty() {
                return ack.send(&fail("Type a request first.")).ok().unwrap_or(());
            }
            if req.tools.is_empty() || req.tools.len() > sysone::toolgen::MAX_CATALOGUE {
                return ack.send(&fail(format!("Offer between 1 and {} tools.", sysone::toolgen::MAX_CATALOGUE))).ok().unwrap_or(());
            }
            let Some(trained) = s.sysone_trained.lock().unwrap().clone() else {
                return ack.send(&fail("No trained System One model yet — train one first.")).ok().unwrap_or(());
            };
            let result = tokio::task::spawn_blocking(move || {
                trained.call_tool(&req.state, &req.tools)
            })
            .await;
            match result {
                Ok(Ok(out)) => ack.send(&json!({ "ok": true, "result": out })).ok(),
                Ok(Err(e)) => ack.send(&fail(e)).ok(),
                Err(e) => ack.send(&fail(format!("inference crashed: {e}"))).ok(),
            };
        });
    }


    // ------------------------------------------------------------------ laya
    {
        socket.on("sysone:laya:status", async move |ack: AckSender| {
            ack.send(&json!({ "ok": true, "status": sysone::layahub::LayaStatus::current(&sysone::laya::cache_dir()) })).ok();
        });
    }
    {
        let s = s.clone();
        let shared = shared.clone();
        socket.on("sysone:laya:download", async move |ack: AckSender| {
            let dir = sysone::laya::cache_dir();
            if sysone::laya::is_downloaded(&dir) {
                return ack.send(&json!({ "ok": true })).ok().unwrap_or(());
            }
            let cancel = Arc::new(AtomicBool::new(false));
            {
                let mut slot = shared.laya_download.lock().unwrap();
                if slot.is_some() {
                    return ack.send(&fail("Laya is already downloading.")).ok().unwrap_or(());
                }
                *slot = Some(cancel.clone());
            }
            ack.send(&json!({ "ok": true })).ok();
            let (s2, shared2) = (s.clone(), shared.clone());
            tokio::spawn(async move {
                let emit = |st: sysone::layahub::LayaStatus| {
                    let _ = s2.socket.emit("sysone:laya:status", &st);
                };
                emit(sysone::layahub::LayaStatus { state: "downloading", done: 0, total: sysone::laya::WEIGHTS_BYTES_HINT, message: String::new() });
                let s3 = s2.clone();
                let result = sysone::layahub::download(&dir, cancel.clone(), move |done, total| {
                    let _ = s3.socket.emit("sysone:laya:status", &sysone::layahub::LayaStatus { state: "downloading", done, total, message: String::new() });
                })
                .await;
                *shared2.laya_download.lock().unwrap() = None;
                match result {
                    Ok(()) => emit(sysone::layahub::LayaStatus::current(&dir)),
                    Err(e) if e == "cancelled" => emit(sysone::layahub::LayaStatus::current(&dir)),
                    Err(e) => emit(sysone::layahub::LayaStatus { state: "error", done: 0, total: 0, message: e }),
                }
            });
        });
    }
    {
        let shared = shared.clone();
        socket.on("sysone:laya:cancel", async move |ack: AckSender| {
            if let Some(c) = shared.laya_download.lock().unwrap().as_ref() {
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

    // ------------------------------------------------------------------ mcp
    {
        let shared = shared.clone();
        socket.on("mcp:tools", async move |Data(v): Data<Value>, ack: AckSender| {
            let cfg: sysone::mcp::McpServerConfig = match parse(v) {
                Ok(r) => r,
                Err(e) => return ack.send(&fail(e)).ok().unwrap_or(()),
            };
            if let Err(e) = sysone::mcp::validate(&cfg, shared.mcp_stdio) {
                return ack.send(&fail(e)).ok().unwrap_or(());
            }
            let Ok(_permit) = shared.mcp_conns.clone().try_acquire_owned() else {
                return ack.send(&fail("Too many MCP connections are open right now — try again in a moment.")).ok().unwrap_or(());
            };
            match sysone::mcp::list_tools(&cfg, shared.mcp_stdio).await {
                Ok(l) => ack.send(&json!({ "ok": true, "tools": l.tools, "serverName": l.server_name, "serverVersion": l.server_version, "elapsedMs": l.elapsed_ms })).ok(),
                Err(e) => ack.send(&fail(e)).ok(),
            };
        });
    }

    socket.on_disconnect(async move |socket: SocketRef| {
        tracing::info!("client disconnected: {}", socket.id);
        s.close(); // stops runs and deletes this session's uploaded files
    });
}
