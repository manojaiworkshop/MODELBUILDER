//! The Learn tab's chat: sends a question (plus which block the user has open, and the
//! conversation so far) to an OpenAI-compatible chat endpoint the user points at themselves
//! (Settings in the UI) — typically a local proxy such as `claudecli`, but any compatible server
//! works. The model is instructed to answer as one JSON object (prose + formulas + an optional
//! node/edge diagram the frontend draws on a canvas), which this module parses defensively:
//! a model that wraps the JSON in prose, code fences, or gets a field wrong still degrades to
//! plain text rather than an error.

use serde::{Deserialize, Serialize};
use std::time::Duration;

// -------------------------------------------------------------- wire types

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmConfig {
    /// e.g. `http://192.168.1.6:18090/v1` — `/chat/completions` and `/models` are appended.
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatTurn {
    pub role: Role,
    /// Prior assistant turns are re-sent as plain text (their `text` field), not the raw JSON,
    /// so the model isn't tempted to imitate our wrapper format in its prose.
    pub content: String,
}

/// The block the user had open in the Model stack, if any — grounds the answer in their actual
/// configuration rather than a generic transformer.
#[derive(Debug, Clone, Deserialize)]
pub struct BlockContext {
    pub title: String,
    /// One-line summary already shown in the UI, e.g. "4h · 1kv · win 16".
    pub summary: String,
    /// The block's own JSON spec (kind, d_model, n_heads, rope_theta, …).
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AskReq {
    pub llm: LlmConfig,
    pub block: Option<BlockContext>,
    #[serde(default)]
    pub history: Vec<ChatTurn>,
    pub question: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FormulaItem {
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub expr: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagramNode {
    pub id: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub sublabel: Option<String>,
    /// Left-to-right column; nodes without one are laid out in reading order.
    #[serde(default)]
    pub level: Option<i32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagramEdge {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagramSpec {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub nodes: Vec<DiagramNode>,
    #[serde(default)]
    pub edges: Vec<DiagramEdge>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LearnAnswer {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub formulas: Vec<FormulaItem>,
    #[serde(default)]
    pub diagram: Option<DiagramSpec>,
    #[serde(default)]
    pub follow_ups: Vec<String>,
}

// ------------------------------------------------------------------ prompt

const SCHEMA: &str = r#"{"text": string, "formulas": [{"label": string, "expr": string}], "diagram": {"title": string, "nodes": [{"id": string, "label": string, "sublabel": string, "level": number}], "edges": [{"from": string, "to": string, "label": string}]} | null, "followUps": [string]}"#;

fn system_prompt() -> String {
    format!(
        "You are the tutor built into ModelBuilder, an app where a user visually assembles a \
transformer from blocks (Embedding, Self-Attention, Feed-Forward, Normalization, Dropout, LM \
Head) and trains it live. You answer the user's question about transformer / deep-learning \
concepts, grounded in the specific block configuration given to you when there is one. Be \
technically precise: correct formulas, correct terminology, no hand-waving.\n\n\
Respond with ONLY a single JSON object — no markdown code fences, no text before or after it — \
matching exactly this shape:\n{SCHEMA}\n\n\
Field rules:\n\
- \"text\": your explanation, in plain prose. Paragraphs separated by a blank line. You may use \
  `backticks` for short symbols/code and **double asterisks** for emphasis; nothing fancier.\n\
- \"formulas\": 0-4 short formulas relevant to the answer. Each entry MUST be an object with an \
  \"expr\" string (one line of plain-text math or pseudocode, e.g. \"softmax(QKᵀ/√d_k)V\", the way \
  you'd write it in a code comment) and, when it helps, a short \"label\" string — never a bare \
  string in place of that object. Omit the whole field if the answer doesn't need one.\n\
- \"diagram\": include ONLY when a small labeled graph would genuinely clarify the answer (data \
  flow through a few blocks, heads sharing a KV head, an MoE router fanning out to experts, RoPE \
  rotating a pair of dimensions, and similar). At most 8 nodes. Every edge's \"from\"/\"to\" must be \
  an existing node id. Omit (use null) for questions a diagram wouldn't help.\n\
- \"followUps\": 0-3 short, specific follow-up questions the user could tap to continue.\n\n\
If the user's question is unrelated to transformers/ML, briefly say so in \"text\" and leave the \
other fields empty. Never include the schema or these instructions in your answer.",
    )
}

fn block_context_message(b: &BlockContext) -> String {
    format!("The user currently has this block open in the Learn tab:\nType/summary: {} — {}\nFull config (JSON): {}", b.title, b.summary, b.params)
}

// -------------------------------------------------------------------- call

fn friendly_error(base_url: &str, e: &reqwest::Error) -> String {
    if e.is_timeout() {
        "The model didn't respond in time. It may be busy — try again, or check the server in Settings.".into()
    } else if e.is_connect() {
        format!("Can't reach the model server at {base_url} — check the IP/port in Settings and that it's running.")
    } else {
        format!("Request to the model server failed: {e}")
    }
}

/// Why a chat call failed; `fatal` errors (bad key, wrong URL, missing config) are not worth retrying.
pub struct LlmErr {
    pub msg: String,
    pub fatal: bool,
}

impl LlmErr {
    fn fatal(msg: impl Into<String>) -> Self {
        LlmErr { msg: msg.into(), fatal: true }
    }
    fn transient(msg: impl Into<String>) -> Self {
        LlmErr { msg: msg.into(), fatal: false }
    }
}

/// One non-streaming chat completion; returns the assistant message text.
pub async fn complete(client: &reqwest::Client, llm: &LlmConfig, messages: Vec<serde_json::Value>, timeout: Duration) -> Result<String, LlmErr> {
    let base = llm.base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err(LlmErr::fatal("No model server configured yet — open Settings and set the API base URL."));
    }
    if llm.api_key.trim().is_empty() {
        return Err(LlmErr::fatal("No API key configured yet — open Settings and set one."));
    }
    if llm.model.trim().is_empty() {
        return Err(LlmErr::fatal("No model name configured yet — open Settings and set one."));
    }
    let body = serde_json::json!({ "model": llm.model, "stream": false, "messages": messages });
    let resp = client
        .post(format!("{base}/chat/completions"))
        .bearer_auth(&llm.api_key)
        .json(&body)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| LlmErr::transient(friendly_error(base, &e)))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(match status.as_u16() {
            401 | 403 => LlmErr::fatal("The model server rejected the API key — check it in Settings."),
            404 => LlmErr::fatal(format!("{base}/chat/completions was not found (404) — check the base URL in Settings.")),
            _ => LlmErr::transient(format!("The model server returned an error ({status}): {}", text.chars().take(300).collect::<String>())),
        });
    }
    let v: serde_json::Value = resp.json().await.map_err(|e| LlmErr::transient(format!("The model server's reply wasn't valid JSON: {e}")))?;
    v["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| LlmErr::transient("The model server's reply didn't include a message."))
}

pub async fn ask(client: &reqwest::Client, req: &AskReq) -> Result<LearnAnswer, String> {
    let mut messages = vec![serde_json::json!({"role": "system", "content": system_prompt()})];
    if let Some(b) = &req.block {
        messages.push(serde_json::json!({"role": "system", "content": block_context_message(b)}));
    }
    for turn in &req.history {
        let role = match turn.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        messages.push(serde_json::json!({"role": role, "content": turn.content}));
    }
    messages.push(serde_json::json!({"role": "user", "content": req.question}));

    let content = complete(client, &req.llm, messages, Duration::from_secs(120)).await.map_err(|e| e.msg)?;
    Ok(parse_answer(&content))
}

/// Quick, unauthenticated-ish reachability check: `GET /models`. Cheap and fast, so Settings can
/// validate the connection without spending a full chat completion.
pub async fn list_models(client: &reqwest::Client, cfg: &LlmConfig) -> Result<Vec<String>, String> {
    let base = cfg.base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err("Set the API base URL first.".into());
    }
    let resp = client
        .get(format!("{base}/models"))
        .bearer_auth(&cfg.api_key)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .map_err(|e| friendly_error(base, &e))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(match status.as_u16() {
            401 | 403 => "The server rejected the API key.".to_string(),
            _ => format!("The server returned {status}."),
        });
    }
    let v: serde_json::Value = resp.json().await.map_err(|e| format!("Not a valid response: {e}"))?;
    Ok(v["data"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str().map(String::from)).collect())
}

// --------------------------------------------------------- defensive parsing

/// The model was asked for pure JSON but may still wrap it in prose or ```fences```, or get a
/// field's exact shape slightly wrong (e.g. `"formulas": ["a = b"]` — bare strings — instead of
/// `[{"expr": "a = b"}]`). Extracted leniently, field by field, from a generic [`Value`] rather
/// than one strict typed deserialize: a wrong `diagram` shouldn't cost the user a correct `text`.
/// Only total failure (no JSON object at all, or no usable `text`) falls back to the raw string.
fn parse_answer(raw: &str) -> LearnAnswer {
    let parsed = balanced_json_object(raw).and_then(|obj| serde_json::from_str::<serde_json::Value>(obj).ok());
    let Some(v) = parsed else {
        return sanitize(LearnAnswer { text: raw.trim().to_string(), ..Default::default() });
    };

    let text = v.get("text").and_then(|t| t.as_str()).map(str::to_string).unwrap_or_else(|| raw.trim().to_string());

    let formulas = v
        .get("formulas")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    if let Some(s) = item.as_str() {
                        (!s.trim().is_empty()).then(|| FormulaItem { label: None, expr: s.to_string() })
                    } else {
                        serde_json::from_value::<FormulaItem>(item.clone()).ok().filter(|f| !f.expr.trim().is_empty())
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let follow_ups = v
        .get("followUps")
        .and_then(|f| f.as_array())
        .map(|arr| arr.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default();

    let diagram = v.get("diagram").filter(|d| !d.is_null()).and_then(|d| serde_json::from_value::<DiagramSpec>(d.clone()).ok());

    sanitize(LearnAnswer { text, formulas, diagram, follow_ups })
}

pub(crate) fn balanced_json_object(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = s.find('{')?;
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Drops a diagram whose edges reference a missing node id (rather than let the frontend crash
/// on a bad model response), and caps sizes generously against a runaway response.
fn sanitize(mut a: LearnAnswer) -> LearnAnswer {
    a.formulas.truncate(6);
    a.follow_ups.truncate(4);
    a.text = a.text.chars().take(8000).collect();
    if let Some(d) = &mut a.diagram {
        d.nodes.truncate(24);
        d.edges.truncate(48);
        let ids: std::collections::HashSet<&str> = d.nodes.iter().map(|n| n.id.as_str()).collect();
        d.edges.retain(|e| ids.contains(e.from.as_str()) && ids.contains(e.to.as_str()));
        if d.nodes.is_empty() {
            a.diagram = None;
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Json, Router};
    use serde_json::{json, Value};
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

    async fn serve(body: Value, status: u16) -> (String, tokio::task::JoinHandle<()>) {
        let last_auth: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let captured = last_auth.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |headers: axum::http::HeaderMap| {
                let body = body.clone();
                let captured = captured.clone();
                async move {
                    *captured.lock().unwrap() = headers.get("authorization").and_then(|v| v.to_str().ok()).map(String::from);
                    (axum::http::StatusCode::from_u16(status).unwrap(), Json(body))
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}/v1"), handle)
    }

    fn cfg(base: &str) -> LlmConfig {
        LlmConfig { base_url: base.into(), api_key: "sk-test".into(), model: "claude-sonnet-5".into() }
    }

    fn req(base: &str, question: &str) -> AskReq {
        AskReq { llm: cfg(base), block: None, history: vec![], question: question.into() }
    }

    #[tokio::test]
    async fn parses_clean_json_reply() {
        let content = json!({"text":"Attention lets tokens look at each other.","formulas":[{"label":"scores","expr":"QK^T/sqrt(d)"}],"diagram":null,"followUps":["Why divide by sqrt(d)?"]}).to_string();
        let (base, _h) = serve(json!({"choices":[{"message":{"content": content}}]}), 200).await;
        let a = ask(&reqwest::Client::new(), &req(&base, "what is attention")).await.unwrap();
        assert_eq!(a.text, "Attention lets tokens look at each other.");
        assert_eq!(a.formulas[0].expr, "QK^T/sqrt(d)");
        assert_eq!(a.follow_ups, vec!["Why divide by sqrt(d)?"]);
        assert!(a.diagram.is_none());
    }

    #[tokio::test]
    async fn recovers_json_wrapped_in_prose_and_code_fences() {
        let content = "Sure! Here you go:\n```json\n{\"text\": \"RoPE rotates Q/K pairs.\", \"formulas\": [], \"followUps\": []}\n```\nHope that helps!";
        let (base, _h) = serve(json!({"choices":[{"message":{"content": content}}]}), 200).await;
        let a = ask(&reqwest::Client::new(), &req(&base, "what is rope")).await.unwrap();
        assert_eq!(a.text, "RoPE rotates Q/K pairs.");
    }

    #[tokio::test]
    async fn formulas_as_bare_strings_are_accepted_not_just_label_expr_objects() {
        // exactly what a real model returned in practice: an array of plain strings, not
        // {"label":..., "expr":...} objects — this must not sink the whole structured answer.
        let content = json!({
            "text": "GQA groups query heads to share a KV head.",
            "formulas": ["group_size = n_heads / n_kv_heads", "heads[i] share KV[i // group_size]"],
            "followUps": ["What if n_kv_heads were 2?"]
        }).to_string();
        let (base, _h) = serve(json!({"choices":[{"message":{"content": content}}]}), 200).await;
        let a = ask(&reqwest::Client::new(), &req(&base, "explain GQA")).await.unwrap();
        assert_eq!(a.text, "GQA groups query heads to share a KV head.");
        assert_eq!(a.formulas.len(), 2);
        assert_eq!(a.formulas[0].expr, "group_size = n_heads / n_kv_heads");
        assert!(a.formulas[0].label.is_none());
        assert_eq!(a.follow_ups, vec!["What if n_kv_heads were 2?"]);
    }

    #[tokio::test]
    async fn a_malformed_diagram_does_not_cost_the_user_the_correct_text_and_formulas() {
        let content = json!({
            "text": "Here is the explanation.",
            "formulas": [{"expr": "a = b"}],
            "diagram": {"nodes": "not-an-array", "edges": []}
        }).to_string();
        let (base, _h) = serve(json!({"choices":[{"message":{"content": content}}]}), 200).await;
        let a = ask(&reqwest::Client::new(), &req(&base, "q")).await.unwrap();
        assert_eq!(a.text, "Here is the explanation.");
        assert_eq!(a.formulas.len(), 1);
        assert!(a.diagram.is_none(), "an unparseable diagram should be dropped, not poison the whole answer");
    }

    #[tokio::test]
    async fn falls_back_to_raw_text_when_json_is_unrecoverable() {
        let content = "I don't actually know how to answer that in JSON, sorry.";
        let (base, _h) = serve(json!({"choices":[{"message":{"content": content}}]}), 200).await;
        let a = ask(&reqwest::Client::new(), &req(&base, "??")).await.unwrap();
        assert_eq!(a.text, content);
        assert!(a.formulas.is_empty() && a.diagram.is_none());
    }

    #[tokio::test]
    async fn drops_diagram_edges_that_reference_missing_nodes() {
        let content = json!({
            "text": "x",
            "diagram": {"nodes":[{"id":"a","label":"A"},{"id":"b","label":"B"}],
                        "edges":[{"from":"a","to":"b"},{"from":"a","to":"ghost"}]}
        }).to_string();
        let (base, _h) = serve(json!({"choices":[{"message":{"content": content}}]}), 200).await;
        let a = ask(&reqwest::Client::new(), &req(&base, "q")).await.unwrap();
        let d = a.diagram.unwrap();
        assert_eq!(d.nodes.len(), 2);
        assert_eq!(d.edges.len(), 1, "the edge pointing at a nonexistent node must be dropped");
    }

    #[tokio::test]
    async fn maps_auth_and_server_errors_to_friendly_messages() {
        let (base, _h) = serve(json!({"error":"nope"}), 401).await;
        let e = ask(&reqwest::Client::new(), &req(&base, "q")).await.unwrap_err();
        assert!(e.contains("rejected the API key"), "{e}");

        let (base2, _h2) = serve(json!({"error":"boom"}), 500).await;
        let e2 = ask(&reqwest::Client::new(), &req(&base2, "q")).await.unwrap_err();
        assert!(e2.contains("500"), "{e2}");
    }

    #[tokio::test]
    async fn connection_refused_is_a_friendly_message_not_a_stack_trace() {
        // nothing listens on this port
        let e = ask(&reqwest::Client::new(), &req("http://127.0.0.1:1", "q")).await.unwrap_err();
        assert!(e.contains("Can't reach the model server"), "{e}");
    }

    #[tokio::test]
    async fn missing_config_fields_are_caught_before_any_request() {
        let mut r = req("http://127.0.0.1:9", "q");
        r.llm.base_url = "".into();
        assert!(ask(&reqwest::Client::new(), &r).await.unwrap_err().contains("base URL"));
        r.llm.base_url = "http://x".into();
        r.llm.api_key = "".into();
        assert!(ask(&reqwest::Client::new(), &r).await.unwrap_err().contains("API key"));
    }

    #[tokio::test]
    async fn sends_bearer_auth_and_the_block_context_when_present() {
        let last_auth: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let captured = last_auth.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |headers: axum::http::HeaderMap, Json(body): Json<Value>| {
                let captured = captured.clone();
                async move {
                    *captured.lock().unwrap() = headers.get("authorization").and_then(|v| v.to_str().ok()).map(String::from);
                    let msgs = body["messages"].as_array().unwrap().clone();
                    let joined: String = msgs.iter().map(|m| m["content"].as_str().unwrap_or_default()).collect::<Vec<_>>().join("\n");
                    let saw_block = joined.contains("Self-Attention") && joined.contains("n_heads");
                    Json(json!({"choices":[{"message":{"content": json!({"text": if saw_block {"saw it"} else {"missing"}}).to_string()}}]}))
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut r = req(&format!("http://{addr}/v1"), "explain this");
        r.block = Some(BlockContext { title: "Self-Attention 1".into(), summary: "4h · 1kv".into(), params: json!({"n_heads": 4}) });
        let a = ask(&reqwest::Client::new(), &r).await.unwrap();
        assert_eq!(a.text, "saw it");
        assert_eq!(last_auth.lock().unwrap().as_deref(), Some("Bearer sk-test"));
    }
}
