//! MCP client for the Tool-calling dataset: connects to a Model Context Protocol server over
//! stdio (a local command), SSE, or streamable HTTP, and lists the tools it exposes — names,
//! descriptions and JSON-Schema parameters — which later become the tool catalogue that
//! tool-call training samples are written against.
//!
//! Each request connects, lists and disconnects (no long-lived child processes or sessions to
//! leak); the UI keeps the result. Server configurations live in the browser, so they survive a
//! backend restart; the backend only ever sees one when asked to use it.
//!
//! Stdio servers run an arbitrary command on the machine the backend runs on, so they are only
//! allowed when the backend listens on loopback (the default) or `ALLOW_MCP_STDIO=1` is set.

use rmcp::model::Tool;
use rmcp::service::ServiceExt;
use rmcp::transport::{SseClientTransport, StreamableHttpClientTransport, TokioChildProcess};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;

const DEFAULT_CONNECT_SECS: u64 = 20;
const DEFAULT_REQUEST_SECS: u64 = 30;
const MAX_CONNECT_SECS: u64 = 120;
const MAX_REQUEST_SECS: u64 = 300;
/// Most tools one server may report (a runaway server shouldn't flood the UI).
const MAX_TOOLS: usize = 1000;
/// Characters of a child's stderr kept for error messages.
const STDERR_TAIL: usize = 1200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    Stdio,
    Sse,
    Http,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerConfig {
    /// Also the prefix of qualified tool names (`<name>__<tool>`), so it is kept to `[A-Za-z0-9_-]`.
    pub name: String,
    pub transport: Transport,
    /// stdio: the program to run (no shell).
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// stdio: extra environment variables, on top of the backend's own.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub cwd: Option<String>,
    /// sse / http: the endpoint.
    #[serde(default)]
    pub url: String,
    /// Seconds allowed for listing the tools once connected.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Seconds allowed for starting / connecting and the MCP handshake.
    #[serde(default)]
    pub connect_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolInfo {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    /// JSON Schema of the arguments.
    pub input_schema: Value,
    pub output_schema: Option<Value>,
    pub read_only: Option<bool>,
}

impl From<Tool> for McpToolInfo {
    fn from(t: Tool) -> Self {
        McpToolInfo {
            name: t.name.to_string(),
            title: t.title.or_else(|| t.annotations.as_ref().and_then(|a| a.title.clone())),
            description: t.description.map(|d| d.to_string()),
            input_schema: Value::Object((*t.input_schema).clone()),
            output_schema: t.output_schema.map(|o| Value::Object((*o).clone())),
            read_only: t.annotations.and_then(|a| a.read_only_hint),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListedTools {
    pub tools: Vec<McpToolInfo>,
    /// The server's own name and version from the MCP handshake.
    pub server_name: Option<String>,
    pub server_version: Option<String>,
    pub elapsed_ms: u64,
}

/// Whether stdio servers may be started: only for a loopback backend unless explicitly allowed.
pub fn stdio_allowed(host: &str) -> bool {
    let loopback = matches!(host.trim(), "127.0.0.1" | "::1" | "localhost" | "[::1]");
    loopback || std::env::var("ALLOW_MCP_STDIO").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

pub fn validate(cfg: &McpServerConfig, allow_stdio: bool) -> Result<(), String> {
    let name = cfg.name.trim();
    if name.is_empty() || name.len() > 40 || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err("The server name must be 1–40 characters: letters, digits, '-' or '_'.".into());
    }
    match cfg.transport {
        Transport::Stdio => {
            if !allow_stdio {
                return Err("Stdio servers run a command on the backend machine, so they are disabled while the backend listens on a network address. Set ALLOW_MCP_STDIO=1 to allow them, or use SSE / Streamable HTTP.".into());
            }
            if cfg.command.trim().is_empty() {
                return Err("Enter the command that starts the server.".into());
            }
            if let Some(k) = cfg.env.keys().find(|k| k.is_empty() || k.contains('=') || k.contains('\0')) {
                return Err(format!("Invalid environment variable name {k:?}."));
            }
        }
        Transport::Sse | Transport::Http => {
            let u = cfg.url.trim();
            if !(u.starts_with("http://") || u.starts_with("https://")) || u.len() > 2000 {
                return Err("Enter the server URL, starting with http:// or https://.".into());
            }
        }
    }
    Ok(())
}

fn secs(v: Option<u64>, default: u64, max: u64) -> Duration {
    Duration::from_secs(v.filter(|&s| s >= 1).unwrap_or(default).min(max))
}

fn tail(s: &str, n: usize) -> String {
    let s = s.trim();
    let skip = s.chars().count().saturating_sub(n);
    s.chars().skip(skip).collect()
}

/// Connects, asks for the tool list, disconnects.
pub async fn list_tools(cfg: &McpServerConfig, allow_stdio: bool) -> Result<ListedTools, String> {
    validate(cfg, allow_stdio)?;
    let started = Instant::now();
    let connect_t = secs(cfg.connect_timeout_secs, DEFAULT_CONNECT_SECS, MAX_CONNECT_SECS);
    let request_t = secs(cfg.timeout_secs, DEFAULT_REQUEST_SECS, MAX_REQUEST_SECS);
    let stderr_tail: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let with_stderr = |msg: String| {
        let t = stderr_tail.lock().unwrap().clone();
        if t.trim().is_empty() { msg } else { format!("{msg}\nServer output: {}", tail(&t, STDERR_TAIL)) }
    };
    let timed_out = |what: &str, d: Duration| format!("Timed out after {}s {what}.", d.as_secs());

    // `serve` runs the MCP handshake, so the connect timeout covers starting the server too.
    let connected = tokio::time::timeout(connect_t, async {
        match cfg.transport {
            Transport::Http => ().serve(StreamableHttpClientTransport::from_uri(cfg.url.trim().to_string())).await.map_err(|e| format!("Could not connect: {e}")),
            Transport::Sse => {
                let t = SseClientTransport::start(cfg.url.trim().to_string()).await.map_err(|e| format!("Could not open the SSE stream: {e}"))?;
                ().serve(t).await.map_err(|e| format!("Could not connect: {e}"))
            }
            Transport::Stdio => {
                let mut cmd = tokio::process::Command::new(cfg.command.trim());
                cmd.args(&cfg.args).envs(&cfg.env).kill_on_drop(true);
                if let Some(dir) = cfg.cwd.as_deref().map(str::trim).filter(|d| !d.is_empty()) {
                    cmd.current_dir(dir);
                }
                let (transport, stderr) = TokioChildProcess::builder(cmd)
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("Could not start `{}`: {e}", cfg.command.trim()))?;
                if let Some(mut err) = stderr {
                    let buf = stderr_tail.clone();
                    tokio::spawn(async move {
                        let mut chunk = [0u8; 512];
                        while let Ok(n) = err.read(&mut chunk).await {
                            if n == 0 {
                                break;
                            }
                            let mut b = buf.lock().unwrap();
                            b.push_str(&String::from_utf8_lossy(&chunk[..n]));
                            if b.len() > 8 * STDERR_TAIL {
                                let keep = tail(&b, STDERR_TAIL);
                                *b = keep;
                            }
                        }
                    });
                }
                ().serve(transport).await.map_err(|e| format!("The server did not complete the MCP handshake: {e}"))
            }
        }
    })
    .await;
    if !matches!(connected, Ok(Ok(_))) && cfg.transport == Transport::Stdio {
        // let the stderr reader catch up, so the server's own error text makes it into the message
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let service = match connected {
        Err(_) => return Err(with_stderr(timed_out("connecting", connect_t))),
        Ok(Err(e)) => return Err(with_stderr(e)),
        Ok(Ok(s)) => s,
    };

    let (server_name, server_version) = service.peer_info().map(|i| (Some(i.server_info.name.clone()), Some(i.server_info.version.clone()))).unwrap_or((None, None));
    let listed = tokio::time::timeout(request_t, service.list_all_tools()).await;
    let _ = service.cancel().await; // closes the connection / stops the child
    let mut tools = match listed {
        Err(_) => return Err(with_stderr(timed_out("listing the tools", request_t))),
        Ok(Err(e)) => return Err(with_stderr(format!("The server refused to list its tools: {e}"))),
        Ok(Ok(t)) => t,
    };
    tools.truncate(MAX_TOOLS);
    let mut tools: Vec<McpToolInfo> = tools.into_iter().map(Into::into).collect();
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(ListedTools { tools, server_name, server_version, elapsed_ms: started.elapsed().as_millis() as u64 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdio(cmd: &str) -> McpServerConfig {
        McpServerConfig { name: "t".into(), transport: Transport::Stdio, command: cmd.into(), args: vec![], env: BTreeMap::new(), cwd: None, url: String::new(), timeout_secs: Some(5), connect_timeout_secs: Some(5) }
    }

    #[test]
    fn validates_names_urls_and_stdio_policy() {
        assert!(validate(&stdio("npx"), true).is_ok());
        assert!(validate(&stdio("npx"), false).unwrap_err().contains("ALLOW_MCP_STDIO"));
        assert!(validate(&stdio("  "), true).unwrap_err().contains("command"));
        let mut c = stdio("npx");
        c.name = "bad name!".into();
        assert!(validate(&c, true).unwrap_err().contains("name"));
        let mut h = McpServerConfig { transport: Transport::Http, url: "ftp://x".into(), ..stdio("") };
        assert!(validate(&h, false).unwrap_err().contains("http"));
        h.url = "http://127.0.0.1:9/mcp".into();
        assert!(validate(&h, false).is_ok());
    }

    #[test]
    fn loopback_hosts_allow_stdio() {
        assert!(stdio_allowed("127.0.0.1") && stdio_allowed("localhost") && stdio_allowed("::1"));
    }

    #[tokio::test]
    async fn a_missing_command_is_a_clear_error() {
        let e = list_tools(&stdio("definitely-not-a-real-binary-xyz"), true).await.unwrap_err();
        assert!(e.contains("Could not start"), "{e}");
    }

    #[tokio::test]
    async fn a_server_that_exits_immediately_reports_its_output() {
        let mut c = stdio("sh");
        c.args = vec!["-c".into(), "echo boom-from-server >&2; exit 3".into()];
        let e = list_tools(&c, true).await.unwrap_err();
        assert!(e.contains("handshake") || e.contains("Could not"), "{e}");
        assert!(e.contains("boom-from-server"), "stderr should be surfaced: {e}");
    }

    #[tokio::test]
    async fn an_unreachable_http_server_fails_fast_not_forever() {
        let c = McpServerConfig { transport: Transport::Http, url: "http://127.0.0.1:1/mcp".into(), ..stdio("") };
        let t = Instant::now();
        assert!(list_tools(&c, false).await.is_err());
        assert!(t.elapsed() < Duration::from_secs(10));
    }
}
