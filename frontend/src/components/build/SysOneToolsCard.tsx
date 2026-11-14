"use client";
import { AlertTriangle, CheckCircle2, ChevronDown, ChevronRight, Loader2, Plug, Settings2, Sparkles, Square, Wrench, XCircle } from "lucide-react";
import { useMemo, useState } from "react";
import { MCP_NAME_RE, parseArgs, parseEnv, qualifiedToolName, toolParams } from "@/lib/mcp";
import { useStudio } from "@/lib/store";
import { fetchMcpTools, generateToolSamples, stopSysOneGenerate } from "@/lib/sync";
import type { McpServer, McpTool, McpTransport } from "@/lib/types";
import { Badge, Button, Card, Field, NumberInput, SectionHeader, Segmented, cn } from "../ui/kit";

const inputCls = "h-10 w-full rounded-lg border border-border bg-surface px-3 text-sm outline-none placeholder:text-muted/60 focus:border-accent disabled:opacity-50";
const areaCls = "w-full resize-y rounded-lg border border-border bg-surface px-3 py-2 font-mono text-xs leading-relaxed outline-none placeholder:text-muted/60 focus:border-accent";

const TRANSPORTS: { value: McpTransport; label: string; title: string }[] = [
  { value: "stdio", label: "Stdio", title: "Start a local command that speaks MCP on stdin/stdout" },
  { value: "sse", label: "SSE", title: "Connect to a server-sent-events endpoint (older MCP servers)" },
  { value: "http", label: "Streamable HTTP", title: "Connect to a streamable-HTTP endpoint" },
];

/** The small text buttons on a server row. */
function TextBtn({ children, onClick, tone = "muted", disabled, active, title }: { children: React.ReactNode; onClick: () => void; tone?: "muted" | "ok" | "accent" | "danger"; disabled?: boolean; active?: boolean; title?: string }) {
  const color = { muted: "text-muted hover:text-fg", ok: "text-ok", accent: "text-accent", danger: "text-danger" }[tone];
  return (
    <button type="button" onClick={onClick} disabled={disabled} title={title} className={cn("rounded px-1.5 py-1 text-[11px] font-medium transition-colors hover:bg-surface-2 disabled:opacity-40", color, active && "font-semibold underline decoration-2 underline-offset-4")}>
      {children}
    </button>
  );
}

function ToolDetails({ server, tool }: { server: McpServer; tool: McpTool }) {
  const params = toolParams(tool);
  return (
    <details className="group rounded-lg border border-border bg-surface">
      <summary className="flex cursor-pointer list-none items-start gap-2 px-3 py-2">
        <ChevronRight size={13} className="mt-0.5 shrink-0 text-muted transition-transform group-open:rotate-90" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <code className="break-all text-xs font-medium">{qualifiedToolName(server, tool)}</code>
            {tool.readOnly && <Badge tone="ok">read-only</Badge>}
            <Badge>{params.length} arg{params.length === 1 ? "" : "s"}</Badge>
          </div>
          {tool.description && <p className="mt-0.5 line-clamp-2 text-xs text-muted group-open:line-clamp-none">{tool.description}</p>}
        </div>
      </summary>
      <div className="space-y-2 border-t border-border px-3 py-2.5">
        {params.length === 0 ? (
          <p className="text-xs text-muted">This tool takes no arguments.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[420px] text-left text-xs">
              <thead>
                <tr className="text-muted">
                  <th className="py-1 pr-3 font-medium">Argument</th>
                  <th className="py-1 pr-3 font-medium">Type</th>
                  <th className="py-1 font-medium">Description</th>
                </tr>
              </thead>
              <tbody>
                {params.map((p) => (
                  <tr key={p.name} className="border-t border-border align-top">
                    <td className="py-1.5 pr-3">
                      <code className="font-medium">{p.name}</code>
                      {p.required && <span className="ml-1 text-danger" title="required">*</span>}
                    </td>
                    <td className="py-1.5 pr-3 text-muted">{p.type}</td>
                    <td className="py-1.5 text-muted">
                      {p.description}
                      {p.enumValues && <div className="mt-0.5 flex flex-wrap gap-1">{p.enumValues.slice(0, 12).map((v) => <Badge key={v}>{v}</Badge>)}{p.enumValues.length > 12 && <span>+{p.enumValues.length - 12}</span>}</div>}
                      {p.defaultValue !== null && <div className="mt-0.5">default <code>{p.defaultValue}</code></div>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <details>
          <summary className="cursor-pointer text-[11px] text-muted hover:text-fg">Raw input schema</summary>
          <pre className="mt-1.5 max-h-60 overflow-auto rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-snug">{JSON.stringify(tool.inputSchema, null, 2)}</pre>
        </details>
      </div>
    </details>
  );
}

function ServerRow({ server }: { server: McpServer }) {
  const tools = useStudio((s) => s.mcpTools[server.id]);
  const status = useStudio((s) => s.mcpStatus[server.id]);
  const update = useStudio((s) => s.updateMcpServer);
  const remove = useStudio((s) => s.removeMcpServer);
  const conn = useStudio((s) => s.conn);
  const [open, setOpen] = useState(false);
  const busy = !!status?.busy;
  const offline = conn !== "connected";

  return (
    <li className={cn("rounded-xl border border-border bg-surface", !server.enabled && "opacity-70")}>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-3 py-2.5">
        <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-label={open ? "Hide tools" : "Show tools"} className="text-muted hover:text-fg">
          {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        </button>
        <code className="text-sm font-medium">{server.name}</code>
        <Badge>{server.transport === "http" ? "http" : server.transport}</Badge>
        {tools && <Badge tone="accent">{tools.length} tool{tools.length === 1 ? "" : "s"}</Badge>}
        <span className="flex-1" />
        <div className="flex flex-wrap items-center">
          <TextBtn tone={server.enabled ? "ok" : "muted"} active={server.enabled} onClick={() => update(server.id, { enabled: true })} title="Include this server's tools in the dataset">Enable</TextBtn>
          <TextBtn tone="muted" active={!server.enabled} onClick={() => update(server.id, { enabled: false })} title="Leave this server's tools out of the dataset">Disable</TextBtn>
          <TextBtn tone="accent" disabled={busy || offline} onClick={() => fetchMcpTools(server.id, "refresh")} title="Reconnect and read the tool list again">Refresh tools</TextBtn>
          <TextBtn disabled={busy || offline} onClick={() => fetchMcpTools(server.id, "test")} title="Check that the server can be reached">Test</TextBtn>
          <TextBtn active={open} onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Show"}</TextBtn>
          <TextBtn tone="danger" onClick={() => remove(server.id)}>Delete</TextBtn>
        </div>
      </div>

      {status && (
        <div className={cn("flex items-start gap-1.5 border-t border-border px-3 py-1.5 text-xs", status.ok === false ? "text-danger" : "text-muted")}>
          {status.busy ? <Loader2 size={13} className="mt-0.5 shrink-0 animate-spin text-accent" /> : status.ok ? <CheckCircle2 size={13} className="mt-0.5 shrink-0 text-ok" /> : <XCircle size={13} className="mt-0.5 shrink-0" />}
          <span className="min-w-0 whitespace-pre-wrap break-words">
            {status.ok && status.serverName ? `${status.serverName}${status.serverVersion ? ` v${status.serverVersion}` : ""} · ` : ""}
            {status.message}
          </span>
        </div>
      )}

      {open && (
        <div className="space-y-2 border-t border-border px-3 py-3">
          <p className="text-[11px] text-muted">
            {server.transport === "stdio" ? <>Runs <code>{[server.command, ...server.args].join(" ")}</code></> : <>Connects to <code>{server.url}</code></>}
          </p>
          {!tools ? (
            <p className="text-xs text-muted">No tools loaded yet — press “Refresh tools”.</p>
          ) : tools.length === 0 ? (
            <p className="text-xs text-muted">This server reports no tools.</p>
          ) : (
            tools.map((t) => <ToolDetails key={t.name} server={server} tool={t} />)
          )}
        </div>
      )}
    </li>
  );
}

function AddServerForm() {
  const servers = useStudio((s) => s.mcpServers);
  const add = useStudio((s) => s.addMcpServer);
  const [name, setName] = useState("");
  const [transport, setTransport] = useState<McpTransport>("stdio");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [env, setEnv] = useState("");
  const [cwd, setCwd] = useState("");
  const [url, setUrl] = useState("");
  const [timeout, setTimeoutS] = useState("");
  const [connectTimeout, setConnectTimeoutS] = useState("");
  const [error, setError] = useState("");
  const stdio = transport === "stdio";

  const secs = (v: string): number | null | "bad" => {
    if (!v.trim()) return null;
    const n = Number(v);
    return Number.isInteger(n) && n >= 1 && n <= 300 ? n : "bad";
  };

  function submit() {
    const nm = name.trim();
    if (!MCP_NAME_RE.test(nm)) return setError("Name: 1–40 characters — letters, digits, “-” or “_”.");
    if (servers.some((s) => s.name === nm)) return setError(`A server called “${nm}” already exists.`);
    if (stdio && !command.trim()) return setError("Enter the command that starts the server.");
    if (!stdio && !/^https?:\/\/\S+$/.test(url.trim())) return setError("Enter the server URL, starting with http:// or https://.");
    const e = parseEnv(env);
    if (stdio && e.error) return setError(`Environment: ${e.error}`);
    const t = secs(timeout);
    const c = secs(connectTimeout);
    if (t === "bad" || c === "bad") return setError("Timeouts are whole seconds between 1 and 300.");
    setError("");
    const id = `mcp${Date.now().toString(36)}`;
    add({
      id, name: nm, transport,
      command: stdio ? command.trim() : "", args: stdio ? parseArgs(args) : [], env: stdio ? e.env : {}, cwd: stdio ? cwd.trim() : "",
      url: stdio ? "" : url.trim(), timeoutSecs: t, connectTimeoutSecs: c, enabled: true,
    });
    setName(""); setCommand(""); setArgs(""); setEnv(""); setCwd(""); setUrl(""); setTimeoutS(""); setConnectTimeoutS("");
    fetchMcpTools(id, "refresh"); // read its tools right away
  }

  return (
    <div className="space-y-3 border-t border-border pt-4">
      <h4 className="text-sm font-semibold">Add MCP server</h4>
      <Field label="Name" help="Used as the prefix of its tool names, e.g. my-tool__list_rows.">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-tool" className={inputCls} />
      </Field>
      <div>
        <span className="mb-1 block text-xs font-medium text-muted">Transport</span>
        <Segmented value={transport} onChange={setTransport} options={TRANSPORTS} className="flex w-full [&>button]:flex-1 [&>button]:py-1.5 [&>button]:text-xs" />
      </div>
      {stdio ? (
        <>
          <Field label="Command">
            <input value={command} onChange={(e) => setCommand(e.target.value)} placeholder="npx" className={inputCls} />
          </Field>
          <Field label="Arguments (one per line)">
            <textarea value={args} onChange={(e) => setArgs(e.target.value)} rows={3} spellCheck={false} placeholder={"-y\n@my/mcp-server"} className={areaCls} />
          </Field>
          <Field label="Environment (KEY=value, one per line)">
            <textarea value={env} onChange={(e) => setEnv(e.target.value)} rows={3} spellCheck={false} placeholder="API_KEY=..." className={areaCls} />
          </Field>
          <Field label="Working directory (optional)">
            <input value={cwd} onChange={(e) => setCwd(e.target.value)} placeholder="/path" className={inputCls} />
          </Field>
        </>
      ) : (
        <Field label={transport === "sse" ? "SSE URL" : "Server URL"}>
          <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder={transport === "sse" ? "http://localhost:8000/sse" : "http://localhost:8000/mcp"} className={inputCls} />
        </Field>
      )}
      <div className="grid grid-cols-2 gap-3">
        <Field label="Timeout (s) (optional)" help="How long listing the tools may take. Default 30.">
          <input value={timeout} onChange={(e) => setTimeoutS(e.target.value)} inputMode="numeric" className={inputCls} />
        </Field>
        <Field label="Connect timeout (s) (optional)" help="How long starting / connecting may take. Default 20.">
          <input value={connectTimeout} onChange={(e) => setConnectTimeoutS(e.target.value)} inputMode="numeric" className={inputCls} />
        </Field>
      </div>
      {error && <p role="alert" className="text-xs text-danger">{error}</p>}
      <Button variant="primary" className="h-10 w-full" onClick={submit}><Plug size={14} /> Add MCP server</Button>
    </div>
  );
}


/** Pick tools, and have the model write requests + calls for each (several tools at once). */
function ToolSamplesPanel() {
  const servers = useStudio((s) => s.mcpServers);
  const toolsBy = useStudio((s) => s.mcpTools);
  const llm = useStudio((s) => s.llm);
  const conn = useStudio((s) => s.conn);
  const progress = useStudio((s) => s.sysoneGenProgress);
  const setSettingsOpen = useStudio((s) => s.setSettingsOpen);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [perTool, setPerTool] = useState(10);
  const [parallel, setParallel] = useState(4);
  const [others, setOthers] = useState(3);
  const [error, setError] = useState("");
  const [started, setStarted] = useState(false);

  // every tool of the enabled servers, by qualified name
  const groups = useMemo(
    () => servers.filter((s) => s.enabled).map((s) => ({ server: s, tools: (toolsBy[s.id] ?? []).map((t) => ({ tool: t, name: qualifiedToolName(s, t) })) })).filter((g) => g.tools.length > 0),
    [servers, toolsBy],
  );
  const all = groups.flatMap((g) => g.tools.map((t) => t.name));
  const chosen = all.filter((n) => picked.has(n));
  const running = progress?.state === "running";
  const llmReady = !!(llm.baseUrl.trim() && llm.model.trim());
  const expected = chosen.length * perTool;

  const toggle = (name: string) => setPicked((p) => { const n = new Set(p); if (n.has(name)) n.delete(name); else n.add(name); return n; });
  const setMany = (names: string[], on: boolean) => setPicked((p) => { const n = new Set(p); for (const x of names) { if (on) n.add(x); else n.delete(x); } return n; });

  async function start() {
    setError("");
    setStarted(true);
    const err = await generateToolSamples(chosen, perTool, others, parallel);
    if (err) setError(err);
  }

  if (groups.length === 0) {
    return <p className="rounded-lg bg-surface-2 px-3 py-3 text-xs text-muted">Once an enabled server has tools, you can pick them here and generate training samples for them.</p>;
  }

  return (
    <div className="space-y-3 border-t border-border pt-4">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="text-sm font-semibold">Generate tool-calling samples</h4>
        <Badge>{chosen.length} of {all.length} tools picked</Badge>
        <span className="flex-1" />
        <Button size="sm" disabled={running} onClick={() => setMany(all, true)}>All</Button>
        <Button size="sm" disabled={running} onClick={() => setMany(all, false)}>None</Button>
      </div>
      <p className="text-xs text-muted">
        For each picked tool the model writes realistic requests with the right arguments. Each becomes one sample with questions of all three kinds: <b className="font-medium text-fg">choice</b> (which tool, which value), <b className="font-medium text-fg">yes/no</b> (right tool? argument given? true?) and a <b className="font-medium text-fg">score</b> (how completely the request specifies the call).
      </p>

      <div className="max-h-72 space-y-2 overflow-y-auto rounded-lg border border-border p-2">
        {groups.map((g) => {
          const names = g.tools.map((t) => t.name);
          const n = names.filter((x) => picked.has(x)).length;
          return (
            <div key={g.server.id}>
              <div className="flex items-center gap-2 px-1 py-1">
                <input type="checkbox" checked={n === names.length} ref={(el) => { if (el) el.indeterminate = n > 0 && n < names.length; }} disabled={running} onChange={(e) => setMany(names, e.target.checked)} aria-label={`Pick every tool of ${g.server.name}`} className="accent-[var(--accent)]" />
                <code className="text-xs font-semibold">{g.server.name}</code>
                <span className="text-[11px] text-muted">{n}/{names.length}</span>
              </div>
              <ul className="ml-5 divide-y divide-border">
                {g.tools.map(({ tool, name }) => (
                  <li key={name}>
                    <label className={cn("flex cursor-pointer items-start gap-2 px-1 py-1.5 text-xs hover:bg-surface-2", running && "cursor-default opacity-70")}>
                      <input type="checkbox" checked={picked.has(name)} disabled={running} onChange={() => toggle(name)} className="mt-0.5 accent-[var(--accent)]" />
                      <span className="min-w-0 flex-1">
                        <code className="break-all font-medium">{tool.name}</code>
                        {tool.description && <span className="line-clamp-1 block text-muted" title={tool.description}>{tool.description}</span>}
                      </span>
                      <Badge>{toolParams(tool).length} arg{toolParams(tool).length === 1 ? "" : "s"}</Badge>
                    </label>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Field label="Samples per tool" help="Requests written for each picked tool.">
          <NumberInput value={perTool} onChange={setPerTool} min={1} max={50} disabled={running} />
        </Field>
        <Field label="Tools at once" help="Tools processed in parallel — one model request each.">
          <NumberInput value={parallel} onChange={setParallel} min={1} max={8} disabled={running} />
        </Field>
        <Field label="Wrong tools shown" help="Other tools offered next to the right one in the “which tool” question.">
          <NumberInput value={others} onChange={setOthers} min={0} max={6} disabled={running} />
        </Field>
      </div>

      {!llmReady && (
        <div className="flex items-center gap-2 rounded-lg border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-warn">
          <AlertTriangle size={14} className="shrink-0" />
          <span className="flex-1">Connect a model server first (URL and model name).</span>
          <Button size="sm" onClick={() => setSettingsOpen(true)}><Settings2 size={13} /> Settings</Button>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {running ? (
          <Button variant="danger" onClick={() => stopSysOneGenerate()}><Square size={13} fill="currentColor" /> Stop</Button>
        ) : (
          <Button variant="primary" disabled={conn !== "connected" || !llmReady || chosen.length === 0} onClick={start}>
            <Sparkles size={14} /> Generate ≈ {expected.toLocaleString()} samples
          </Button>
        )}
        {chosen.length > 0 && <span className="text-xs text-muted">{chosen.length} tool{chosen.length === 1 ? "" : "s"}, {Math.min(parallel, chosen.length)} at a time</span>}
      </div>
      {error && <p className="text-xs text-danger">{error}</p>}

      {started && progress && (
        <div role="status" className="rounded-lg border border-border bg-surface-2/50 px-3 py-2.5">
          <div className="flex items-center gap-2 text-sm font-medium">
            {running && <Loader2 size={14} className="animate-spin text-accent" />}
            {running ? "Writing tool-calling samples…" : progress.state === "done" ? "Finished" : progress.state === "stopped" ? "Stopped" : "Failed"}
            <span className="ml-auto text-xs font-normal tabular-nums text-muted">{progress.done}/{progress.total} tools · {progress.samples} samples{progress.failed ? ` · ${progress.failed} failed` : ""}</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2">
            <div className={cn("h-full rounded-full transition-[width] duration-200", progress.state === "error" ? "bg-danger" : "bg-accent")} style={{ width: `${progress.total ? Math.max(2, (progress.done / progress.total) * 100) : 0}%` }} />
          </div>
          {progress.message && <p className={cn("mt-1.5 break-words text-xs", progress.state === "error" || progress.failed ? "text-danger" : "text-muted")}>{progress.message}</p>}
          {progress.state === "done" && progress.samples > 0 && <p className="mt-1.5 text-xs text-muted">The samples are in the System One dataset above — train on them with Run.</p>}
        </div>
      )}
    </div>
  );
}

export function SysOneToolsCard() {
  const servers = useStudio((s) => s.mcpServers);
  const tools = useStudio((s) => s.mcpTools);
  const nTools = servers.filter((s) => s.enabled).reduce((n, s) => n + (tools[s.id]?.length ?? 0), 0);

  return (
    <Card>
      <SectionHeader
        title="Tool-calling dataset"
        sub="Connect MCP servers to read their tool definitions — the catalogue that tool-call training data is written against."
        icon={<Wrench size={16} className="text-accent" />}
        right={nTools > 0 ? <Badge tone="accent">{nTools} tool{nTools === 1 ? "" : "s"}</Badge> : undefined}
      />
      <div className="space-y-4 px-4 pb-4">
        {servers.length === 0 ? (
          <p className="rounded-lg bg-surface-2 px-3 py-3 text-xs text-muted">No MCP servers yet. Add one below — its tools are listed as soon as it connects.</p>
        ) : (
          <ul className="space-y-2">{servers.map((s) => <ServerRow key={s.id} server={s} />)}</ul>
        )}
        <ToolSamplesPanel />
        <AddServerForm />
      </div>
    </Card>
  );
}
