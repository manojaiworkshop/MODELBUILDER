"use client";
import { Brain, Check, Copy, Loader2, Play, Plus, Square, Trash2, Undo2, Wrench, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { callToolWithModel, inferSysOne, startSysOneTrain, stopSysOneTrain } from "@/lib/sync";
import { isRunning, useStudio } from "@/lib/store";
import type { SysOneQuestion, ToolCallResult } from "@/lib/types";
import { Badge, Button, Card, SectionHeader, cn } from "../ui/kit";

const H = 180;
const M = { l: 40, r: 12, t: 10, b: 20 };

function niceMax(v: number) {
  const p = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-6))));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

/** A minimal self-contained loss chart — System One's step shape (noul/opt/conf breakdown) doesn't
 * match the causal model's `LossChart`, so this is its own small SVG rather than a shared one. */
function SysOneLossChart() {
  const steps = useStudio((s) => s.sysoneSteps);
  // fill the card: track the container's width (the SVG is drawn in real pixels)
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(520);
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(260, el.clientWidth)));
    ro.observe(el);
    setW(Math.max(260, el.clientWidth));
    return () => ro.disconnect();
  }, []);

  const { path, yMax, xMax, last } = useMemo(() => {
    if (steps.length === 0) return { path: "", yMax: 1, xMax: 1, last: null };
    const maxPts = 400;
    const size = Math.max(1, Math.ceil(steps.length / maxPts));
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i < steps.length; i += size) {
      const chunk = steps.slice(i, i + size).map((s) => s.loss).filter((v) => Number.isFinite(v));
      if (chunk.length) pts.push({ x: steps[Math.min(i + size, steps.length) - 1].step, y: chunk.reduce((a, b) => a + b, 0) / chunk.length });
    }
    const smooth = pts.reduce<{ x: number; y: number }[]>((acc, p) => {
      const prev = acc.at(-1);
      acc.push({ x: p.x, y: prev ? prev.y * 0.85 + p.y * 0.15 : p.y });
      return acc;
    }, []);
    const yMax = niceMax(Math.max(0.1, ...smooth.map((p) => p.y)) * 1.1);
    // Scaled to progress so far (not the full planned run length): with long real-data runs the
    // line would otherwise sit squeezed in the first few percent of the width for most of
    // training — the step/percent counter above already says how far there is to go.
    const xMax = Math.max(steps[steps.length - 1].step, 10);
    return { path: smooth, yMax, xMax, last: steps[steps.length - 1] };
  }, [steps]);

  const iw = w - M.l - M.r;
  const ih = H - M.t - M.b;
  const X = (x: number) => M.l + (x / xMax) * iw;
  const Y = (y: number) => M.t + ih - (Math.min(y, yMax) / yMax) * ih;
  const d = Array.isArray(path) ? path.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("") : "";
  const ticksY = [0, 0.5, 1].map((f) => f * yMax);

  return (
    <div ref={wrap} className="relative w-full">
      <svg width={w} height={H} role="img" aria-label="Training loss over steps" className="max-w-full">
        {ticksY.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={w - M.r} y1={Y(t)} y2={Y(t)} stroke="var(--border)" strokeDasharray={t === 0 ? undefined : "3 4"} />
            <text x={M.l - 6} y={Y(t) + 3.5} textAnchor="end" fontSize="10" fill="var(--muted)">{t.toFixed(2)}</text>
          </g>
        ))}
        {steps.length > 0 && <path d={d} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />}
        {last && <circle cx={X(last.step)} cy={Y(Math.min(last.loss, yMax))} r="4" fill="var(--accent)" className="live-dot" />}
      </svg>
      {steps.length === 0 && <div className="absolute inset-0 flex items-center justify-center text-sm text-muted">Press Run to start training</div>}
    </div>
  );
}

function StatusCard() {
  const status = useStudio((s) => s.sysoneStatus);
  const running = isRunning(status);
  const steps = useStudio((s) => s.sysoneSteps);
  const last = steps.at(-1);
  const pct = status.total ? Math.min(100, (status.step / status.total) * 100) : 0;

  return (
    <Card>
      <SectionHeader
        title="Live training"
        sub="Decision-model loss: yes/no + choice/score cross-entropy, plus a weighted confidence term."
        right={
          running ? (
            <Button variant="danger" size="sm" onClick={() => stopSysOneTrain()}><Square size={12} fill="currentColor" /> Stop</Button>
          ) : (
            <Button variant="primary" size="sm" onClick={() => startSysOneTrain()}><Play size={12} fill="currentColor" /> Run</Button>
          )
        }
      />
      <div className="px-4 pb-2 text-xs text-muted">
        {running && <Loader2 size={12} className="mr-1 inline animate-spin text-accent" />}
        step {status.step} / {status.total || "?"} ({pct.toFixed(0)}%)
        {last && <> · loss {last.loss.toFixed(3)} (noul {last.noul.toFixed(3)} · opt {last.opt.toFixed(3)} · conf {last.conf.toFixed(3)})</>}
        {status.message && <span className={cn("block", status.state === "error" ? "text-danger" : "text-warn")}>{status.message}</span>}
      </div>
      <div className="px-4 pb-4">
        <SysOneLossChart />
      </div>
    </Card>
  );
}

function pct(n: number) {
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—";
}

function EvalsCard() {
  const evals = useStudio((s) => s.sysoneEvals);
  if (evals.length === 0) return null;
  return (
    <Card>
      <SectionHeader title="Validation per epoch" sub="Accuracy and calibration (ECE) on the held-out 10% of samples. The best-calibrated epoch's weights are kept." />
      <div className="overflow-x-auto px-4 pb-4">
        <table className="w-full min-w-[480px] text-left text-xs">
          <thead>
            <tr className="text-muted">
              <th className="py-1 pr-3 font-medium">Epoch</th>
              <th className="py-1 pr-3 font-medium">Yes/no</th>
              <th className="py-1 pr-3 font-medium">Choice</th>
              <th className="py-1 pr-3 font-medium">Score</th>
              <th className="py-1 pr-3 font-medium">ECE</th>
            </tr>
          </thead>
          <tbody>
            {evals.map((e) => (
              <tr key={e.epoch} className={cn("border-t border-border", e.best && "text-accent")}>
                <td className="py-1 pr-3 tabular-nums">{e.epoch}{e.best && <Badge tone="accent" className="ml-1.5">best</Badge>}</td>
                <td className="py-1 pr-3 tabular-nums">{pct(e.report.noulAcc)}</td>
                <td className="py-1 pr-3 tabular-nums">{pct(e.report.choiceAcc)}</td>
                <td className="py-1 pr-3 tabular-nums">{pct(e.report.scoreAcc)}</td>
                <td className="py-1 pr-3 tabular-nums">{e.report.probEce.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

let qid = 0;
type DraftQ = SysOneQuestion & { id: number };

function AnswerRow({ q, a }: { q: DraftQ; a: import("@/lib/types").SysOneAnswer }) {
  if (q.type === "noul" && a.type === "noul") {
    return <li className="text-xs"><span className="font-medium">{q.text}</span> — P(true) = <span className="text-accent">{pct(a.p)}</span> <span className="text-muted">(confidence {pct(a.confidence)})</span></li>;
  }
  if ((q.type === "choice" || q.type === "score") && (a.type === "choice" || a.type === "score")) {
    const labels = q.type === "choice" ? q.options : q.levels;
    return (
      <li className="text-xs">
        <span className="font-medium">{q.text}</span>
        <ul className="ml-3 mt-0.5 space-y-0.5">
          {a.probs.map((p, i) => (
            <li key={i} className={cn(i === (a.type === "choice" ? a.best : a.probs.indexOf(Math.max(...a.probs))) ? "text-accent" : "text-muted")}>
              {labels[i] ?? `option ${i}`}: {pct(p)}
            </li>
          ))}
        </ul>
        <span className="text-muted">confidence {pct(a.confidence)}{a.type === "score" && <> · expected score {a.score.toFixed(2)}</>}</span>
      </li>
    );
  }
  return null;
}

function InferCard() {
  const trained = useStudio((s) => s.sysoneTrained);
  const infer = useStudio((s) => s.sysoneInfer);
  const [state, setState] = useState("");
  const [questions, setQuestions] = useState<DraftQ[]>([{ id: qid++, type: "noul", text: "" }]);

  // Snapshots from before each add / remove (question or option), so "Undo" can bring it back.
  // Typing in a field is not a step of its own.
  const [history, setHistory] = useState<DraftQ[][]>([]);
  const change = (next: (qs: DraftQ[]) => DraftQ[]) => {
    setHistory((h) => [...h.slice(-29), questions]);
    setQuestions(next(questions));
  };
  const undo = () => {
    const prev = history.at(-1);
    if (!prev) return;
    setHistory((h) => h.slice(0, -1));
    setQuestions(prev);
  };

  const addQ = (type: SysOneQuestion["type"]) =>
    change((qs) => [...qs, type === "noul" ? { id: qid++, type, text: "" } : type === "choice" ? { id: qid++, type, text: "", options: ["", ""] } : { id: qid++, type, text: "", levels: ["low", "high"] }]);
  const removeQ = (id: number) => change((qs) => qs.filter((q) => q.id !== id));
  const updateQ = (id: number, patch: Partial<DraftQ>) => setQuestions((qs) => qs.map((q) => (q.id === id ? ({ ...q, ...patch } as DraftQ) : q)));
  const labelsOf = (q: DraftQ) => (q.type === "choice" ? q.options : q.type === "score" ? q.levels : []);
  const keyOf = (q: DraftQ) => (q.type === "choice" ? "options" : "levels");
  const addOption = (q: DraftQ) => change((qs) => qs.map((x) => (x.id === q.id ? ({ ...x, [keyOf(q)]: [...labelsOf(q), ""] } as DraftQ) : x)));
  const removeOption = (q: DraftQ, i: number) => change((qs) => qs.map((x) => (x.id === q.id ? ({ ...x, [keyOf(q)]: labelsOf(q).filter((_, j) => j !== i) } as DraftQ) : x)));

  const ready = state.trim().length > 0 && questions.length > 0 && questions.every((q) => q.text.trim() && (q.type === "noul" || (q.type === "choice" ? q.options : q.levels).every((o) => o.trim())));

  async function ask() {
    const clean: SysOneQuestion[] = questions.map((q) => (q.type === "noul" ? { type: q.type, text: q.text } : q.type === "choice" ? { type: q.type, text: q.text, options: q.options } : { type: q.type, text: q.text, levels: q.levels }));
    await inferSysOne(state, clean);
  }

  if (!trained) {
    return (
      <Card>
        <SectionHeader title="Ask it" icon={<Brain size={16} className="text-accent" />} sub="Train a model first — this asks the last trained System One model directly." />
      </Card>
    );
  }

  return (
    <Card>
      <SectionHeader title="Ask it" icon={<Brain size={16} className="text-accent" />} sub="One forward pass answers every question below, with a calibrated probability and confidence for each." />
      <div className="space-y-3 px-4 pb-4">
        <textarea
          value={state}
          onChange={(e) => setState(e.target.value)}
          placeholder="Describe a situation — a ticket, a message, a scenario…"
          rows={2}
          className="w-full resize-none rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <div className="space-y-2">
          {questions.map((q) => (
            <div key={q.id} className="rounded-lg border border-border p-2.5">
              <div className="flex items-center gap-2">
                <input value={q.text} onChange={(e) => updateQ(q.id, { text: e.target.value })} placeholder={q.type === "noul" ? "a yes/no statement…" : "the question…"} className="h-8 min-w-0 flex-1 rounded-md border border-border bg-surface px-2 text-xs outline-none focus:border-accent" />
                <Badge>{q.type}</Badge>
                <button onClick={() => removeQ(q.id)} aria-label="Remove question" className="rounded p-1 text-muted hover:bg-surface-2 hover:text-danger"><Trash2 size={13} /></button>
              </div>
              {(q.type === "choice" || q.type === "score") && (
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {labelsOf(q).map((o, i) => (
                    <span key={i} className="group relative">
                      <input
                        value={o}
                        onChange={(e) => {
                          const list = [...labelsOf(q)];
                          list[i] = e.target.value;
                          updateQ(q.id, { [keyOf(q)]: list } as Partial<DraftQ>);
                        }}
                        placeholder={q.type === "choice" ? `option ${i + 1}` : `level ${i + 1}`}
                        className="h-7 w-28 rounded border border-border bg-surface pl-1.5 pr-6 text-[11px] outline-none focus:border-accent"
                      />
                      <button
                        onClick={() => removeOption(q, i)}
                        disabled={labelsOf(q).length <= 2}
                        aria-label={`Remove ${q.type === "choice" ? "option" : "level"} ${i + 1}`}
                        title={labelsOf(q).length <= 2 ? "A question needs at least 2" : "Remove"}
                        className="absolute right-0.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-muted hover:bg-surface-2 hover:text-danger disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-muted"
                      >
                        <X size={11} />
                      </button>
                    </span>
                  ))}
                  <button
                    onClick={() => addOption(q)}
                    disabled={labelsOf(q).length >= 10}
                    className="flex h-7 items-center gap-1 rounded border border-dashed border-border px-1.5 text-[11px] text-muted hover:border-accent hover:text-accent disabled:opacity-40"
                  >
                    <Plus size={11} /> {q.type === "choice" ? "option" : "level"}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" onClick={() => addQ("noul")}><Plus size={12} /> yes/no</Button>
          <Button size="sm" onClick={() => addQ("choice")}><Plus size={12} /> choice</Button>
          <Button size="sm" onClick={() => addQ("score")}><Plus size={12} /> score</Button>
          <Button size="sm" variant="ghost" disabled={history.length === 0} onClick={undo} title="Undo the last add / remove of a question or option"><Undo2 size={12} /> Undo</Button>
          <span className="flex-1" />
          <Button variant="primary" size="sm" disabled={!ready || infer.running} onClick={ask}>
            {infer.running ? <Loader2 size={12} className="animate-spin" /> : <Brain size={12} />} Decide
          </Button>
        </div>
        {infer.error && <p className="text-xs text-danger">{infer.error}</p>}
        {infer.answers && (
          <ul className="space-y-2 rounded-lg border border-border bg-surface-2/50 p-3">
            {infer.answers.map((a, i) => (questions[i] ? <AnswerRow key={i} q={questions[i]} a={a} /> : null))}
          </ul>
        )}
      </div>
    </Card>
  );
}

/** Request in, tool call out: the trained model picks a tool and fills in its arguments. */
function ToolCallCard() {
  const trained = useStudio((s) => s.sysoneTrained);
  const nTools = useStudio((s) => s.mcpServers.filter((x) => x.enabled).reduce((n, x) => n + (s.mcpTools[x.id]?.length ?? 0), 0));
  const [request, setRequest] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ToolCallResult | null>(null);
  const [copied, setCopied] = useState(false);

  if (!trained) {
    return (
      <Card>
        <SectionHeader title="Call a tool" icon={<Wrench size={16} className="text-accent" />} sub="Train a model on a tool-calling dataset first — then give it a request and it picks the tool and fills in the arguments." />
      </Card>
    );
  }

  const text = result ? `[tool_calls]\n${JSON.stringify(result.toolCalls, null, 2)}` : "";

  async function run() {
    setBusy(true);
    setError("");
    const res = await callToolWithModel(request.trim());
    setBusy(false);
    if (res.ok) setResult(res.result);
    else {
      setResult(null);
      setError(res.error);
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable: the text is selectable */
    }
  }

  return (
    <Card>
      <SectionHeader title="Call a tool" icon={<Wrench size={16} className="text-accent" />} sub={`The model reads the request, picks one of your ${nTools} enabled tool${nTools === 1 ? "" : "s"}, and fills in its arguments.`} />
      <div className="space-y-3 px-4 pb-4">
        <textarea
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && request.trim() && !busy) run(); }}
          placeholder="e.g. show me the last 20 IFC tracks, newest first"
          rows={2}
          className="w-full resize-none rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <div className="flex items-center gap-2">
          <Button variant="primary" size="sm" disabled={!request.trim() || busy || nTools === 0} onClick={run}>
            {busy ? <Loader2 size={12} className="animate-spin" /> : <Wrench size={12} />} Call
          </Button>
          {nTools === 0 && <span className="text-xs text-warn">Enable an MCP server with tools in the Data tab first.</span>}
        </div>
        {error && <p className="text-xs text-danger">{error}</p>}

        {result && (
          <div className="space-y-3">
            <div className="relative">
              <pre className="max-h-72 overflow-auto rounded-lg border border-border bg-surface-2 p-3 pr-16 font-mono text-[11px] leading-snug">{text}</pre>
              <Button size="sm" className="absolute right-2 top-2" onClick={copy}>{copied ? <Check size={12} /> : <Copy size={12} />} {copied ? "Copied" : "Copy"}</Button>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs">
              <code className="break-all font-medium">{result.tool}</code>
              <Badge tone={result.toolConfidence >= 0.6 ? "ok" : "warn"} title="probability of this tool in the “which tool” choice">{pct(result.toolConfidence)} sure</Badge>
              {result.specificity && <Badge title="how completely the request specifies the call">request: {result.specificity}</Badge>}
            </div>

            {result.missing.length > 0 && (
              <p className={cn("rounded-lg border px-3 py-2 text-xs", result.missing.some((m) => m.required) ? "border-warn/40 bg-warn/5 text-warn" : "border-border text-muted")}>
                Left out — free-text values the model can’t produce yet:{" "}
                {result.missing.map((m, i) => (
                  <span key={m.path}>{i > 0 && ", "}<code>{m.path}</code>{m.required && " (required)"}</span>
                ))}
                . Add them to the call yourself.
              </p>
            )}

            <details>
              <summary className="cursor-pointer text-xs font-medium text-muted hover:text-fg">Tool candidates ({result.candidates.length})</summary>
              <table className="mt-1.5 w-full text-left text-xs">
                <thead><tr className="text-muted"><th className="py-1 pr-3 font-medium">Tool</th><th className="py-1 pr-3 font-medium">Right tool?</th><th className="py-1 font-medium">Choice</th></tr></thead>
                <tbody>
                  {result.candidates.slice(0, 8).map((c) => (
                    <tr key={c.name} className={cn("border-t border-border", c.name === result.tool && "text-accent")}>
                      <td className="break-all py-1 pr-3 font-mono">{c.name}</td>
                      <td className="py-1 pr-3 tabular-nums">{pct(c.relevance)}</td>
                      <td className="py-1 tabular-nums">{c.choice === null ? "—" : pct(c.choice)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>

            <details>
              <summary className="cursor-pointer text-xs font-medium text-muted hover:text-fg">Argument decisions ({result.decisions.length})</summary>
              <ul className="mt-1.5 space-y-1 text-xs">
                {result.decisions.map((d, i) => (
                  <li key={i} className={cn(!d.used && "opacity-50")}>
                    <code>{d.path}</code> <span className="text-muted">{d.question.split(": ")[1] ?? d.question}</span> → <b className="font-medium">{d.answer}</b>{" "}
                    <span className="tabular-nums text-muted">{pct(d.p)} · confidence {pct(d.confidence)}{!d.used && " · not used"}</span>
                  </li>
                ))}
              </ul>
            </details>
          </div>
        )}
      </div>
    </Card>
  );
}

export function SysOneObservePanel() {
  return (
    <div className="space-y-4 overflow-y-auto p-4">
      <StatusCard />
      <EvalsCard />
      <InferCard />
      <ToolCallCard />
    </div>
  );
}
