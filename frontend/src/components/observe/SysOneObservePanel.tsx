"use client";
import { Brain, Loader2, Play, Plus, Square, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { inferSysOne, startSysOneTrain, stopSysOneTrain } from "@/lib/sync";
import { isRunning, useStudio } from "@/lib/store";
import type { SysOneQuestion } from "@/lib/types";
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
  const [w] = useState(520);

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
    <div className="relative w-full">
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

  const addQ = (type: SysOneQuestion["type"]) =>
    setQuestions((qs) => [...qs, type === "noul" ? { id: qid++, type, text: "" } : type === "choice" ? { id: qid++, type, text: "", options: ["", ""] } : { id: qid++, type, text: "", levels: ["low", "high"] }]);
  const removeQ = (id: number) => setQuestions((qs) => qs.filter((q) => q.id !== id));
  const updateQ = (id: number, patch: Partial<DraftQ>) => setQuestions((qs) => qs.map((q) => (q.id === id ? ({ ...q, ...patch } as DraftQ) : q)));

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
                  {(q.type === "choice" ? q.options : q.levels).map((o, i) => (
                    <input
                      key={i}
                      value={o}
                      onChange={(e) => {
                        const key = q.type === "choice" ? "options" : "levels";
                        const list = [...(q.type === "choice" ? q.options : q.levels)];
                        list[i] = e.target.value;
                        updateQ(q.id, { [key]: list } as Partial<DraftQ>);
                      }}
                      placeholder={q.type === "choice" ? `option ${i + 1}` : `level ${i + 1}`}
                      className="h-7 w-24 rounded border border-border bg-surface px-1.5 text-[11px] outline-none focus:border-accent"
                    />
                  ))}
                  <button
                    onClick={() => {
                      const key = q.type === "choice" ? "options" : "levels";
                      const list = [...(q.type === "choice" ? q.options : q.levels), ""];
                      updateQ(q.id, { [key]: list } as Partial<DraftQ>);
                    }}
                    className="flex h-7 items-center gap-1 rounded border border-dashed border-border px-1.5 text-[11px] text-muted hover:border-accent hover:text-accent"
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

export function SysOneObservePanel() {
  return (
    <div className="space-y-4 overflow-y-auto p-4">
      <StatusCard />
      <EvalsCard />
      <InferCard />
    </div>
  );
}
