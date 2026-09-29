"use client";
import { Play, Square } from "lucide-react";
import { fmtParams } from "@/lib/blocks";
import { isRunning, useStudio } from "@/lib/store";
import { startTraining, stopTraining } from "@/lib/sync";
import { Badge, Button, Card, SectionHeader, cn } from "../ui/kit";
import { LossChart } from "./LossChart";

function Tile({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="rounded-lg bg-surface-2 px-3 py-2">
      <div className="text-[11px] text-muted">{label}</div>
      <div className={cn("text-lg font-semibold leading-tight tabular-nums", tone)}>{value}</div>
      {sub && <div className="text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

const STATE_TONE = { idle: "neutral", preparing: "accent", running: "ok", stopped: "warn", done: "ok", error: "danger" } as const;

export function TrainingCard() {
  const status = useStudio((s) => s.status);
  const series = useStudio((s) => s.series);
  const rev = useStudio((s) => s.series.rev);
  const inspect = useStudio((s) => s.inspect);
  const running = isRunning(status);
  const conn = useStudio((s) => s.conn);
  const blocking = !!inspect && !inspect.ok;
  void rev;

  const n = series.loss.length;
  const last = n ? series.loss[n - 1] : undefined;
  const recent = series.loss.slice(-10).filter((v) => v !== undefined);
  const avg = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : undefined;
  const ev = series.evals.at(-1);
  const tps = series.tps.slice(-10).filter(Boolean);
  const speed = tps.length ? tps.reduce((a, b) => a + b, 0) / tps.length : undefined;
  const pct = status.total ? Math.min(100, (status.step / status.total) * 100) : 0;
  const bestVal = series.evals.length ? Math.min(...series.evals.map((e) => e.valLoss)) : undefined;
  const overfit = series.evals.length >= 3 && ev !== undefined && bestVal !== undefined && avg !== undefined && ev.valLoss > bestVal * 1.25 && ev.valLoss > avg * 1.5;

  return (
    <Card>
      <SectionHeader
        title="Live training"
        sub={status.message || "Build a model on the left, then press Run."}
        right={
          <div className="flex items-center gap-2">
            <Badge tone={STATE_TONE[status.state]}>
              {running && <span className="live-dot h-1.5 w-1.5 rounded-full bg-ok" />}
              {status.state}
            </Badge>
            {running ? (
              <Button variant="danger" size="sm" onClick={() => stopTraining()}><Square size={12} fill="currentColor" /> Stop</Button>
            ) : (
              <Button variant="primary" size="sm" disabled={conn !== "connected" || blocking} onClick={() => startTraining()} title={blocking ? "Fix the model first" : undefined}>
                <Play size={12} fill="currentColor" /> Run
              </Button>
            )}
          </div>
        }
      />
      <div className="px-4 pb-1">
        <div className="h-1.5 overflow-hidden rounded-full bg-surface-2" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
          <div className={cn("h-full rounded-full bg-accent transition-[width] duration-300", running && "opacity-90")} style={{ width: `${pct}%` }} />
        </div>
        <div className="mt-1 flex justify-between text-[11px] tabular-nums text-muted">
          <span>step {status.step.toLocaleString()} / {(status.total || 0).toLocaleString()}</span>
          <span>{pct.toFixed(0)}%</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 px-4 pb-3 pt-2 @lg:grid-cols-4 lg:grid-cols-4">
        <Tile label="Train loss" value={avg !== undefined ? avg.toFixed(3) : "—"} sub={last !== undefined ? `last ${last.toFixed(3)}` : undefined} tone="text-accent" />
        <Tile label="Validation loss" value={ev ? ev.valLoss.toFixed(3) : "—"} sub={ev ? `perplexity ${ev.perplexity.toFixed(1)}` : undefined} tone="text-ok" />
        <Tile label="Speed" value={speed ? `${Math.round(speed).toLocaleString()}` : "—"} sub="tokens / sec" />
        <Tile label="Parameters" value={inspect ? fmtParams(inspect.totalParams) : "—"} sub={inspect ? `vocab ${inspect.vocabSize}` : undefined} />
      </div>
      <LossChart />
      {overfit && (
        <p className="mx-4 mb-3 rounded-lg border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-warn">
          <b>Overfitting:</b> training loss keeps falling while validation loss rises — the model is memorizing your text instead of learning general patterns. More data, dropout or fewer steps will help.
        </p>
      )}
      {status.state === "error" && <p className="border-t border-border px-4 py-3 text-sm text-danger">{status.message}</p>}
    </Card>
  );
}

/** Slim always-visible progress + loss readout for the Observe header. */
export function MiniStatus() {
  const status = useStudio((s) => s.status);
  const rev = useStudio((s) => s.series.rev);
  const series = useStudio((s) => s.series);
  void rev;
  if (status.state === "idle") return null;
  const recent = series.loss.slice(-10).filter((v) => v !== undefined);
  const avg = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : undefined;
  const ev = series.evals.at(-1);
  const pct = status.total ? Math.min(100, (status.step / status.total) * 100) : 0;
  return (
    <div className="flex items-center gap-3 text-xs tabular-nums">
      <span className="text-muted">step {status.step}/{status.total}</span>
      {avg !== undefined && <span className="text-accent">train {avg.toFixed(3)}</span>}
      {ev && <span className="text-ok">val {ev.valLoss.toFixed(3)}</span>}
      <div className="hidden h-1.5 w-24 overflow-hidden rounded-full bg-surface-2 sm:block"><div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${pct}%` }} /></div>
    </div>
  );
}
