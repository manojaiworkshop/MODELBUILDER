"use client";
import { Activity } from "lucide-react";
import { isRunning, useStudio } from "@/lib/store";
import { Badge } from "../ui/kit";
import { EmbeddingMap } from "./EmbeddingMap";
import { GenerateBox } from "./GenerateBox";
import { GradientFlow } from "./GradientFlow";
import { MiniStatus, TrainingCard } from "./Metrics";
import { NetworkFlow } from "./NetworkFlow";
import { TokenTable } from "./TokenTable";

function ErrorBanner() {
  const status = useStudio((s) => s.status);
  const clear = useStudio((s) => s.setStatus);
  if (status.state !== "error") return null;
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-danger/40 bg-danger/5 px-4 py-3 text-sm text-danger">
      <span className="min-w-0 flex-1"><b>Training stopped.</b> {status.message}</span>
      <button onClick={() => clear({ state: "idle", message: "", step: 0, total: 0 })} className="shrink-0 text-xs underline">Dismiss</button>
    </div>
  );
}

function Steps() {
  const idle = useStudio((s) => s.status.state === "idle");
  if (!idle) return null;
  const steps = [
    ["1", "Build", "Drag blocks into the stack on the left, or load a preset."],
    ["2", "Feed", "Pick or upload text in the Data tab."],
    ["3", "Run", "Press Run and watch loss, attention and embeddings change live."],
  ];
  return (
    <ol className="grid gap-2 sm:grid-cols-3">
      {steps.map(([n, t, d]) => (
        <li key={n} className="flex gap-3 rounded-xl border border-border bg-surface px-3 py-2.5">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent-soft text-xs font-semibold text-accent">{n}</span>
          <span className="text-xs leading-snug"><b className="block text-sm">{t}</b><span className="text-muted">{d}</span></span>
        </li>
      ))}
    </ol>
  );
}

export function ObservePanel() {
  const running = useStudio((s) => isRunning(s.status));
  const busy = useStudio((s) => s.inspectBusy);
  return (
    <div className="flex h-full min-h-0 flex-col bg-bg">
      <div className="flex shrink-0 items-center gap-2 border-b border-border bg-surface px-4 py-2.5">
        <Activity size={16} className="text-accent" />
        <h2 className="text-sm font-semibold">Observe</h2>
        <span className="hidden text-xs text-muted md:inline">how the transformer behaves</span>
        <div className="flex-1" />
        <MiniStatus />
        {running ? <Badge tone="ok"><span className="live-dot h-1.5 w-1.5 rounded-full bg-ok" />live</Badge> : busy ? <Badge>updating…</Badge> : null}
      </div>
      <div className="@container min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        <ErrorBanner />
        <Steps />
        <TokenTable />
        <TrainingCard />
        <NetworkFlow />
        <GradientFlow />
        <EmbeddingMap />
        <GenerateBox />
        <div className="h-6" />
      </div>
    </div>
  );
}
