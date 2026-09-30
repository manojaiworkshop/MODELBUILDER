"use client";
import { GraduationCap } from "lucide-react";
import { isRunning, useStudio } from "@/lib/store";
import { Badge, Segmented, cn } from "../ui/kit";

/** What the next Run learns from, and whether it starts over or builds on the last run. */
export function TrainSourceBar() {
  const source = useStudio((s) => s.trainSource);
  const mode = useStudio((s) => s.trainMode);
  const setSource = useStudio((s) => s.setTrainSource);
  const setMode = useStudio((s) => s.setTrainMode);
  const nQa = useStudio((s) => s.qa.length);
  const qaRunning = useStudio((s) => s.qaProgress?.state === "running");
  const locked = useStudio((s) => isRunning(s.status));
  const hasWeights = useStudio((s) => s.inspect?.trained === true);
  const canContinue = source === "qa" && hasWeights;

  // nothing to choose between until there are pairs
  if (nQa === 0 && source === "text") return null;

  const hint =
    source === "text"
      ? "Training on the text and files from the Data tab."
      : nQa === 0
        ? "No pairs yet — generate some in the Data tab."
        : mode === "continue" && canContinue
          ? `Keeps the trained weights and the vocabulary they were trained with, and learns from ${nQa.toLocaleString()} pairs.`
          : `Starts from random weights and learns a vocabulary from ${nQa.toLocaleString()} pairs.`;

  return (
    <section aria-label="Training data for the next run" className="rounded-xl border border-border bg-surface px-4 py-3 shadow-card">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="flex items-center gap-2 text-sm font-semibold"><GraduationCap size={16} className="text-accent" /> Next run</span>
        <label className="flex items-center gap-2 text-xs text-muted">
          Learn from
          <Segmented
            value={source}
            onChange={(v) => { if (!locked) { setSource(v); if (v === "text") setMode("new"); } }}
            options={[
              { value: "text", label: "Text / files", title: "The data in the Data tab" },
              { value: "qa", label: <>Q&A pairs {nQa > 0 && <Badge className="ml-1">{nQa}</Badge>}</>, title: "The question-answer pairs generated from your documents" },
            ]}
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-muted">
          Start
          <Segmented
            value={mode === "continue" && !canContinue ? "new" : mode}
            onChange={(v) => { if (!locked && (v === "new" || canContinue)) setMode(v); }}
            options={[
              { value: "new", label: "New model", title: "Fresh random weights" },
              {
                value: "continue",
                label: "Continue trained",
                title: canContinue ? "Keep training the model from the last run" : source === "text" ? "Available for Q&A pairs, after a model has been trained" : "Train a model first",
              },
            ]}
            className={cn(!canContinue && "[&_[aria-selected=false]:last-child]:opacity-45")}
          />
        </label>
      </div>
      <p className="mt-2 text-xs text-muted">
        {hint}
        {qaRunning && source === "qa" && " (pairs are still being generated — the run uses what exists when you press Run)"}
      </p>
    </section>
  );
}
