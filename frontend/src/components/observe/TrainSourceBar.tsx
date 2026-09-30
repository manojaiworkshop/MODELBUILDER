"use client";
import { CheckCircle2, Download, GraduationCap, Loader2 } from "lucide-react";
import { useState } from "react";
import { downloadModel } from "@/lib/sync";
import { isRunning, useStudio } from "@/lib/store";
import { Badge, Segmented, cn } from "../ui/kit";

const BUNDLE_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "README.md"];

/** Once a run has finished: the trained model as a Hugging Face-style folder. */
function ModelDownload() {
  const status = useStudio((s) => s.status);
  const withChat = useStudio((s) => s.trainSource !== "text" && (s.trainSource === "qa" ? s.qa : s.summaries).some((p) => p.style === "chat"));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const done = status.state === "done";

  async function get() {
    setBusy(true);
    setError("");
    const err = await downloadModel();
    setBusy(false);
    if (err) setError(err);
  }

  return (
    <div className="mt-3 rounded-lg border border-border bg-surface-2/50 px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <CheckCircle2 size={15} className={done ? "text-ok" : "text-muted"} />
        <span className="text-sm font-medium">{done ? "Training finished" : `Stopped at step ${status.step}`}</span>
        <span className="flex-1" />
        <button
          onClick={get}
          disabled={busy}
          className="inline-flex h-8 items-center gap-1.5 rounded-md bg-accent px-3 text-xs font-medium text-white hover:opacity-90 disabled:opacity-60"
          title="Download the trained model as a Hugging Face-style folder (zip)"
        >
          {busy ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />} Download model
        </button>
      </div>
      <p className="mt-1.5 text-xs text-muted">
        Hugging Face format (.zip): {BUNDLE_FILES.join(" · ")}
        {withChat && " · chat_template.jinja"}
      </p>
      {error && <p role="alert" className="mt-1 text-xs text-danger">{error}</p>}
    </div>
  );
}

/** What the next Run learns from, and whether it starts over or builds on the last run. */
export function TrainSourceBar() {
  const source = useStudio((s) => s.trainSource);
  const mode = useStudio((s) => s.trainMode);
  const setSource = useStudio((s) => s.setTrainSource);
  const setMode = useStudio((s) => s.setTrainMode);
  const nQa = useStudio((s) => s.qa.length);
  const qaRunning = useStudio((s) => s.qaProgress?.state === "running");
  const nSummary = useStudio((s) => s.summaries.length);
  const summarizeRunning = useStudio((s) => s.summaryProgress?.state === "running");
  const locked = useStudio((s) => isRunning(s.status));
  const hasWeights = useStudio((s) => s.inspect?.trained === true);
  const canContinue = (source === "qa" || source === "summary") && hasWeights;

  const status = useStudio((s) => s.status);
  const canDownload = (status.state === "done" || status.state === "stopped") && hasWeights && status.step > 0;
  const choices = !(nQa === 0 && nSummary === 0 && source === "text");

  // nothing to choose between until there are pairs or summaries, and nothing to download until a run finished
  if (!choices && !canDownload) return null;

  const count = source === "qa" ? nQa : nSummary;
  const noun = source === "qa" ? "pairs" : "summaries";
  const hint =
    source === "text"
      ? "Training on the text and files from the Data tab."
      : count === 0
        ? `No ${noun} yet — generate some in the Data tab.`
        : mode === "continue" && canContinue
          ? `Keeps the trained weights and the vocabulary they were trained with, and learns from ${count.toLocaleString()} ${noun}.`
          : `Starts from random weights and learns a vocabulary from ${count.toLocaleString()} ${noun}.`;

  return (
    <section aria-label="Training data for the next run" className="rounded-xl border border-border bg-surface px-4 py-3 shadow-card">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="flex items-center gap-2 text-sm font-semibold"><GraduationCap size={16} className="text-accent" /> Next run</span>
        {choices && (<>
        <label className="flex items-center gap-2 text-xs text-muted">
          Learn from
          <Segmented
            value={source}
            onChange={(v) => { if (!locked) { setSource(v); if (v === "text") setMode("new"); } }}
            options={[
              { value: "text", label: "Text / files", title: "The data in the Data tab" },
              { value: "qa", label: <>Q&A pairs {nQa > 0 && <Badge className="ml-1">{nQa}</Badge>}</>, title: "The question-answer pairs generated from your documents" },
              { value: "summary", label: <>Summaries {nSummary > 0 && <Badge className="ml-1">{nSummary}</Badge>}</>, title: "The page summaries generated from your documents" },
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
                title: canContinue ? "Keep training the model from the last run" : source === "text" ? "Available for Q&A pairs or summaries, after a model has been trained" : "Train a model first",
              },
            ]}
            className={cn(!canContinue && "[&_[aria-selected=false]:last-child]:opacity-45")}
          />
        </label>
        </>)}
      </div>
      {choices && <p className="mt-2 text-xs text-muted">
        {hint}
        {qaRunning && source === "qa" && " (pairs are still being generated — the run uses what exists when you press Run)"}
        {summarizeRunning && source === "summary" && " (summaries are still being generated — the run uses what exists when you press Run)"}
      </p>}
      {canDownload && <ModelDownload />}
    </section>
  );
}
