"use client";
import { Cpu, Download, Loader2, Sparkles, X } from "lucide-react";
import { useEffect } from "react";
import { cancelLayaDownload, checkLaya, downloadLaya } from "@/lib/sync";
import { LAYA_PRESET, SCRATCH_PRESET, isRunning, useStudio } from "@/lib/store";
import type { SysOneHyper } from "@/lib/types";
import { Button, Card, Field, NumberInput, SectionHeader, SelectInput, cn } from "../ui/kit";

type NumKey = Exclude<keyof SysOneHyper, "device" | "base" | "layaLayers" | "layaMaxLen" | "layaQuestions">;
type Def = { key: NumKey; label: string; help: string; min: number; max: number; step?: number; float?: boolean };

const OPT: Def[] = [
  { key: "epochs", label: "Epochs", help: "Full passes over the training samples.", min: 1, max: 200 },
  { key: "batchSize", label: "Batch size", help: "Samples per optimizer step.", min: 1, max: 256 },
  { key: "lr", label: "Learning rate", help: "Peak Adam learning rate (cosine decay to 0).", min: 0.00001, max: 1, step: 0.0001, float: true },
  { key: "warmupSteps", label: "Warmup steps", help: "Linear LR ramp-up at the start.", min: 0, max: 5000, step: 10 },
  { key: "confWeight", label: "Confidence weight", help: "How much the learned-confidence loss counts toward the total.", min: 0, max: 5, step: 0.1, float: true },
  { key: "gradClip", label: "Grad clip", help: "Clip global gradient norm.", min: 0.01, max: 100, step: 0.5, float: true },
  { key: "seed", label: "Seed", help: "Same seed → same initial weights and batches.", min: 0, max: 999999 },
];

type LayaDef = { key: "layaLayers" | "layaMaxLen" | "layaQuestions"; label: string; help: string; min: number; max: number; step?: number };
const LAYA_OPT: LayaDef[] = [
  { key: "layaLayers", label: "Layers to fine-tune", help: "Of Laya's 28 encoder layers, how many (from the top) are updated; the rest stay as pretrained. More layers adapt better but need much more GPU memory: 4 ≈ 13 GB, all 28 ≈ 24+ GB.", min: 0, max: 28 },
  { key: "layaMaxLen", label: "Tokens per question", help: "Length of each question sequence (its options plus the state, which is cut to fit). Laya reads one question at a time.", min: 64, max: 2048, step: 64 },
  { key: "layaQuestions", label: "Questions per sample", help: "Each epoch draws this many questions from every sample (each one is its own sequence). Fewer = faster epochs.", min: 1, max: 64 },
];

const gb = (n: number) => (n / 1e9).toFixed(2);

function LayaBase() {
  const hyper = useStudio((s) => s.sysoneHyper);
  const setHyper = useStudio((s) => s.setSysoneHyper);
  const laya = useStudio((s) => s.layaStatus);
  const conn = useStudio((s) => s.conn);
  const locked = useStudio((s) => isRunning(s.sysoneStatus));
  useEffect(() => {
    if (conn === "connected") checkLaya();
  }, [conn]);
  const base = hyper.base ?? "scratch";
  const pick = (b: "scratch" | "laya") => {
    if (b !== base) setHyper(b === "laya" ? LAYA_PRESET : SCRATCH_PRESET);
  };
  const pct = laya.total ? Math.min(100, (laya.done / laya.total) * 100) : 0;

  const choice = (b: "scratch" | "laya", title: string, desc: string, icon: React.ReactNode) => (
    <button
      type="button"
      disabled={locked}
      onClick={() => pick(b)}
      className={cn("flex flex-1 flex-col items-start gap-1 rounded-lg border px-3 py-2.5 text-left transition-colors disabled:opacity-60", base === b ? "border-accent bg-accent/5" : "border-border hover:bg-surface-2")}
    >
      <span className="flex items-center gap-1.5 text-sm font-medium">{icon}{title}</span>
      <span className="text-xs text-muted">{desc}</span>
    </button>
  );

  return (
    <Card>
      <SectionHeader title="Start from" sub="Train a small model from random weights, or fine-tune Laya — a pretrained 421M-parameter decision model — on your samples." />
      <div className="space-y-3 px-4 pb-4">
        <div className="flex flex-col gap-2 sm:flex-row">
          {choice("scratch", "From scratch", "A small model (≈ 1M parameters) learns everything from your samples. Runs on CPU or GPU.", <Cpu size={14} className="text-accent" />)}
          {choice("laya", "Laya (pretrained)", "Starts from convaiinnovations/laya (Apache-2.0) and adapts it to your data. Already understands language; needs the GPU.", <Sparkles size={14} className="text-accent" />)}
        </div>

        {base === "laya" && (
          <div className="space-y-3 rounded-lg bg-surface-2/60 p-3">
            {laya.state === "ready" ? (
              <p className="text-xs text-muted">Laya is on this machine ({gb(laya.total)} GB). Pick how much of it to fine-tune below, then press Run in the Observe tab.</p>
            ) : laya.state === "downloading" ? (
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 text-xs">
                  <Loader2 size={13} className="animate-spin text-accent" />
                  <span>Downloading Laya… {laya.total ? `${gb(laya.done)} / ${gb(laya.total)} GB` : ""}</span>
                  <span className="flex-1" />
                  <Button size="sm" variant="ghost" onClick={() => cancelLayaDownload()}><X size={13} /> Cancel</Button>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${Math.max(2, pct)}%` }} /></div>
              </div>
            ) : (
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className={cn("flex-1", laya.state === "error" ? "text-danger" : "text-muted")}>
                  {laya.state === "error" ? laya.message : "Laya isn't on this machine yet — about 0.84 GB from huggingface.co/convaiinnovations/laya."}
                </span>
                <Button size="sm" variant="primary" onClick={() => downloadLaya()}><Download size={13} /> {laya.state === "error" ? "Retry" : "Import Laya"}</Button>
              </div>
            )}
            <div className="grid grid-cols-2 gap-x-3 gap-y-3 sm:grid-cols-3">
              {LAYA_OPT.map((d) => (
                <Field key={d.key} label={d.label} help={d.help}>
                  <NumberInput value={hyper[d.key]} onChange={(v) => setHyper({ [d.key]: v })} min={d.min} max={d.max} step={d.step} disabled={locked} />
                </Field>
              ))}
            </div>
            {hyper.layaLayers > 8 && <p className="text-xs text-warn">Fine-tuning more than ~8 layers may not fit in GPU memory — lower the batch size if the run fails.</p>}
            {hyper.device !== "gpu" && <p className="text-xs text-warn">Training Laya on the CPU is impractically slow — set “Train on” below to GPU.</p>}
          </div>
        )}
      </div>
    </Card>
  );
}

export function SysOneTrainPanel() {
  const hyper = useStudio((s) => s.sysoneHyper);
  const setHyper = useStudio((s) => s.setSysoneHyper);
  const locked = useStudio((s) => isRunning(s.sysoneStatus));
  const nSamples = useStudio((s) => s.sysoneSamples.length);
  // the backend holds out the last ~10% (at least 1) for validation
  const nTrain = Math.max(0, nSamples - Math.max(1, Math.floor(nSamples / 10)));
  const perEpoch = Math.max(1, Math.ceil(nTrain / Math.max(1, hyper.batchSize)));
  const steps = perEpoch * hyper.epochs;

  return (
    <div className="space-y-4 p-4">
      <LayaBase />
      <Card>
        <SectionHeader title="Optimization" sub="Plain Adam with linear warmup then cosine decay. The last ~10% of samples are held out for per-epoch validation; the best-calibrated epoch (not just the most accurate) is kept." />
        <div className="grid grid-cols-2 gap-x-3 gap-y-3 px-4 pb-4 @sm:grid-cols-3 sm:grid-cols-3">
          {OPT.map((d) => (
            <Field key={d.key} label={d.label} help={d.help}>
              <NumberInput value={hyper[d.key]} onChange={(v) => setHyper({ [d.key]: v })} min={d.min} max={d.max} step={d.step} float={d.float} disabled={locked} />
            </Field>
          ))}
          <Field label="Train on" help="CPU uses every core. GPU (Vulkan) is usually much faster for bigger models or batches; if it can't be opened the run falls back to the CPU. The trained model is always served from the CPU.">
            <SelectInput value={hyper.device ?? "cpu"} onChange={(v) => setHyper({ device: v as SysOneHyper["device"] })} disabled={locked} options={[{ value: "cpu", label: "CPU (all cores)" }, { value: "gpu", label: "GPU (Vulkan)" }]} />
          </Field>
        </div>
        <p className="border-t border-border px-4 py-2.5 text-xs text-muted">
          {nSamples.toLocaleString()} sample{nSamples === 1 ? "" : "s"} generated in the Data tab
          {nSamples === 0 && " — generate some before pressing Run."}
          {nSamples > 0 && <> · ≈ {nTrain.toLocaleString()} for training → {perEpoch} step{perEpoch === 1 ? "" : "s"} per epoch × {hyper.epochs} epochs = <b className="font-medium text-fg">{steps.toLocaleString()} optimizer steps</b></>}
        </p>
        {nSamples > 0 && steps < 300 && (
          <p className="border-t border-warn/40 bg-warn/5 px-4 py-2.5 text-xs text-warn">
            That is very few updates — the model won’t learn much (loss stays near where it started). Aim for a few hundred or more: raise Epochs (e.g. {Math.ceil(400 / perEpoch)}), or lower Batch size (16–32 suits a small dataset).
          </p>
        )}
      </Card>
    </div>
  );
}
