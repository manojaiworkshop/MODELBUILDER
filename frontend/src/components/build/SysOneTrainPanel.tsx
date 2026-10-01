"use client";
import { isRunning, useStudio } from "@/lib/store";
import type { SysOneHyper } from "@/lib/types";
import { Card, Field, NumberInput, SectionHeader, SelectInput } from "../ui/kit";

type Def = { key: Exclude<keyof SysOneHyper, "device">; label: string; help: string; min: number; max: number; step?: number; float?: boolean };

const OPT: Def[] = [
  { key: "epochs", label: "Epochs", help: "Full passes over the training samples.", min: 1, max: 200 },
  { key: "batchSize", label: "Batch size", help: "Samples per optimizer step.", min: 1, max: 256 },
  { key: "lr", label: "Learning rate", help: "Peak Adam learning rate (cosine decay to 0).", min: 0.00001, max: 1, step: 0.0001, float: true },
  { key: "warmupSteps", label: "Warmup steps", help: "Linear LR ramp-up at the start.", min: 0, max: 5000, step: 10 },
  { key: "confWeight", label: "Confidence weight", help: "How much the learned-confidence loss counts toward the total.", min: 0, max: 5, step: 0.1, float: true },
  { key: "gradClip", label: "Grad clip", help: "Clip global gradient norm.", min: 0.01, max: 100, step: 0.5, float: true },
  { key: "seed", label: "Seed", help: "Same seed → same initial weights and batches.", min: 0, max: 999999 },
];

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
