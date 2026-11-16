"use client";
import { useStudio } from "@/lib/store";
import type { Hyper } from "@/lib/types";
import { Card, Field, NumberInput, SectionHeader, SelectInput } from "../ui/kit";

type Def = { key: Exclude<keyof Hyper, "device">; label: string; help: string; min: number; max: number; step?: number; float?: boolean };

const OPT: Def[] = [
  { key: "steps", label: "Steps", help: "Number of optimizer updates.", min: 1, max: 20000, step: 50 },
  { key: "batchSize", label: "Batch size", help: "Sequences per step.", min: 1, max: 64 },
  { key: "seqLen", label: "Context length", help: "Tokens the model sees at once (max 256).", min: 4, max: 256, step: 8 },
  { key: "lr", label: "Learning rate", help: "Peak AdamW learning rate (cosine decay to 10%).", min: 0.00001, max: 1, step: 0.0005, float: true },
  { key: "warmupSteps", label: "Warmup steps", help: "Linear LR ramp-up at the start.", min: 0, max: 5000, step: 10 },
  { key: "weightDecay", label: "Weight decay", help: "AdamW decoupled decay.", min: 0, max: 1, step: 0.01, float: true },
  { key: "gradClip", label: "Grad clip", help: "Clip global gradient norm (0 = off).", min: 0, max: 100, step: 0.5, float: true },
  { key: "seed", label: "Seed", help: "Same seed → same initial weights and batches.", min: 0, max: 999999 },
];
const LIVE: Def[] = [
  { key: "probeEvery", label: "Snapshot every", help: "Steps between full network snapshots (attention, activations, gradients).", min: 1, max: 1000 },
  { key: "evalEvery", label: "Validate every", help: "Steps between validation-loss measurements.", min: 1, max: 5000 },
  { key: "sampleEvery", label: "Sample every", help: "Steps between text samples.", min: 1, max: 5000 },
  { key: "sampleTokens", label: "Sample length", help: "Tokens per sample (0 = none).", min: 0, max: 500, step: 10 },
  { key: "temperature", label: "Temperature", help: "0 = greedy; higher = more random.", min: 0, max: 5, step: 0.1, float: true },
];

function Grid({ defs, disabled, children }: { defs: Def[]; disabled: boolean; children?: React.ReactNode }) {
  const hyper = useStudio((s) => s.hyper);
  const setHyper = useStudio((s) => s.setHyper);
  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-3 px-4 pb-4 @sm:grid-cols-3 sm:grid-cols-3">
      {defs.map((d) => (
        <Field key={d.key} label={d.label} help={d.help}>
          <NumberInput value={hyper[d.key]} onChange={(v) => setHyper({ [d.key]: v })} min={d.min} max={d.max} step={d.step} float={d.float} disabled={disabled} />
        </Field>
      ))}
      {children}
    </div>
  );
}

export function TrainPanel({ locked }: { locked: boolean }) {
  const hyper = useStudio((s) => s.hyper);
  const setHyper = useStudio((s) => s.setHyper);
  const tokens = useStudio((s) => s.dataInfo?.tokens);
  const seen = hyper.steps * hyper.batchSize * hyper.seqLen;
  return (
    <div className="space-y-4 p-4">
      <Card>
        <SectionHeader title="Optimization" sub="AdamW with linear warmup then cosine decay." />
        <Grid defs={OPT} disabled={locked}>
          <Field label="Train on" help="CPU uses every core. GPU (Vulkan) is much faster for bigger models, batches and contexts; if it can't be opened the run falls back to the CPU. Snapshots and samples are always computed on the CPU.">
            <SelectInput value={hyper.device ?? "cpu"} onChange={(v) => setHyper({ device: v as Hyper["device"] })} disabled={locked} options={[{ value: "cpu", label: "CPU (all cores)" }, { value: "gpu", label: "GPU (Vulkan)" }]} />
          </Field>
        </Grid>
        <p className="border-t border-border px-4 py-2.5 text-xs text-muted">
          This run processes ≈ {seen.toLocaleString()} tokens
          {tokens ? <> — about {(seen / tokens).toFixed(1)}× your dataset.</> : "."}
          {tokens !== undefined && tokens < hyper.seqLen + 2 && <span className="text-danger"> Data has too few tokens for this context length.</span>}
        </p>
      </Card>
      <Card>
        <SectionHeader title="Live feedback" sub="How often the right-hand panel refreshes. Frequent snapshots cost some speed." />
        <Grid defs={LIVE} disabled={false} />
      </Card>
    </div>
  );
}
