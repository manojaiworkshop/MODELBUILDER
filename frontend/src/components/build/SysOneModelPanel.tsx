"use client";
import { Brain } from "lucide-react";
import { isRunning, useStudio } from "@/lib/store";
import type { SysOneHyper } from "@/lib/types";
import { Card, Field, NumberInput, SectionHeader } from "../ui/kit";

type Def = { key: Exclude<keyof SysOneHyper, "device" | "base" | "layaLayers" | "layaMaxLen" | "layaQuestions">; label: string; help: string; min: number; max: number; step?: number };

const ARCH: Def[] = [
  { key: "dModel", label: "d_model", help: "Width of every token's vector. Must be a multiple of heads.", min: 8, max: 2048, step: 8 },
  { key: "nHeads", label: "Heads", help: "Query attention heads.", min: 1, max: 64 },
  { key: "nKvHeads", label: "KV heads", help: "Key/value heads (grouped-query attention). Must divide heads evenly.", min: 1, max: 64 },
  { key: "nLayers", label: "Layers", help: "Stacked encoder blocks.", min: 1, max: 48 },
  { key: "dFf", label: "Feed-forward width", help: "Hidden width of each block's MLP.", min: 8, max: 8192, step: 8 },
  { key: "dropout", label: "Dropout", help: "Regularization inside attention and the MLP.", min: 0, max: 0.9, step: 0.05 },
  { key: "maxSeqLen", label: "Max sequence length", help: "Longest packed [state + all questions] the model can see.", min: 16, max: 4096, step: 16 },
  { key: "merges", label: "BPE merges", help: "Tokenizer vocabulary size, learned from your generated samples.", min: 0, max: 5000, step: 10 },
];

export function SysOneModelPanel() {
  const hyper = useStudio((s) => s.sysoneHyper);
  const setHyper = useStudio((s) => s.setSysoneHyper);
  const locked = useStudio((s) => isRunning(s.sysoneStatus));
  const nSamples = useStudio((s) => s.sysoneSamples.length);

  return (
    <div className="space-y-4 overflow-y-auto p-4">
      {hyper.base === "laya" && (
        <p className="rounded-lg border border-accent/40 bg-accent/5 px-3 py-2.5 text-xs">
          <strong>Starting from Laya.</strong> The architecture below is ignored: training fine-tunes the pretrained Laya encoder (ModernBERT-large, 28 layers, 1024 wide) and its decision head. Laya reads one question per sequence, with each option scored at its own [MASK] token. Choose how much to fine-tune in the Training tab.
        </p>
      )}
      <Card>
        <SectionHeader
          title="System One — JEV model"
          sub="A bidirectional transformer encoder that reads a state plus typed questions (yes/no, choice, score) and answers all of them with calibrated probabilities in one forward pass — no text generation. A fixed architecture, not a stack of drag-and-drop blocks."
          icon={<Brain size={16} className="text-accent" />}
        />
        <div className="space-y-2 px-4 pb-4 text-xs text-muted">
          <p>
            <strong className="text-fg">How it reads input:</strong> the state and each question are packed into one token sequence with segment
            markers. A structured attention mask lets state tokens see only the state, and each question see the state plus itself — never other
            questions — so answers from a multi-question sample never leak into each other.
          </p>
          <p>
            <strong className="text-fg">How it answers:</strong> three small heads read the hidden state at each question&apos;s marker token — one for
            yes/no probability, one (shared) for choice/score option logits, one for a learned confidence estimate — trained with proper scoring
            rules so the probabilities are calibrated, not just accurate.
          </p>
          <p>
            Generate training data in the <strong className="text-fg">Data</strong> tab, then press <strong className="text-fg">Run</strong> to
            train. Its vocabulary is learned from that data, independent of the Data tab&apos;s own text/BPE settings.
            {nSamples === 0 && " No samples yet."}
          </p>
        </div>
      </Card>
      <Card>
        <SectionHeader title="Architecture" sub="Changing these after training starts fresh weights next run." />
        <div className="grid grid-cols-2 gap-x-3 gap-y-3 px-4 pb-4 @sm:grid-cols-3 sm:grid-cols-3">
          {ARCH.map((d) => (
            <Field key={d.key} label={d.label} help={d.help}>
              <NumberInput value={hyper[d.key]} onChange={(v) => setHyper({ [d.key]: v })} min={d.min} max={d.max} step={d.step} float={d.step !== undefined && d.step < 1} disabled={locked} />
            </Field>
          ))}
        </div>
      </Card>
    </div>
  );
}
