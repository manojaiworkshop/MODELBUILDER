"use client";
import { blockLabel, CATALOG } from "@/lib/blocks";
import { useStudio } from "@/lib/store";
import { Card, SectionHeader } from "../ui/kit";

/** Gradient / weight norms per block at the latest snapshot — a quick health check of what is learning. */
export function GradientFlow() {
  const blocks = useStudio((s) => s.blocks);
  const snap = useStudio((s) => s.gradSnap);
  // only meaningful while the stack still matches what was trained
  const rows = snap && snap.shapes.length === blocks.length ? snap.layers.filter((l) => l.gradNorm !== null && blocks[l.index]) : [];
  const max = Math.max(1e-9, ...rows.map((r) => r.gradNorm ?? 0));
  if (rows.length === 0) return null;
  return (
    <Card>
      <SectionHeader title="Gradient flow" sub={`How strongly each block was being updated at step ${snap?.step ?? "?"}. Vanishing bars = that block is barely learning.`} />
      <div className="space-y-2 px-4 pb-4">
        {rows.map((r) => {
          const b = blocks[r.index];
          return (
            <div key={r.index} className="flex items-center gap-2 text-xs">
              <span className="w-32 truncate text-muted">{blockLabel(blocks, r.index)}</span>
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${Math.max(1.5, ((r.gradNorm ?? 0) / max) * 100)}%`, background: CATALOG[b.type].color }} />
              </div>
              <span className="w-16 text-right tabular-nums">{(r.gradNorm ?? 0).toFixed(4)}</span>
              <span className="hidden w-20 text-right tabular-nums text-muted sm:block" title="weight L2 norm">|w| {(r.weightNorm ?? 0).toFixed(2)}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
