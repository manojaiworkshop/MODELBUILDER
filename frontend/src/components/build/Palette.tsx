"use client";
import { useDraggable } from "@dnd-kit/core";
import { Boxes } from "lucide-react";
import { CATALOG, PALETTE_ORDER } from "@/lib/blocks";
import { useStudio } from "@/lib/store";
import type { BlockType } from "@/lib/types";
import { cn } from "../ui/kit";

export type PaletteKind = BlockType | "transformer";

/** One square tool, MS-Paint-toolbox style: icon only, name on hover, click or drag to place. */
function Tool({ kind, disabled, why }: { kind: PaletteKind; disabled?: boolean; why?: string }) {
  const addBlock = useStudio((s) => s.addBlock);
  const addTransformerBlock = useStudio((s) => s.addTransformerBlock);
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({ id: `palette:${kind}`, data: { kind }, disabled });
  const def = kind === "transformer" ? null : CATALOG[kind];
  const Icon = def ? def.icon : Boxes;
  const color = def ? def.color : "#f43f5e";
  const title = def ? def.title : "Transformer Block";
  const blurb = def
    ? def.blurb
    : "One full pre-norm transformer block — Self-Attention then Feed-Forward, each already bundling its own Norm and residual add (see their pre_norm/residual settings) — the unit real models stack × N times.";

  return (
    <button
      ref={setNodeRef}
      {...attributes}
      {...listeners}
      onClick={() => (kind === "transformer" ? addTransformerBlock() : addBlock(kind))}
      disabled={disabled}
      aria-label={`Add ${title}`}
      title={disabled ? why : `${title} — ${blurb}\n\nClick to add, or drag onto the stack.`}
      className={cn(
        "group relative flex h-11 w-11 shrink-0 touch-none flex-col items-center justify-center gap-0.5 rounded-lg border bg-surface shadow-sm transition",
        disabled ? "cursor-not-allowed opacity-40" : "cursor-grab border-border hover:border-border-strong hover:shadow active:cursor-grabbing",
        !disabled && "border-border",
        isDragging && "opacity-40",
      )}
    >
      <span className="flex h-6 w-6 items-center justify-center rounded-md" style={{ background: `color-mix(in srgb, ${color} 18%, transparent)`, color }}>
        <Icon size={kind === "transformer" ? 15 : 14} />
      </span>
      {kind === "transformer" && <span className="absolute -right-1 -top-1 rounded-full bg-danger px-1 text-[8px] font-bold leading-tight text-white">×N</span>}
    </button>
  );
}

/** The block toolbox — a grid of icon tools (MS Paint style) above the stack: click or drag a
 * tool to place it. Primitive blocks sit on the left; the composite Transformer Block on the right. */
/** MS-Paint-style toolbox: one vertical column of icon tools, meant to sit in a narrow sidebar
 * that's vertically centered on screen rather than docked to the top. */
export function Palette({ locked }: { locked: boolean }) {
  const blocks = useStudio((s) => s.blocks);
  const hasEmbed = blocks.some((b) => b.type === "embedding");
  const hasHead = blocks.some((b) => b.type === "head");
  return (
    <div className="flex flex-col items-center gap-1.5">
      {PALETTE_ORDER.map((t) => {
        const used = (t === "embedding" && hasEmbed) || (t === "head" && hasHead);
        return <Tool key={t} kind={t} disabled={locked || used} why={locked ? "Locked while training" : "Already in the stack (only one allowed)"} />;
      })}
      <div className="h-px w-8 shrink-0 bg-border" aria-hidden />
      <Tool kind="transformer" disabled={locked} why="Locked while training" />
    </div>
  );
}
