"use client";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { AlertTriangle, ChevronDown, Code2, Copy, GripVertical, Info, Trash2, XCircle } from "lucide-react";
import { blockLabel, CATALOG, fmtParams } from "@/lib/blocks";
import { useStudio } from "@/lib/store";
import type { Block, Issue, ShapeInfo } from "@/lib/types";
import { Badge, cn } from "../ui/kit";
import { BlockEditor } from "./BlockEditor";

export function BlockCard({
  block, index, locked, issues, shape, dropBefore,
}: { block: Block; index: number; locked: boolean; issues: Issue[]; shape?: ShapeInfo; dropBefore: boolean }) {
  const blocks = useStudio((s) => s.blocks);
  const selected = useStudio((s) => s.selectedId === block.id);
  const hovered = useStudio((s) => s.hoverId === block.id);
  const select = useStudio((s) => s.select);
  const hover = useStudio((s) => s.hover);
  const remove = useStudio((s) => s.removeBlock);
  const duplicate = useStudio((s) => s.duplicateBlock);
  const openLearn = useStudio((s) => s.openLearn);
  const openBlockCode = useStudio((s) => s.openBlockCode);

  const def = CATALOG[block.type];
  const fixed = block.type === "embedding" || block.type === "head";
  const { attributes, listeners, setNodeRef, setActivatorNodeRef, transform, transition, isDragging } = useSortable({
    id: `b:${block.id}`,
    disabled: { draggable: fixed || locked },
  });
  const Icon = def.icon;
  const worst = issues.some((i) => i.level === "error") ? "error" : issues.some((i) => i.level === "warning") ? "warning" : issues.length ? "info" : null;
  const hasResidual = (block.type === "attention" || block.type === "ffn") && block.residual;

  return (
    <div
      ref={setNodeRef}
      style={{ transform: CSS.Translate.toString(transform), transition, opacity: isDragging ? 0.4 : 1 }}
      className="relative fade-in"
      onMouseEnter={() => hover(block.id)}
      onMouseLeave={() => hover(null)}
    >
      {dropBefore && <div className="absolute -top-2 left-0 right-0 z-10 h-1 rounded-full bg-accent" />}
      {hasResidual && (
        <svg
          className="pointer-events-none absolute -left-3.5 top-0 h-full w-3.5 overflow-visible text-accent/50"
          aria-hidden
          preserveAspectRatio="none"
          viewBox="0 0 14 100"
        >
          <path d="M 14 4 C 2 4, 2 96, 14 96" fill="none" stroke="currentColor" strokeWidth="1.5" markerEnd="url(#res-arrow)" vectorEffect="non-scaling-stroke" />
          <defs>
            <marker id="res-arrow" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 z" fill="currentColor" />
            </marker>
          </defs>
        </svg>
      )}
      <div
        className={cn(
          "overflow-hidden rounded-xl border bg-surface shadow-card transition-colors",
          selected ? "border-accent ring-1 ring-accent/40" : hovered ? "border-border-strong" : "border-border",
          worst === "error" && !selected && "border-danger/60",
        )}
        title={hasResidual ? "Residual (skip) connection: this block's output is added back to its own input — the curved arrow at left" : undefined}
      >
        <div className="flex">
          <div className="w-1 shrink-0" style={{ background: def.color }} />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5 px-2 py-2">
              <button
                ref={setActivatorNodeRef}
                {...attributes}
                {...listeners}
                aria-label={`Drag ${def.title}`}
                disabled={fixed || locked}
                className={cn("touch-none rounded p-1 text-muted", fixed || locked ? "opacity-25" : "hover:bg-surface-2 hover:text-fg")}
                title={fixed ? "This block has a fixed position" : "Drag to reorder"}
              >
                <GripVertical size={16} />
              </button>
              <button className="flex min-w-0 flex-1 items-center gap-2.5 text-left" onClick={() => select(selected ? null : block.id)} aria-expanded={selected}>
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg" style={{ background: `color-mix(in srgb, ${def.color} 16%, transparent)`, color: def.color }}>
                  <Icon size={17} />
                </span>
                <span className="min-w-0">
                  <span className="flex items-center gap-1.5 text-sm font-semibold leading-tight">
                    <span className="truncate">{blockLabel(blocks, index)}</span>
                    {worst === "error" && <XCircle size={14} className="shrink-0 text-danger" />}
                    {worst === "warning" && <AlertTriangle size={14} className="shrink-0 text-warn" />}
                    {worst === "info" && <Info size={14} className="shrink-0 text-accent" />}
                  </span>
                  <span className="block truncate text-xs text-muted">{def.summary(block)}</span>
                  <span className="mt-1 flex flex-wrap items-center gap-1 @lg:hidden">
                    {shape && <Badge title="Output shape for one sequence: [tokens × width]">[{shape.outShape[0]}×{shape.outShape[1]}]</Badge>}
                {shape && shape.params > 0 && <Badge tone="accent" title="Trainable parameters">{fmtParams(shape.params)}</Badge>}
                  </span>
                </span>
              </button>
              <div className="hidden items-center gap-1 @lg:flex">
                {shape && <Badge title="Output shape for one sequence: [tokens × width]">[{shape.outShape[0]}×{shape.outShape[1]}]</Badge>}
                {shape && shape.params > 0 && <Badge tone="accent" title="Trainable parameters">{fmtParams(shape.params)}</Badge>}
              </div>
              <button
                aria-label={`Learn about ${def.title}`}
                title="Concepts, formulas and a worked example for this block"
                onClick={() => openLearn(block.id)}
                className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-border-strong bg-surface-2 text-[9px] font-bold leading-none text-muted hover:border-accent hover:bg-accent-soft hover:text-accent"
              >
                i
              </button>
              <button
                aria-label={`Try ${def.title} in code`}
                title="Edit and run a tiny, live snippet for this block — right in your browser"
                onClick={() => openBlockCode(block.id)}
                className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-border-strong bg-surface-2 text-muted hover:border-accent hover:bg-accent-soft hover:text-accent"
              >
                <Code2 size={9} />
              </button>
              {!fixed && (
                <button aria-label="Duplicate block" disabled={locked} onClick={() => duplicate(block.id)} className="rounded p-1.5 text-muted hover:bg-surface-2 hover:text-fg disabled:opacity-30">
                  <Copy size={14} />
                </button>
              )}
              <button aria-label="Remove block" disabled={locked} onClick={() => remove(block.id)} className="rounded p-1.5 text-muted hover:bg-surface-2 hover:text-danger disabled:opacity-30">
                <Trash2 size={14} />
              </button>
              <button aria-label={selected ? "Collapse" : "Edit parameters"} onClick={() => select(selected ? null : block.id)} className="rounded p-1.5 text-muted hover:bg-surface-2">
                <ChevronDown size={15} className={cn("transition-transform", selected && "rotate-180")} />
              </button>
            </div>

            {issues.length > 0 && (
              <ul className="space-y-1 px-3 pb-2">
                {issues.map((i, k) => (
                  <li key={k} className={cn("text-xs leading-snug", i.level === "error" ? "text-danger" : i.level === "warning" ? "text-warn" : "text-muted")}>
                    {i.message}
                  </li>
                ))}
              </ul>
            )}

            {selected && (
              <div className="border-t border-border bg-surface-2/40 px-3 py-3">
                <BlockEditor block={block} disabled={locked} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
