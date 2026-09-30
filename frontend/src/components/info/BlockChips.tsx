"use client";
import { blockLabel, CATALOG } from "@/lib/blocks";
import { useStudio } from "@/lib/store";
import { cn } from "../ui/kit";

/** Horizontal "jump to a block" navigator, shared by the Learn and Playground panels. Each keeps
 * its own notion of the active block (`activeId`/`onSelect`) so picking one in one panel doesn't
 * move the other. */
export function BlockChips({ activeId, onSelect }: { activeId: string | null; onSelect: (id: string) => void }) {
  const blocks = useStudio((s) => s.blocks);
  return (
    <div className="-mx-1 flex gap-1.5 overflow-x-auto px-1 pb-1">
      {blocks.map((b, i) => {
        const def = CATALOG[b.type];
        const Icon = def.icon;
        const active = b.id === activeId;
        return (
          <button
            key={b.id}
            onClick={() => onSelect(b.id)}
            aria-current={active}
            className={cn(
              "flex shrink-0 items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors",
              active ? "border-accent bg-accent-soft text-accent" : "border-border bg-surface text-muted hover:text-fg",
            )}
          >
            <span className="flex h-4 w-4 items-center justify-center" style={{ color: def.color }}><Icon size={13} /></span>
            {blockLabel(blocks, i)}
          </button>
        );
      })}
    </div>
  );
}
