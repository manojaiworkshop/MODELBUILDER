"use client";
import { Database, Layers, SlidersHorizontal } from "lucide-react";
import { useState } from "react";
import { isRunning, useStudio } from "@/lib/store";
import { Canvas } from "./Canvas";
import { DataPanel } from "./DataPanel";
import { TrainPanel } from "./TrainPanel";
import { cn } from "../ui/kit";

type Tab = "blocks" | "data" | "train";

export function BuildPanel() {
  const [tab, setTab] = useState<Tab>("blocks");
  const nBlocks = useStudio((s) => s.blocks.length);
  const locked = useStudio((s) => isRunning(s.status));
  const tabs: { id: Tab; label: string; icon: typeof Layers; count?: number }[] = [
    { id: "blocks", label: "Model", icon: Layers, count: nBlocks },
    { id: "data", label: "Data", icon: Database },
    { id: "train", label: "Training", icon: SlidersHorizontal },
  ];
  return (
    <div className="@container flex h-full min-h-0 flex-col bg-bg">
      <div role="tablist" className="flex shrink-0 gap-1 border-b border-border bg-surface px-3 pt-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "flex items-center gap-1.5 rounded-t-lg border-b-2 px-3 py-2 text-sm font-medium transition-colors",
              tab === t.id ? "border-accent text-fg" : "border-transparent text-muted hover:text-fg",
            )}
          >
            <t.icon size={15} />
            {t.label}
            {t.count !== undefined && <span className="rounded-full bg-surface-2 px-1.5 text-[11px] tabular-nums text-muted">{t.count}</span>}
          </button>
        ))}
      </div>
      {/* every tab stays mounted so scroll position / drafts survive switching */}
      <div className={cn("min-h-0 flex-1 flex-col", tab === "blocks" ? "flex" : "hidden")}>
        <Canvas />
      </div>
      <div className={cn("min-h-0 flex-1 overflow-y-auto", tab === "data" ? "block" : "hidden")}>
        <DataPanel />
      </div>
      <div className={cn("min-h-0 flex-1 overflow-y-auto", tab === "train" ? "block" : "hidden")}>
        <TrainPanel locked={locked} />
      </div>
    </div>
  );
}
