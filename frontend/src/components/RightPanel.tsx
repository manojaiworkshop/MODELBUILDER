"use client";
import { Activity, Code2 } from "lucide-react";
import { useState } from "react";
import { useStudio } from "@/lib/store";
import { CodePanel } from "./build/CodePanel";
import { ObservePanel } from "./observe/ObservePanel";
import { cn } from "./ui/kit";

type Tab = "code" | "observe";

/** The right-hand column: live-generated code and the training/behaviour view, as two tabs. */
export function RightPanel() {
  const [tab, setTab] = useState<Tab>("observe");
  const nBlocks = useStudio((s) => s.blocks.length);
  const tabs: { id: Tab; label: string; icon: typeof Code2 }[] = [
    { id: "observe", label: "Observe", icon: Activity },
    { id: "code", label: "Live code", icon: Code2 },
  ];
  return (
    <div className="flex h-full min-h-0 flex-col bg-bg">
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
            {t.id === "code" && <span className="rounded-full bg-surface-2 px-1.5 text-[11px] tabular-nums text-muted">{nBlocks}</span>}
          </button>
        ))}
      </div>
      {/* both stay mounted so scroll position and the code panel's language choice survive switching */}
      <div className={cn("min-h-0 flex-1 flex-col", tab === "observe" ? "flex" : "hidden")}>
        <ObservePanel />
      </div>
      <div className={cn("min-h-0 flex-1 flex-col", tab === "code" ? "flex" : "hidden")}>
        <CodePanel />
      </div>
    </div>
  );
}
