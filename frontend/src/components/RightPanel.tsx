"use client";
import { Activity, BookOpen, Code2 } from "lucide-react";
import { useStudio } from "@/lib/store";
import { LiveCodeTab } from "./build/CodePanel";
import { InfoPanel } from "./info/InfoPanel";
import { ObservePanel } from "./observe/ObservePanel";
import { cn } from "./ui/kit";

type Tab = "observe" | "code" | "learn";

/** The right-hand column: behaviour, live-generated code, and the concepts/formulas behind each block — as tabs. */
export function RightPanel() {
  const tab = useStudio((s) => s.rightTab);
  const setTab = useStudio((s) => s.setRightTab);
  const nBlocks = useStudio((s) => s.blocks.length);
  const tabs: { id: Tab; label: string; icon: typeof Code2 }[] = [
    { id: "learn", label: "Learn", icon: BookOpen },
    { id: "code", label: "Live code", icon: Code2 },
    { id: "observe", label: "Observe", icon: Activity },
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
      {/* all three stay mounted so scroll position, the code panel's language, and which block Learn is showing survive switching */}
      <div className={cn("min-h-0 flex-1 flex-col", tab === "observe" ? "flex" : "hidden")}>
        <ObservePanel />
      </div>
      <div className={cn("min-h-0 flex-1 flex-col", tab === "learn" ? "flex" : "hidden")}>
        <InfoPanel />
      </div>
      <div className={cn("min-h-0 flex-1 flex-col", tab === "code" ? "flex" : "hidden")}>
        <LiveCodeTab />
      </div>
    </div>
  );
}
