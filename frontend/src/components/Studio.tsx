"use client";
import { Eye, Hammer } from "lucide-react";
import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";
import { useBackendSync } from "@/lib/sync";
import { useStudio } from "@/lib/store";
import { BuildPanel } from "./build/BuildPanel";
import { LlmSettingsHost } from "./info/LlmSettings";
import { RightPanel } from "./RightPanel";
import { TopBar } from "./TopBar";
import { cn } from "./ui/kit";

function Layout() {
  useBackendSync();
  const splitPct = useStudio((s) => s.splitPct);
  const setSplit = useStudio((s) => s.setSplit);
  const tab = useStudio((s) => s.mobilePane);
  const setTab = useStudio((s) => s.setMobilePane);
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const onMove = useCallback((e: PointerEvent) => {
    if (!dragging.current || !box.current) return;
    const r = box.current.getBoundingClientRect();
    setSplit(((e.clientX - r.left) / r.width) * 100);
  }, [setSplit]);
  useEffect(() => {
    const up = () => { dragging.current = false; document.body.style.cursor = ""; document.body.style.userSelect = ""; };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", up);
    return () => { window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", up); };
  }, [onMove]);

  return (
    <div className="flex h-dvh flex-col">
      <TopBar />
      <LlmSettingsHost />
      <div ref={box} className="flex min-h-0 flex-1 flex-col lg:flex-row" style={{ ["--split" as string]: `${splitPct}%` }}>
        <section aria-label="Build" className={cn("min-h-0 min-w-0 flex-1 lg:w-(--split) lg:flex-none", tab === "build" ? "flex flex-col" : "hidden lg:flex lg:flex-col")}>
          <BuildPanel />
        </section>
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize panels"
          aria-valuenow={Math.round(splitPct)}
          tabIndex={0}
          onPointerDown={(e) => { e.preventDefault(); dragging.current = true; document.body.style.cursor = "col-resize"; document.body.style.userSelect = "none"; }}
          onKeyDown={(e) => { if (e.key === "ArrowLeft") setSplit(splitPct - 2); if (e.key === "ArrowRight") setSplit(splitPct + 2); }}
          className="group hidden w-1.5 shrink-0 cursor-col-resize items-center justify-center bg-border transition-colors hover:bg-accent/50 lg:flex"
        >
          <span className="h-10 w-0.5 rounded-full bg-border-strong group-hover:bg-accent" />
        </div>
        <section aria-label="Live code and observe" className={cn("min-h-0 min-w-0 flex-1", tab === "right" ? "flex flex-col" : "hidden lg:flex lg:flex-col")}>
          <RightPanel />
        </section>
      </div>
      <nav className="grid shrink-0 grid-cols-2 border-t border-border bg-surface lg:hidden" aria-label="Switch panel">
        {([["build", "Build", Hammer], ["right", "Observe", Eye]] as const).map(([id, label, Icon]) => (
          <button key={id} onClick={() => setTab(id)} aria-current={tab === id} className={cn("flex items-center justify-center gap-2 py-3 text-sm font-medium", tab === id ? "text-accent" : "text-muted")}>
            <Icon size={16} /> {label}
          </button>
        ))}
      </nav>
    </div>
  );
}

const noopSubscribe = () => () => {};

export function Studio() {
  // false during SSR and the hydration render, true afterwards: the saved layout/model
  // (read from localStorage when the store is created) never mismatches server markup.
  const ready = useSyncExternalStore(noopSubscribe, () => true, () => false);
  if (!ready) return <div className="flex h-dvh items-center justify-center text-sm text-muted">Loading studio…</div>;
  return <Layout />;
}
