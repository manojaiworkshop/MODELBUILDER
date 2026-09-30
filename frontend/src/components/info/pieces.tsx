"use client";
import { Lightbulb, Sigma, TestTubeDiagonal } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "../ui/kit";

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="mb-1.5 text-sm font-semibold text-fg">{title}</h3>
      <div className="space-y-2 text-[13px] leading-relaxed text-fg/90">{children}</div>
    </section>
  );
}

/** A math/pseudocode formula, set in monospace on its own line(s). */
export function Formula({ children, label }: { children: ReactNode; label?: string }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-code px-3 py-2.5 font-mono text-[12.5px] leading-relaxed text-slate-100">
      {label && <div className="mb-1 text-[10px] font-sans font-semibold uppercase tracking-wide text-slate-400">{label}</div>}
      <div className="whitespace-pre">{children}</div>
    </div>
  );
}

/** A worked numeric example, using the block's actual configured values where possible. */
export function Example({ children, title = "Worked example" }: { children: ReactNode; title?: string }) {
  return (
    <div className="rounded-lg border border-accent/25 bg-accent-soft/60 px-3 py-2.5">
      <div className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-accent"><TestTubeDiagonal size={12} /> {title}</div>
      <div className="space-y-1.5 text-[13px] leading-relaxed">{children}</div>
    </div>
  );
}

export function Why({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-warn/25 bg-warn/[0.06] px-3 py-2.5">
      <div className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-warn"><Lightbulb size={12} /> Why it’s used</div>
      <div className="space-y-1.5 text-[13px] leading-relaxed">{children}</div>
    </div>
  );
}

/** A small conditional aside, used for "this only applies because your block has X enabled" notes. */
export function Note({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "on" | "off" }) {
  return (
    <div
      className={cn(
        "rounded-md border px-2.5 py-2 text-[12.5px] leading-relaxed",
        tone === "on" && "border-ok/30 bg-ok/[0.06]",
        tone === "off" && "border-border bg-surface-2/60 text-muted",
        tone === "neutral" && "border-border bg-surface-2/60",
      )}
    >
      {children}
    </div>
  );
}

/** Inline monospace fragment inside a paragraph, e.g. for `d_model` or a short expression. */
export function M({ children }: { children: ReactNode }) {
  return <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12px] text-fg">{children}</code>;
}

export function Formulaic() {
  return <Sigma size={13} className="text-muted" aria-hidden />;
}
