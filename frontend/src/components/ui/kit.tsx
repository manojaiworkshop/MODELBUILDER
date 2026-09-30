"use client";
import { ChevronDown } from "lucide-react";
import { useEffect, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";

export const cn = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

// ------------------------------------------------------------------ theme
export function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState("light");
  useEffect(() => {
    const read = () => setTheme(document.documentElement.dataset.theme ?? "light");
    read();
    const mo = new MutationObserver(read);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => mo.disconnect();
  }, []);
  const toggle = () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("mb.theme", next);
    } catch {}
  };
  return [theme, toggle];
}

// ---------------------------------------------------------------- buttons
type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "danger" | "soft"; size?: "sm" | "md" };
export function Button({ variant = "soft", size = "md", className, ...p }: BtnProps) {
  const base = "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors disabled:opacity-45 select-none whitespace-nowrap";
  const sizes = { sm: "h-7 px-2.5 text-xs", md: "h-9 px-3.5 text-sm" }[size];
  const variants = {
    primary: "bg-accent text-accent-fg hover:brightness-110 shadow-sm",
    danger: "bg-danger text-white hover:brightness-110 shadow-sm",
    soft: "bg-surface-2 text-fg hover:bg-border border border-border",
    ghost: "text-muted hover:text-fg hover:bg-surface-2",
  }[variant];
  return <button className={cn(base, sizes, variants, className)} {...p} />;
}

export function Badge({ children, tone = "neutral", className, title }: { children: ReactNode; tone?: "neutral" | "ok" | "warn" | "danger" | "accent"; className?: string; title?: string }) {
  const tones = {
    neutral: "bg-surface-2 text-muted border-border",
    ok: "bg-[color-mix(in_srgb,var(--ok)_14%,transparent)] text-ok border-[color-mix(in_srgb,var(--ok)_30%,transparent)]",
    warn: "bg-[color-mix(in_srgb,var(--warn)_14%,transparent)] text-warn border-[color-mix(in_srgb,var(--warn)_30%,transparent)]",
    danger: "bg-[color-mix(in_srgb,var(--danger)_14%,transparent)] text-danger border-[color-mix(in_srgb,var(--danger)_30%,transparent)]",
    accent: "bg-accent-soft text-accent border-[color-mix(in_srgb,var(--accent)_30%,transparent)]",
  }[tone];
  return (
    <span title={title} className={cn("inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-none tabular-nums", tones, className)}>
      {children}
    </span>
  );
}

// ------------------------------------------------------------------ inputs
export function Toggle({ checked, onChange, disabled, label }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn("relative h-5 w-9 shrink-0 rounded-full transition-colors disabled:opacity-50", checked ? "bg-accent" : "bg-border-strong")}
    >
      <span className={cn("absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all", checked ? "left-[18px]" : "left-0.5")} />
    </button>
  );
}

export function NumberInput({
  value, onChange, min, max, step = 1, disabled, className, float,
}: { value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; disabled?: boolean; className?: string; float?: boolean }) {
  const [draft, setDraft] = useState<string | null>(null);
  const clamp = (n: number) => Math.min(max ?? Infinity, Math.max(min ?? -Infinity, n));
  const commit = (s: string) => {
    const n = float ? parseFloat(s) : parseInt(s, 10);
    if (Number.isFinite(n)) onChange(clamp(n));
  };
  return (
    <input
      type="text"
      inputMode="decimal"
      disabled={disabled}
      value={draft ?? String(value)}
      onFocus={(e) => e.currentTarget.select()}
      onChange={(e) => {
        setDraft(e.target.value);
        const n = float ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
        if (Number.isFinite(n) && n >= (min ?? -Infinity) && n <= (max ?? Infinity)) onChange(n);
      }}
      onBlur={() => {
        if (draft !== null) commit(draft);
        setDraft(null);
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        if (e.key === "ArrowUp" || e.key === "ArrowDown") {
          e.preventDefault();
          const d = (e.key === "ArrowUp" ? 1 : -1) * step;
          const next = clamp(+(value + d).toFixed(6));
          setDraft(null);
          onChange(next);
        }
      }}
      className={cn("h-8 w-full rounded-md border border-border bg-surface px-2 text-sm tabular-nums outline-none focus:border-accent disabled:opacity-50", className)}
    />
  );
}

export function SelectInput({ value, onChange, options, disabled }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; disabled?: boolean }) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="h-8 w-full rounded-md border border-border bg-surface px-2 text-sm outline-none focus:border-accent disabled:opacity-50"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}

export function Segmented<T extends string>({ value, onChange, options, className }: { value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; title?: string }[]; className?: string }) {
  return (
    <div role="tablist" className={cn("inline-flex rounded-lg border border-border bg-surface-2 p-0.5", className)}>
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={cn("rounded-md px-2.5 py-1 text-xs font-medium transition-colors", value === o.value ? "bg-surface text-fg shadow-sm" : "text-muted hover:text-fg")}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, help, children, inline }: { label: string; help?: string; children: ReactNode; inline?: boolean }) {
  return (
    <label className={cn("block", inline && "flex items-center justify-between gap-3")}>
      <span className="mb-1 flex items-center gap-1 text-xs font-medium text-muted" title={help}>
        {label}
        {help && <span aria-hidden className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-border-strong text-[9px]">?</span>}
      </span>
      {children}
    </label>
  );
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={cn("rounded-xl border border-border bg-surface shadow-card", className)}>{children}</section>;
}

export function SectionHeader({ title, sub, right, icon, collapsed, onToggle }: { title: string; sub?: string; right?: ReactNode; icon?: ReactNode; collapsed?: boolean; onToggle?: () => void }) {
  const text = (
    <>
      <h3 className="flex items-center gap-2 text-sm font-semibold text-fg">{icon}{title}</h3>
      {sub && !collapsed && <p className="mt-0.5 text-xs text-muted">{sub}</p>}
    </>
  );
  return (
    <div className={cn("flex items-start justify-between gap-3 px-4 pt-3.5", collapsed ? "pb-3.5" : "pb-2")}>
      {onToggle ? (
        <button type="button" onClick={onToggle} aria-expanded={!collapsed} className="flex min-w-0 flex-1 items-start gap-2 text-left">
          <ChevronDown size={16} className={cn("mt-0.5 shrink-0 text-muted transition-transform", collapsed && "-rotate-90")} />
          <div className="min-w-0">{text}</div>
        </button>
      ) : (
        <div className="min-w-0">{text}</div>
      )}
      {right}
    </div>
  );
}

// ------------------------------------------------------------------- modal
export function Modal({ open, onClose, title, sub, icon, children, width = "max-w-lg" }: { open: boolean; onClose: () => void; title: string; sub?: string; icon?: ReactNode; children: ReactNode; width?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    ref.current?.focus();
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="presentation">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[1px]" onClick={onClose} aria-hidden />
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        className={cn("fade-in relative z-10 max-h-[85vh] w-full overflow-y-auto rounded-xl border border-border bg-surface shadow-card outline-none", width)}
      >
        <div className="sticky top-0 z-10 flex items-start justify-between gap-3 border-b border-border bg-surface px-4 py-3.5">
          <div className="min-w-0">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-fg">{icon}{title}</h3>
            {sub && <p className="mt-0.5 text-xs text-muted">{sub}</p>}
          </div>
          <button onClick={onClose} aria-label="Close" className="shrink-0 rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-fg">
            <XIcon />
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

function XIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}
