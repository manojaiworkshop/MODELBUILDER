"use client";
import { useState } from "react";
import { useStudio } from "@/lib/store";
import { Badge, Card, SectionHeader } from "../ui/kit";
import { visible } from "./TokenTable";

const S = 320;
const PAD = 20;

function category(t: string): { name: string; color: string } {
  if (t.length > 1 && /[A-Za-z]/.test(t)) return { name: "multi-char token", color: "#14b8a6" };
  if (/^\s$/.test(t)) return { name: "whitespace", color: "#94a3b8" };
  if (/^[0-9]$/.test(t)) return { name: "digit", color: "#0ea5e9" };
  if (/^[aeiouAEIOU]$/.test(t)) return { name: "vowel", color: "#f43f5e" };
  if (/^[a-z]$/.test(t)) return { name: "consonant", color: "#6366f1" };
  if (/^[A-Z]$/.test(t)) return { name: "capital", color: "#8b5cf6" };
  return { name: "punctuation / other", color: "#f59e0b" };
}
const LEGEND = ["vowel", "consonant", "capital", "digit", "whitespace", "punctuation / other", "multi-char token"].map((n) => ({ name: n, color: category(n === "vowel" ? "a" : n === "consonant" ? "b" : n === "capital" ? "A" : n === "digit" ? "1" : n === "whitespace" ? " " : n === "multi-char token" ? "ab" : ".").color }));

/** PCA of the token embedding matrix. Dots glide between snapshots as training reshapes the space. */
export function EmbeddingMap() {
  const map = useStudio((s) => s.inspect?.map) ?? [];
  const trained = useStudio((s) => s.inspect?.trained);
  const [hover, setHover] = useState<number | null>(null);
  const maxCount = Math.max(1, ...map.map((m) => m.count));
  const labelled = new Set([...map].sort((a, b) => b.count - a.count).slice(0, 22).map((m) => m.id));
  const px = (v: number) => PAD + ((v + 1) / 2) * (S - 2 * PAD);

  return (
    <Card>
      <SectionHeader
        title="Embedding space"
        sub="Every token’s vector squashed to 2-D (PCA). Watch related tokens drift together during training."
        right={<Badge tone={trained ? "ok" : "neutral"}>{trained ? "trained" : "random init"}</Badge>}
      />
      <div className="px-4 pb-4">
        {map.length < 3 ? (
          <div className="rounded-lg bg-surface-2 px-3 py-10 text-center text-xs text-muted">Not enough tokens to plot yet.</div>
        ) : (
          <>
            <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto block w-full max-w-[440px] rounded-lg bg-surface-2/60" role="img" aria-label="Token embedding scatter plot">
              <line x1={PAD} x2={S - PAD} y1={S / 2} y2={S / 2} stroke="var(--border)" />
              <line y1={PAD} y2={S - PAD} x1={S / 2} x2={S / 2} stroke="var(--border)" />
              {map.map((m) => {
                const c = category(m.text);
                const r = 3 + 4 * Math.sqrt(m.count / maxCount);
                const active = hover === m.id;
                return (
                  <g key={m.id} style={{ transform: `translate(${px(m.x)}px, ${px(-m.y)}px)`, transition: "transform 0.6s ease" }} onMouseEnter={() => setHover(m.id)} onMouseLeave={() => setHover(null)}>
                    <circle r={active ? r + 2 : r} fill={c.color} fillOpacity={active ? 1 : 0.75} stroke="var(--surface)" strokeWidth="1" />
                    {(labelled.has(m.id) || active) && (
                      <text y={-r - 3} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--fg)" style={{ paintOrder: "stroke", stroke: "var(--surface)", strokeWidth: 2.5 }}>{visible(m.text)}</text>
                    )}
                  </g>
                );
              })}
            </svg>
            <div className="mt-2 flex flex-wrap justify-center gap-x-3 gap-y-1 text-[11px] text-muted">
              {LEGEND.map((l) => <span key={l.name} className="flex items-center gap-1"><i className="inline-block h-2 w-2 rounded-full" style={{ background: l.color }} />{l.name}</span>)}
            </div>
          </>
        )}
      </div>
    </Card>
  );
}
