"use client";
import { ArrowRight } from "lucide-react";
import { useStudio } from "@/lib/store";
import { Badge, Card, SectionHeader } from "../ui/kit";
import { Heatmap } from "../ui/Heatmap";

export const TINTS = ["#6366f1", "#0ea5e9", "#14b8a6", "#f59e0b", "#f43f5e", "#8b5cf6"];
export const visible = (t: string) => t.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥");
const ROW = 20;

/** sentence → tokens → ids → embedding vectors: the first thing the model does. */
export function TokenTable() {
  const probe = useStudio((s) => s.probe);
  const setProbe = useStudio((s) => s.setProbe);
  const inspect = useStudio((s) => s.inspect);
  const embed = inspect?.layers.find((l) => l.kind === "embedding");
  const tokens = inspect?.tokens ?? [];
  const shown = inspect?.ok && embed && embed.act.rows === tokens.length;

  return (
    <Card>
      <SectionHeader
        title="Tokens → IDs → Embeddings"
        sub="Type any sentence: watch it become tokens, ids, then the vectors the network actually sees."
        right={inspect && <Badge tone={inspect.trained ? "ok" : "neutral"}>{inspect.trained ? `trained · step ${inspect.step}` : "random init"}</Badge>}
      />
      <div className="space-y-3 px-4 pb-4">
        <input
          value={probe}
          onChange={(e) => setProbe(e.target.value)}
          spellCheck={false}
          aria-label="Sentence to trace through the model"
          placeholder="Type a sentence to trace…"
          className="h-9 w-full rounded-lg border border-border bg-surface px-3 font-mono text-sm outline-none focus:border-accent"
        />
        {tokens.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted">
            <span>{probe.length} chars</span> <ArrowRight size={12} /> <span className="font-medium text-fg">{tokens.length} tokens</span> <ArrowRight size={12} />
            <span>{inspect?.layers[0]?.act.cols ?? "?"}-dim vectors</span>
            {inspect && tokens.length >= 32 && <Badge tone="warn">probe capped at 32 tokens</Badge>}
          </div>
        )}
        {shown ? (
          <>
          <div className="max-h-[280px] overflow-auto rounded-lg">
            <div className="grid min-w-[300px] grid-cols-[minmax(80px,auto)_52px_1fr] items-center gap-x-3">
              <div className="pb-1 text-[11px] font-medium text-muted">token</div>
              <div className="pb-1 text-[11px] font-medium text-muted">id</div>
              <div className="pb-1 text-[11px] font-medium text-muted">embedding vector (evenly sampled dims)</div>
              <div>
                {tokens.map((t, i) => (
                  <div key={i} className="flex items-center" style={{ height: ROW }}>
                    <span
                      className="max-w-40 truncate rounded px-1.5 py-0.5 font-mono text-xs"
                      style={{ background: `color-mix(in srgb, ${TINTS[i % TINTS.length]} 18%, transparent)`, color: `color-mix(in srgb, ${TINTS[i % TINTS.length]} 70%, var(--fg))` }}
                    >
                      {visible(t.text)}
                    </span>
                  </div>
                ))}
              </div>
              <div>
                {tokens.map((t, i) => (
                  <div key={i} className="flex items-center font-mono text-xs tabular-nums text-muted" style={{ height: ROW }}>{t.id}</div>
                ))}
              </div>
              <Heatmap data={embed.act.data} rows={embed.act.rows} cols={embed.act.cols} cellH={ROW} mode="diverging" rowLabels={tokens.map((t) => visible(t.text))} unit="" />
            </div>
          </div>
          <p className="text-[11px] text-muted">
            Red = positive, blue = negative. Embeddings start random; training organizes them so tokens used alike get similar vectors.
          </p>
          </>
        ) : (
          <div className="rounded-lg bg-surface-2 px-3 py-6 text-center text-xs text-muted">
            {inspect && !inspect.ok ? "Fix the model on the left to see embeddings." : "Waiting for the backend…"}
          </div>
        )}
      </div>
    </Card>
  );
}
