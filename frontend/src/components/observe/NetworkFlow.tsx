"use client";
import { ArrowDown } from "lucide-react";
import { blockLabel, CATALOG, fmtParams } from "@/lib/blocks";
import { useStudio } from "@/lib/store";
import type { Block, LayerTrace, NextToken, TokenInfo } from "@/lib/types";
import { Badge, Card, SectionHeader, cn } from "../ui/kit";
import { Heatmap } from "../ui/Heatmap";
import { visible } from "./TokenTable";

const rowsH = (t: number) => Math.max(3, Math.min(10, Math.round(110 / Math.max(1, t))));

function AttentionMaps({ trace, tokens }: { trace: LayerTrace; tokens: TokenInfo[] }) {
  const a = trace.attn!;
  const labels = tokens.map((t) => visible(t.text));
  return (
    <div>
      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(112px, 1fr))" }}>
        {a.heads.map((h, i) => (
          <div key={i}>
            <div className="mb-1 flex items-center justify-between text-[11px] text-muted">
              <span className="font-medium">Head {i + 1}</span>
              {a.sink_mass && <span title="Probability mass parked on the attention sink (attends to nothing)">sink {(a.sink_mass[i] * 100).toFixed(0)}%</span>}
            </div>
            <Heatmap data={h.data} rows={h.rows} cols={h.cols} mode="sequential" max={1} square rowLabels={labels} colLabels={labels} />
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-muted">Rows = the token asking, columns = the tokens it looks at. Bright = high attention. Hover a cell for the pair.</p>
    </div>
  );
}

function MoeView({ trace, tokens }: { trace: LayerTrace; tokens: TokenInfo[] }) {
  const m = trace.moe!;
  const labels = tokens.map((t) => visible(t.text));
  const max = Math.max(...m.usage, 1e-6);
  return (
    <div className="grid gap-4 @lg:grid-cols-[1fr_1fr] lg:grid-cols-[1fr_1fr]">
      <div>
        <div className="mb-1 text-[11px] font-medium text-muted">Router: which experts each token uses</div>
        <Heatmap data={m.gates.data} rows={m.gates.rows} cols={m.gates.cols} mode="sequential" max={1} cellH={Math.max(8, rowsH(tokens.length) + 4)} rowLabels={labels} colLabels={m.usage.map((_, i) => `expert ${i}`)} />
        <div className="mt-1 flex justify-between text-[10px] text-muted">
          {m.usage.map((_, i) => <span key={i}>E{i}</span>)}
        </div>
      </div>
      <div>
        <div className="mb-1 text-[11px] font-medium text-muted">Expert load (mean routing weight)</div>
        <div className="space-y-1">
          {m.usage.map((u, i) => (
            <div key={i} className="flex items-center gap-2 text-[11px]">
              <span className="w-5 text-muted">E{i}</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-amber-500" style={{ width: `${(u / max) * 100}%` }} /></div>
              <span className="w-9 text-right tabular-nums text-muted">{(u * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[11px] text-muted">Balanced bars mean every expert is being used; one dominant bar means routing has collapsed onto a single expert.</p>
      </div>
    </div>
  );
}

function HeadView({ next, loss }: { next: NextToken[]; loss: number | null }) {
  return (
    <div>
      <div className="mb-1.5 text-[11px] font-medium text-muted">Most likely next token after your sentence (among tokens in your data)</div>
      <div className="space-y-1">
        {next.map((t, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className="w-16 truncate rounded bg-surface-2 px-1.5 py-0.5 font-mono">{visible(t.text)}</span>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-rose-500" style={{ width: `${Math.max(1, t.p * 100)}%` }} /></div>
            <span className="w-11 text-right tabular-nums text-muted">{(t.p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      {loss !== null && <p className="mt-2 text-[11px] text-muted">Loss on this sentence: <b className="text-fg">{loss.toFixed(3)}</b> (perplexity {Math.exp(loss).toFixed(1)}). Lower means it predicts the sentence better.</p>}
    </div>
  );
}

function BlockViz({ block, index, trace, tokens, maxGrad, next, loss }: { block: Block; index: number; trace?: LayerTrace; tokens: TokenInfo[]; maxGrad: number; next: NextToken[]; loss: number | null }) {
  const blocks = useStudio((s) => s.blocks);
  const hoverId = useStudio((s) => s.hoverId);
  const selectedId = useStudio((s) => s.selectedId);
  const hover = useStudio((s) => s.hover);
  const select = useStudio((s) => s.select);
  const def = CATALOG[block.type];
  const Icon = def.icon;
  const focus = hoverId === block.id || selectedId === block.id;

  return (
    <div id={`viz-${block.id}`} onMouseEnter={() => hover(block.id)} onMouseLeave={() => hover(null)} className={cn("overflow-hidden rounded-xl border bg-surface transition-colors", focus ? "border-accent ring-1 ring-accent/40" : "border-border")}>
      <div className="flex">
        <div className="w-1 shrink-0" style={{ background: def.color }} />
        <div className="min-w-0 flex-1 p-3">
          <button className="flex w-full flex-wrap items-center gap-2 text-left" onClick={() => select(block.id)}>
            <span className="flex h-7 w-7 items-center justify-center rounded-lg" style={{ background: `color-mix(in srgb, ${def.color} 16%, transparent)`, color: def.color }}><Icon size={15} /></span>
            <span className="text-sm font-semibold">{blockLabel(blocks, index)}</span>
            <span className="text-xs text-muted">{def.summary(block)}</span>
            <span className="flex-1" />
            {trace && <Badge title="Output shape">[{trace.outShape[0]}×{trace.outShape[1]}]</Badge>}
            {trace && trace.params > 0 && <Badge tone="accent">{fmtParams(trace.params)} params</Badge>}
          </button>

          {trace && trace.gradNorm !== null && (
            <div className="mt-2 flex items-center gap-2 text-[11px] text-muted" title="L2 norm of this block's gradients at the latest snapshot">
              <span className="w-16">gradient</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-emerald-500" style={{ width: `${Math.max(2, (trace.gradNorm / maxGrad) * 100)}%` }} /></div>
              <span className="w-14 text-right tabular-nums">{trace.gradNorm.toFixed(4)}</span>
            </div>
          )}

          <div className="mt-3">
            {!trace ? (
              <div className="h-10 animate-pulse rounded-lg bg-surface-2" />
            ) : trace.kind === "attention" && trace.attn ? (
              <AttentionMaps trace={trace} tokens={tokens} />
            ) : trace.kind === "ffn" && trace.moe ? (
              <MoeView trace={trace} tokens={tokens} />
            ) : trace.kind === "head" ? (
              <HeadView next={next} loss={loss} />
            ) : trace.act.rows > 0 ? (
              <div>
                <div className="mb-1 text-[11px] font-medium text-muted">Activations after this block (tokens × dims)</div>
                <Heatmap data={trace.act.data} rows={trace.act.rows} cols={trace.act.cols} cellH={rowsH(trace.act.rows)} rowLabels={tokens.map((t) => visible(t.text))} />
              </div>
            ) : null}
            {trace && trace.kind !== "head" && (
              <div className="mt-2 flex flex-wrap gap-x-4 text-[11px] text-muted tabular-nums">
                <span>mean {trace.stats.mean.toFixed(2)}</span>
                <span>std {trace.stats.std.toFixed(2)}</span>
                <span>max |x| {trace.stats.absMax.toFixed(2)}</span>
              </div>
            )}
            {trace && trace.kind === "attention" && trace.act.rows > 0 && (
              <details className="mt-2">
                <summary className="cursor-pointer text-[11px] font-medium text-muted hover:text-fg">Residual stream after this block</summary>
                <div className="mt-1"><Heatmap data={trace.act.data} rows={trace.act.rows} cols={trace.act.cols} cellH={rowsH(trace.act.rows)} rowLabels={tokens.map((t) => visible(t.text))} /></div>
              </details>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function NetworkFlow() {
  const blocks = useStudio((s) => s.blocks);
  const inspect = useStudio((s) => s.inspect);
  const conn = useStudio((s) => s.conn);
  const synced = !!inspect && inspect.shapes.length === blocks.length;
  const traces = new Map<number, LayerTrace>();
  if (inspect?.ok && synced) inspect.layers.forEach((l) => traces.set(l.index, l));
  const maxGrad = Math.max(1e-9, ...(inspect?.layers ?? []).map((l) => l.gradNorm ?? 0));
  const errors = (inspect?.issues ?? []).filter((i) => i.level === "error");

  return (
    <Card>
      <SectionHeader title="Inside the network" sub="Every block of your stack, with what it computes for the sentence above. Hover a block here or on the left to link them." />
      <div className="space-y-0 px-4 pb-4">
        {conn !== "connected" && (
          <div className="mb-3 rounded-lg border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-warn">
            Backend offline — start it with <code className="font-mono">cargo run --release</code> in <code className="font-mono">backend/</code>.
          </div>
        )}
        {inspect && !inspect.ok && (
          <div className="mb-3 rounded-lg border border-danger/40 bg-danger/5 px-3 py-2 text-xs text-danger">
            <div className="mb-1 font-medium">This model can’t run yet:</div>
            <ul className="list-disc pl-4">{errors.length ? errors.map((e, i) => <li key={i}>{e.message}</li>) : <li>{inspect.error}</li>}</ul>
          </div>
        )}
        {blocks.length === 0 && <div className="rounded-lg bg-surface-2 px-3 py-8 text-center text-sm text-muted">Add blocks on the left and they appear here.</div>}
        {blocks.map((b, i) => (
          <div key={b.id}>
            <BlockViz block={b} index={i} trace={traces.get(i)} tokens={inspect?.tokens ?? []} maxGrad={maxGrad} next={inspect?.nextTokens ?? []} loss={inspect?.probeLoss ?? null} />
            {i < blocks.length - 1 && (
              <div className="flex h-7 items-center justify-center gap-2 text-[11px] text-muted" aria-hidden>
                <ArrowDown size={14} />
                {inspect && inspect.shapes.length === blocks.length && <span className="tabular-nums">[{inspect.shapes[i].outShape[0]} × {inspect.shapes[i].outShape[1]}]</span>}
              </div>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
}
