"use client";
import { AlertTriangle, Brain, Download, Loader2, Settings2, Sparkles, Square, Trash2, Wand2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { clearSysOne, generateSysOne, loadQaPages, removeSysOne, stopSysOneGenerate } from "@/lib/sync";
import { fmtCount } from "@/lib/format";
import { useStudio } from "@/lib/store";
import type { QaPageList, SysOneQuestion, SysOneTarget } from "@/lib/types";
import { DatasetIO } from "./DatasetIO";
import { SysOneHfImportModal } from "./SysOneHfImportModal";
import { SysOneSynthModal } from "./SysOneSynthModal";
import { Badge, Button, Card, Field, NumberInput, SectionHeader, SelectInput, cn } from "../ui/kit";

/** "1-5, 9, 12-14" -> the page numbers in it that exist. */
function parseRanges(text: string, available: Set<number>): number[] {
  const out = new Set<number>();
  for (const part of text.split(/[,\s]+/).filter(Boolean)) {
    const m = part.match(/^(\d+)(?:-(\d+))?$/);
    if (!m) continue;
    const a = Number(m[1]);
    const b = m[2] ? Number(m[2]) : a;
    for (let n = Math.min(a, b); n <= Math.max(a, b) && n - Math.min(a, b) < 100_000; n++) if (available.has(n)) out.add(n);
  }
  return [...out].sort((x, y) => x - y);
}

/** "urgent? → 92%" / "team? → infra (70%)" — a compact one-line preview of a question+target pair. */
function previewQt(q: SysOneQuestion, t: SysOneTarget | undefined): string {
  if (!t) return q.text;
  if (q.type === "noul" && t.type === "noul") return `${q.text} → ${Math.round(t.p * 100)}%`;
  if ((q.type === "choice" || q.type === "score") && (t.type === "choice" || t.type === "score")) {
    const labels = q.type === "choice" ? q.options : q.levels;
    let best = 0;
    for (let i = 1; i < t.probs.length; i++) if (t.probs[i] > t.probs[best]) best = i;
    const pct = Math.round((t.probs[best] ?? 0) * 100);
    return `${q.text} → ${labels[best] ?? "?"} (${pct}%)`;
  }
  return q.text;
}

export function SysOneCard() {
  const docs = useStudio((s) => s.docs);
  const samples = useStudio((s) => s.sysoneSamples);
  const progress = useStudio((s) => s.sysoneGenProgress);
  const llm = useStudio((s) => s.llm);
  const conn = useStudio((s) => s.conn);
  const setSettingsOpen = useStudio((s) => s.setSettingsOpen);
  const [synthOpen, setSynthOpen] = useState(false);
  const [hfOpen, setHfOpen] = useState(false);

  const ready = docs.filter((d) => d.state === "ready");
  const [docId, setDocId] = useState("");
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [range, setRange] = useState("");
  const [perPage, setPerPage] = useState(3);
  const [parallel, setParallel] = useState(4);
  const [startError, setStartError] = useState("");
  const [showSamples, setShowSamples] = useState(true);

  const current = ready.find((d) => d.id === docId) ?? ready[0];
  const currentId = current?.id ?? "";

  const textBytes = current?.textBytes;
  const key = `${currentId}:${textBytes}`;
  const [loaded, setLoaded] = useState<{ key: string; list: QaPageList | null; error: string } | null>(null);
  useEffect(() => {
    if (!currentId || conn !== "connected") return;
    let live = true;
    loadQaPages(currentId).then((r) => {
      if (!live) return;
      setLoaded({ key, list: r.ok ? r.list : null, error: r.ok ? "" : r.error });
      if (r.ok) setPicked(new Set(r.list.pages.slice(0, 10).map((p) => p.n)));
    });
    return () => {
      live = false;
    };
  }, [currentId, key, conn]);
  const fresh = loaded?.key === key;
  const list = fresh ? loaded.list : null;
  const loadError = fresh ? loaded.error : "";
  const loading = !!currentId && conn === "connected" && !fresh;

  const available = useMemo(() => new Set(list?.pages.map((p) => p.n) ?? []), [list]);
  const running = progress?.state === "running";
  const llmReady = !!(llm.baseUrl.trim() && llm.apiKey.trim() && llm.model.trim());
  const label = list?.synthetic ? "section" : "page";
  const total = samples.length;
  const expected = picked.size * perPage;

  const toggle = (n: number) =>
    setPicked((p) => {
      const next = new Set(p);
      if (next.has(n)) next.delete(n);
      else next.add(n);
      return next;
    });

  async function start() {
    setStartError("");
    const err = await generateSysOne(currentId, [...picked].sort((a, b) => a - b), perPage, parallel);
    if (err) setStartError(err);
  }

  return (
    <Card>
      <SectionHeader
        title="System One dataset"
        sub="Have a language model write decision samples — a situation, typed questions, and calibrated probability targets — either grounded in pages you pick, or entirely invented for a domain."
        icon={<Brain size={16} className="text-accent" />}
        right={total > 0 ? <Badge tone="accent">{total.toLocaleString()} sample{total === 1 ? "" : "s"}</Badge> : undefined}
      />
      <div className="space-y-4 px-4 pb-4">
        <div className="flex items-center justify-between gap-2 rounded-lg border border-dashed border-accent/40 bg-accent-soft/20 px-3 py-2.5">
          <div className="min-w-0 text-xs text-muted">
            <span className="font-medium text-fg">No document? </span>
            Invent samples for any domain — networking, marketing, support, anything.
          </div>
          <Button size="sm" disabled={running} onClick={() => setSynthOpen(true)}><Wand2 size={13} /> Generate synthetic data</Button>
        </div>
        <SysOneSynthModal open={synthOpen} onClose={() => setSynthOpen(false)} />

        <div className="flex items-center justify-between gap-2 rounded-lg border border-dashed border-border px-3 py-2.5">
          <div className="min-w-0 text-xs text-muted">
            <span className="font-medium text-fg">Want real data? </span>
            Import labeled rows straight from a public Hugging Face dataset — no LLM needed.
          </div>
          <Button size="sm" disabled={running} onClick={() => setHfOpen(true)}><Download size={13} /> Import from Hugging Face</Button>
        </div>
        <SysOneHfImportModal open={hfOpen} onClose={() => setHfOpen(false)} />

        {ready.length === 0 ? (
          <p className="rounded-lg bg-surface-2 px-3 py-3 text-xs text-muted">Or upload a document above (PDF, Word, PowerPoint, text…) to generate samples grounded in its pages.</p>
        ) : (
          <>
            <Field label="Document">
              <SelectInput value={currentId} onChange={setDocId} disabled={running} options={ready.map((d) => ({ value: d.id, label: d.name }))} />
            </Field>

            {loading && <p className="flex items-center gap-2 text-xs text-muted"><Loader2 size={14} className="animate-spin" /> Finding pages…</p>}
            {loadError && <p className="text-xs text-danger">{loadError}</p>}

            {list && (
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-medium text-muted">
                    {picked.size} of {list.pages.length} {label}s selected
                  </span>
                  <span className="flex-1" />
                  <Button size="sm" disabled={running} onClick={() => setPicked(new Set(list.pages.map((p) => p.n)))}>All</Button>
                  <Button size="sm" disabled={running} onClick={() => setPicked(new Set())}>None</Button>
                </div>
                <input
                  value={range}
                  disabled={running}
                  onChange={(e) => setRange(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") setPicked(new Set(parseRanges(range, available)));
                  }}
                  placeholder="Type a range like 1-5, 9, 12-14 and press Enter"
                  aria-label={`Select ${label}s by number`}
                  className="h-8 w-full rounded-md border border-border bg-surface px-2 text-xs outline-none focus:border-accent disabled:opacity-50"
                />
                <div className="max-h-56 overflow-y-auto rounded-lg border border-border">
                  <ul className="divide-y divide-border">
                    {list.pages.map((p) => (
                      <li key={p.n}>
                        <label className={cn("flex cursor-pointer items-start gap-2.5 px-3 py-1.5 text-xs hover:bg-surface-2", running && "cursor-default opacity-70")}>
                          <input type="checkbox" checked={picked.has(p.n)} disabled={running} onChange={() => toggle(p.n)} className="mt-0.5 accent-[var(--accent)]" />
                          <span className="w-14 shrink-0 font-medium tabular-nums">{label === "page" ? "Page" : "Sec."} {p.n}</span>
                          <span className="min-w-0 flex-1 truncate text-muted" title={p.preview}>{p.preview}</span>
                          <span className="shrink-0 tabular-nums text-muted">{fmtCount(p.chars)}</span>
                        </label>
                      </li>
                    ))}
                  </ul>
                </div>
                {list.synthetic && <p className="text-[11px] text-muted">This format has no fixed pages, so the text is cut into ~3,500-character sections.{list.truncated ? " Only the first sections are offered." : ""}</p>}
                {list.pages.length === 0 && <p className="text-xs text-warn">No {label}s with enough text to write decision samples about.</p>}
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <Field label="Samples per page" help="Decision samples (situation + questions + targets) requested from the model for each selected page.">
                <NumberInput value={perPage} onChange={setPerPage} min={1} max={10} disabled={running} />
              </Field>
              <Field label="Parallel requests" help="Pages sent to the model at the same time.">
                <NumberInput value={parallel} onChange={setParallel} min={1} max={16} disabled={running} />
              </Field>
            </div>

            {!llmReady && (
              <div className="flex items-center gap-2 rounded-lg border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-warn">
                <AlertTriangle size={14} className="shrink-0" />
                <span className="flex-1">Connect a model server first (URL, API key, model).</span>
                <Button size="sm" onClick={() => setSettingsOpen(true)}><Settings2 size={13} /> Settings</Button>
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2">
              {running ? (
                <Button variant="danger" onClick={() => stopSysOneGenerate()}><Square size={13} fill="currentColor" /> Stop</Button>
              ) : (
                <Button variant="primary" disabled={conn !== "connected" || !llmReady || picked.size === 0} onClick={start}>
                  <Sparkles size={14} /> Generate ≈ {expected.toLocaleString()} sample{expected === 1 ? "" : "s"}
                </Button>
              )}
              {llmReady && <Button size="sm" variant="ghost" onClick={() => setSettingsOpen(true)} title="Model server settings"><Settings2 size={13} /> {llm.model}</Button>}
            </div>
            {startError && <p className="text-xs text-danger">{startError}</p>}
          </>
        )}

        {progress && (
          <div role="status" className="rounded-lg border border-border bg-surface-2/50 px-3 py-2.5">
            <div className="flex items-center gap-2 text-sm font-medium">
              {running && <Loader2 size={14} className="animate-spin text-accent" />}
              {running ? "Writing decision samples…" : progress.state === "done" ? "Finished" : progress.state === "stopped" ? "Stopped" : "Failed"}
              <span className="ml-auto text-xs font-normal tabular-nums text-muted">{progress.done}/{progress.total} requests · {progress.samples} samples{progress.failed ? ` · ${progress.failed} failed` : ""}</span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2">
              <div className={cn("h-full rounded-full transition-[width] duration-200", progress.state === "error" ? "bg-danger" : "bg-accent")} style={{ width: `${progress.total ? Math.max(2, (progress.done / progress.total) * 100) : 0}%` }} />
            </div>
            {progress.message && <p className={cn("mt-1.5 text-xs", progress.state === "error" || progress.failed ? "text-danger" : "text-muted")}>{progress.message}</p>}
          </div>
        )}

        {total > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <button onClick={() => setShowSamples((v) => !v)} className="text-xs font-medium text-muted hover:text-fg">{showSamples ? "Hide" : "Show"} samples</button>
              <span className="flex-1" />
              <Button size="sm" variant="ghost" disabled={running} onClick={() => clearSysOne()}><Trash2 size={13} /> Clear all</Button>
            </div>
            {showSamples && (
              <ul className="max-h-80 divide-y divide-border overflow-y-auto rounded-lg border border-border">
                {samples.map((sm) => (
                  <li key={sm.id} className="group flex items-start gap-2 px-3 py-2">
                    <div className="min-w-0 flex-1 text-xs leading-snug">
                      <div className="font-medium">{sm.state}</div>
                      <ul className="mt-1 space-y-0.5 text-muted">
                        {sm.questions.map((q, i) => <li key={i}>{previewQt(q, sm.targets[i])}</li>)}
                      </ul>
                      <div className="mt-1 text-[10px] text-muted/80">{sm.source}</div>
                    </div>
                    <button onClick={() => removeSysOne([sm.id])} aria-label="Delete this sample" className="rounded p-1 text-muted opacity-0 hover:bg-surface-2 hover:text-danger focus:opacity-100 group-hover:opacity-100"><Trash2 size={13} /></button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <DatasetIO kind="sysone" count={total} busy={running} />
      </div>
    </Card>
  );
}
