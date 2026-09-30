"use client";
import { AlertTriangle, FileText, Loader2, Settings2, Sparkles, Square, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { clearSummaries, generateSummaries, loadQaPages, removeSummaries, stopSummarize } from "@/lib/sync";
import { fmtCount } from "@/lib/format";
import { useStudio } from "@/lib/store";
import type { DataStyle, QaPageList } from "@/lib/types";
import { DatasetIO } from "./DatasetIO";
import { StyleField } from "./StyleField";
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

export function SummaryCard() {
  const [collapsed, setCollapsed] = useState(false);
  const [style, setStyle] = useState<DataStyle>("simple");
  const [system, setSystem] = useState("");
  const docs = useStudio((s) => s.docs);
  const summaries = useStudio((s) => s.summaries);
  const progress = useStudio((s) => s.summaryProgress);
  const llm = useStudio((s) => s.llm);
  const conn = useStudio((s) => s.conn);
  const setSettingsOpen = useStudio((s) => s.setSettingsOpen);
  const setTrainSource = useStudio((s) => s.setTrainSource);

  const ready = docs.filter((d) => d.state === "ready");
  const [docId, setDocId] = useState("");
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [range, setRange] = useState("");
  const [sentences, setSentences] = useState(3);
  const [parallel, setParallel] = useState(4);
  const [startError, setStartError] = useState("");
  const [showPairs, setShowPairs] = useState(true);

  // the picked document, or the first one while nothing valid is picked
  const current = ready.find((d) => d.id === docId) ?? ready[0];
  const currentId = current?.id ?? "";

  // (re)load the page list whenever the document (or its extracted text) changes
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
  const total = summaries.length;

  const toggle = (n: number) =>
    setPicked((p) => {
      const next = new Set(p);
      if (next.has(n)) next.delete(n);
      else next.add(n);
      return next;
    });

  async function start() {
    setStartError("");
    const err = await generateSummaries(currentId, [...picked].sort((a, b) => a - b), sentences, parallel, style, system);
    if (err) setStartError(err);
  }

  return (
    <Card>
      <SectionHeader
        title="Summary dataset"
        sub="Have a language model write a summary of each page you pick, then train on them from the Observe tab."
        collapsed={collapsed}
        onToggle={() => setCollapsed((v) => !v)}
        icon={<FileText size={16} className="text-accent" />}
        right={total > 0 ? <Badge tone="accent">{total.toLocaleString()} summar{total === 1 ? "y" : "ies"}</Badge> : undefined}
      />
      {!collapsed && <div className="space-y-4 px-4 pb-4">
        {ready.length === 0 ? (
          <p className="rounded-lg bg-surface-2 px-3 py-3 text-xs text-muted">Upload a document above (PDF, Word, PowerPoint, text…) and its pages will show up here.</p>
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
                  placeholder={`Type a range like 1-5, 9, 12-14 and press Enter`}
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
                {list.pages.length === 0 && <p className="text-xs text-warn">No {label}s with enough text to summarize.</p>}
              </div>
            )}

            <StyleField kind="summary" value={style} onChange={setStyle} system={system} onSystem={setSystem} disabled={running} />

            <div className="grid grid-cols-2 gap-3">
              <Field label="Sentences per summary" help="Roughly how long each page's summary should be.">
                <NumberInput value={sentences} onChange={setSentences} min={1} max={10} disabled={running} />
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
                <Button variant="danger" onClick={() => stopSummarize()}><Square size={13} fill="currentColor" /> Stop</Button>
              ) : (
                <Button variant="primary" disabled={conn !== "connected" || !llmReady || picked.size === 0} onClick={start}>
                  <Sparkles size={14} /> Generate ≈ {picked.size.toLocaleString()} summar{picked.size === 1 ? "y" : "ies"}
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
              {running ? "Writing summaries…" : progress.state === "done" ? "Finished" : progress.state === "stopped" ? "Stopped" : "Failed"}
              <span className="ml-auto text-xs font-normal tabular-nums text-muted">{progress.done}/{progress.total} pages · {progress.summaries} summaries{progress.failed ? ` · ${progress.failed} failed` : ""}</span>
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
              <button onClick={() => setShowPairs((v) => !v)} className="text-xs font-medium text-muted hover:text-fg">{showPairs ? "Hide" : "Show"} summaries</button>
              <span className="flex-1" />
              <Button size="sm" onClick={() => setTrainSource("summary")} title="Use these summaries for the next training run">Train on these</Button>
              <Button size="sm" variant="ghost" disabled={running} onClick={() => clearSummaries()}><Trash2 size={13} /> Clear all</Button>
            </div>
            {showPairs && (
              <ul className="max-h-80 divide-y divide-border overflow-y-auto rounded-lg border border-border">
                {summaries.map((p) => (
                  <li key={p.id} className="group flex items-start gap-2 px-3 py-2">
                    <div className="min-w-0 flex-1 text-xs leading-snug">
                      <div className="font-medium">{p.summary}</div>
                      <div className="mt-1 text-[10px] text-muted/80">{p.doc}{p.page > 0 ? ` · p. ${p.page}` : ""}</div>
                    </div>
                    <button onClick={() => removeSummaries([p.id])} aria-label="Delete this summary" className="rounded p-1 text-muted opacity-0 hover:bg-surface-2 hover:text-danger focus:opacity-100 group-hover:opacity-100"><Trash2 size={13} /></button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <DatasetIO kind="summary" count={total} busy={running} />
      </div>}
    </Card>
  );
}
