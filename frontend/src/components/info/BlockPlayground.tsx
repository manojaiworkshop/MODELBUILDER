"use client";
import { AlertTriangle, Loader2, MousePointerClick, Play, RotateCcw, ShieldCheck } from "lucide-react";
import { useRef, useState } from "react";
import { blockLabel, CATALOG } from "@/lib/blocks";
import { starterCode } from "@/lib/playgroundSnippets";
import { runInSandbox, type LogEntry } from "@/lib/sandbox";
import { useStudio } from "@/lib/store";
import type { Block } from "@/lib/types";
import { Button, cn } from "../ui/kit";
import { Heatmap } from "../ui/Heatmap";
import { BlockChips } from "./BlockChips";

function OutputLine({ entry }: { entry: LogEntry }) {
  return (
    <div className="space-y-1">
      <div className="whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed text-fg">{entry.text}</div>
      {entry.data && (
        <Heatmap
          data={entry.data.values}
          rows={entry.data.rows}
          cols={entry.data.cols}
          mode="diverging"
          cellH={entry.data.rows === 1 ? 20 : 16}
          cellW={entry.data.cols <= 6 ? 32 : undefined}
          className="max-w-md"
        />
      )}
    </div>
  );
}

function Editor({ block }: { block: Block }) {
  const saved = useStudio((s) => s.playgroundCode[block.id]);
  const setCode = useStudio((s) => s.setPlaygroundCode);
  const resetCode = useStudio((s) => s.resetPlaygroundCode);
  const starter = starterCode(block);
  const code = saved ?? starter;
  const dirty = saved !== undefined && saved !== starter;

  const [running, setRunning] = useState(false);
  // no effect needed to clear this on block switch: the parent remounts this component via
  // `key={active.id}`, so `result` already starts fresh whenever the active block changes.
  const [result, setResult] = useState<{ entries: LogEntry[]; error?: string; ms: number } | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  const run = async () => {
    setRunning(true);
    const r = await runInSandbox(code);
    setResult(r);
    setRunning(false);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <Button size="sm" variant="primary" onClick={run} disabled={running}>
          {running ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} fill="currentColor" />} Run
        </Button>
        <Button size="sm" disabled={!dirty} onClick={() => resetCode(block.id)} title="Restore the starter snippet">
          <RotateCcw size={13} /> Reset
        </Button>
        <div className="flex-1" />
        <span className="hidden items-center gap-1 text-[11px] text-muted sm:flex" title="Runs in an isolated Web Worker: no network, no storage, no access to this page. Nothing is sent to the server.">
          <ShieldCheck size={13} className="text-ok" /> runs in your browser
        </span>
      </div>
      <textarea
        ref={taRef}
        value={code}
        onChange={(e) => setCode(block.id, e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Tab") {
            e.preventDefault();
            const el = e.currentTarget;
            const { selectionStart: s, selectionEnd: en } = el;
            const next = `${code.slice(0, s)}  ${code.slice(en)}`;
            setCode(block.id, next);
            requestAnimationFrame(() => el.setSelectionRange(s + 2, s + 2));
          }
          if ((e.key === "Enter" && (e.metaKey || e.ctrlKey))) {
            e.preventDefault();
            run();
          }
        }}
        spellCheck={false}
        rows={12}
        aria-label={`Editable code for ${block.type}`}
        className="min-h-40 flex-1 resize-none border-b border-border bg-code px-3 py-2.5 font-mono text-[12.5px] leading-relaxed text-slate-200 outline-none"
        style={{ tabSize: 2 }}
      />
      <div className="min-h-24 flex-1 overflow-y-auto bg-surface px-3 py-2.5">
        {running && <div className="flex items-center gap-2 text-xs text-muted"><Loader2 size={13} className="animate-spin" /> Running…</div>}
        {!running && !result && <p className="text-xs text-muted">Press Run (or ⌘/Ctrl+Enter) to execute this snippet.</p>}
        {!running && result && (
          <div className="space-y-3">
            {result.entries.map((e, i) => <OutputLine key={i} entry={e} />)}
            {result.error && (
              <div className="flex items-start gap-2 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-xs text-danger">
                <AlertTriangle size={14} className="mt-0.5 shrink-0" /> {result.error}
              </div>
            )}
            {result.entries.length === 0 && !result.error && <p className="text-xs text-muted">Ran with no output — add a print(...) call to see something.</p>}
            <p className="text-[10px] text-muted">{result.ms} ms</p>
          </div>
        )}
      </div>
    </div>
  );
}

export function BlockPlayground() {
  const blocks = useStudio((s) => s.blocks);
  const codeBlockId = useStudio((s) => s.codeBlockId);
  const openBlockCode = useStudio((s) => s.openBlockCode);
  const active = blocks.find((b) => b.id === codeBlockId) ?? blocks[0] ?? null;

  if (blocks.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center text-sm text-muted">
        <MousePointerClick size={22} />
        <p>Add blocks on the Model tab, then click their <span className="mx-0.5 inline-flex h-4 w-4 items-center justify-center rounded-full border border-border-strong bg-surface-2 text-[9px] font-bold">{"</>"}</span> button to try one here.</p>
      </div>
    );
  }

  return (
    <div className={cn("flex min-h-0 flex-1 flex-col")}>
      <div className="space-y-1 border-b border-border bg-surface px-3 py-2.5">
        <div className="px-1 text-[11px] font-medium text-muted">Jump to a block</div>
        <BlockChips activeId={active?.id ?? null} onSelect={openBlockCode} />
      </div>
      {active && (
        <>
          <div className="flex items-center gap-2 px-3 pt-2.5">
            {(() => {
              const def = CATALOG[active.type];
              const Icon = def.icon;
              return <span className="flex h-6 w-6 items-center justify-center rounded-md" style={{ background: `color-mix(in srgb, ${def.color} 16%, transparent)`, color: def.color }}><Icon size={13} /></span>;
            })()}
            <span className="text-sm font-semibold">{blockLabel(blocks, blocks.findIndex((b) => b.id === active.id))}</span>
            <span className="text-xs text-muted">{CATALOG[active.type].summary(active)}</span>
          </div>
          <Editor key={active.id} block={active} />
        </>
      )}
    </div>
  );
}
