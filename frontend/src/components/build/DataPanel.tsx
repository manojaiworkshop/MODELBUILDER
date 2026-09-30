"use client";
import { AlertTriangle, BookOpen, File as FileIcon, FileArchive, FileJson, FileSpreadsheet, FileText, FileUp, FolderUp, Globe, Loader2, Presentation, Trash2, Upload, X } from "lucide-react";
import { useRef, useState, type ReactNode } from "react";
import { call } from "@/lib/backend";
import { fmtBytes, fmtCount, fmtSeconds } from "@/lib/format";
import { SAMPLES } from "@/lib/samples";
import { useStudio } from "@/lib/store";
import type { DocInfo, UploadItem } from "@/lib/types";
import { cancelUpload, MAX_FILE_BYTES, uploadFiles } from "@/lib/upload";
import { QaCard } from "./QaCard";
import { SummaryCard } from "./SummaryCard";
import { importTokenizer, resetTokenizer } from "@/lib/sync";
import { Badge, Button, Card, Field, NumberInput, SectionHeader, Segmented, Toggle, cn } from "../ui/kit";

const TINTS = ["#6366f1", "#0ea5e9", "#14b8a6", "#f59e0b", "#f43f5e", "#8b5cf6"];
const vis = (t: string) => t.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥");
const MAX_PASTE = 2_000_000;

const KIND_ICON: Record<string, { icon: typeof FileIcon; color: string }> = {
  PDF: { icon: FileText, color: "#ef4444" },
  Word: { icon: FileText, color: "#3b82f6" },
  PowerPoint: { icon: Presentation, color: "#f97316" },
  Excel: { icon: FileSpreadsheet, color: "#22c55e" },
  OpenDocument: { icon: FileText, color: "#0ea5e9" },
  EPUB: { icon: BookOpen, color: "#a855f7" },
  HTML: { icon: Globe, color: "#14b8a6" },
  RTF: { icon: FileText, color: "#64748b" },
  ZIP: { icon: FileArchive, color: "#eab308" },
  gzip: { icon: FileArchive, color: "#eab308" },
  text: { icon: FileText, color: "#64748b" },
};

// ------------------------------------------------------------- folder drops

async function walk(entry: FileSystemEntry, out: File[], budget: { left: number }): Promise<void> {
  if (budget.left <= 0) return;
  if (entry.isFile) {
    const f = await new Promise<File>((ok, err) => (entry as FileSystemFileEntry).file(ok, err));
    // keep the folder path in the name; File wraps the same bytes, nothing is copied
    out.push(new File([f], entry.fullPath.replace(/^\//, ""), { type: f.type, lastModified: f.lastModified }));
    budget.left--;
  } else if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    for (;;) {
      const batch = await new Promise<FileSystemEntry[]>((ok, err) => reader.readEntries(ok, err));
      if (!batch.length) break;
      for (const e of batch) await walk(e, out, budget);
    }
  }
}

async function filesFromDrop(dt: DataTransfer): Promise<{ files: File[]; capped: boolean }> {
  const budget = { left: 5000 };
  const out: File[] = [];
  const entries = Array.from(dt.items ?? [])
    .map((i) => i.webkitGetAsEntry?.())
    .filter((e): e is FileSystemEntry => !!e);
  if (entries.length) {
    for (const e of entries) await walk(e, out, budget);
  } else {
    out.push(...Array.from(dt.files));
  }
  return { files: out, capped: budget.left <= 0 };
}

// ------------------------------------------------------------------ rows

function DocRow({ d }: { d: DocInfo }) {
  const k = d.kind ? KIND_ICON[d.kind] ?? { icon: FileIcon, color: "#64748b" } : { icon: FileIcon, color: "#94a3b8" };
  const Icon = k.icon;
  const remove = () => call("data:remove", { id: d.id }).catch(() => {});
  const toggle = (enabled: boolean) => call("data:toggle", { id: d.id, enabled }).catch(() => {});
  return (
    <li className={cn("flex items-start gap-3 px-4 py-2.5", d.state === "ready" && !d.enabled && "opacity-55")}>
      <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg" style={{ background: `color-mix(in srgb, ${k.color} 15%, transparent)`, color: k.color }}>
        {d.state === "extracting" ? <Loader2 size={16} className="animate-spin" /> : <Icon size={16} />}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="max-w-full truncate text-sm font-medium" title={d.name}>{d.name}</span>
          {d.kind && <Badge>{d.kind}</Badge>}
        </div>
        <div className="mt-0.5 text-xs text-muted">
          {d.state === "extracting" && "Reading text out of the file…"}
          {d.state === "ready" && <>{fmtBytes(d.size)} → <b className="font-medium text-fg">{fmtBytes(d.textBytes)}</b> of text{d.note ? ` · ${d.note}` : ""}</>}
          {d.state === "error" && <span className="text-danger">{d.error}</span>}
        </div>
      </div>
      {d.state === "ready" && <Toggle checked={d.enabled} onChange={toggle} label={`Use ${d.name} for training`} />}
      <button onClick={remove} aria-label={`Remove ${d.name}`} className="rounded p-1.5 text-muted hover:bg-surface-2 hover:text-danger"><Trash2 size={15} /></button>
    </li>
  );
}

function UploadRow({ u }: { u: UploadItem }) {
  const pct = u.size ? Math.min(100, (u.loaded / u.size) * 100) : 0;
  return (
    <li className="flex items-start gap-3 px-4 py-2.5">
      <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent-soft text-accent">
        {u.state === "error" ? <AlertTriangle size={16} className="text-danger" /> : <FileUp size={16} />}
      </span>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium" title={u.name}>{u.name}</div>
        {u.state === "error" ? (
          <div className="mt-0.5 text-xs text-danger">{u.error}</div>
        ) : (
          <>
            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-2" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
              <div className="h-full rounded-full bg-accent transition-[width] duration-200" style={{ width: `${u.state === "queued" ? 0 : pct}%` }} />
            </div>
            <div className="mt-1 text-xs tabular-nums text-muted">{u.state === "queued" ? "Waiting…" : `Uploading ${fmtBytes(u.loaded)} of ${fmtBytes(u.size)} · ${pct.toFixed(0)}%`}</div>
          </>
        )}
      </div>
      <button onClick={() => cancelUpload(u.id)} aria-label={u.state === "error" ? `Dismiss ${u.name}` : `Cancel ${u.name}`} className="rounded p-1.5 text-muted hover:bg-surface-2 hover:text-fg"><X size={15} /></button>
    </li>
  );
}

// ------------------------------------------------------------------ cards

function SourcesCard() {
  const docs = useStudio((s) => s.docs);
  const uploads = useStudio((s) => s.uploads);
  const conn = useStudio((s) => s.conn);
  const text = useStudio((s) => s.text);
  const usePasted = useStudio((s) => s.usePasted);
  const setUsePasted = useStudio((s) => s.setUsePasted);
  const [drag, setDrag] = useState(false);
  const [note, setNote] = useState("");
  const filesInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const disabled = conn !== "connected";

  const total = docs.reduce((a, d) => a + d.size, 0);
  const pastedBytes = new Blob([text]).size;
  const rows = docs.length + uploads.length + (text ? 1 : 0);

  async function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDrag(false);
    if (disabled) return;
    const { files, capped } = await filesFromDrop(e.dataTransfer);
    setNote(capped ? "Only the first 5,000 files of that folder were queued." : "");
    uploadFiles(files);
  }

  return (
    <Card>
      <SectionHeader
        title="Training data"
        sub="Upload anything readable — PDF, Word, Excel, PowerPoint, text, code, CSV, HTML, EPUB, ZIP — and train on it."
        right={docs.length > 0 ? <Badge tone="accent">{docs.length} file{docs.length === 1 ? "" : "s"} · {fmtBytes(total)}</Badge> : undefined}
      />
      <div className="px-4 pb-3">
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          className={cn("rounded-xl border-2 border-dashed px-4 py-6 text-center transition-colors", drag ? "border-accent bg-accent-soft" : "border-border")}
        >
          <FileUp className="mx-auto mb-2 text-muted" size={24} />
          <p className="text-sm font-medium">{drag ? "Drop to upload" : "Drag files or folders here"}</p>
          <p className="mt-1 text-xs text-muted">Up to {MAX_FILE_BYTES / 1024 ** 3} GB per file · scanned PDFs (images) and legacy .doc/.ppt can’t be read</p>
          <div className="mt-3 flex flex-wrap justify-center gap-2">
            <input ref={filesInput} type="file" multiple className="hidden" onChange={(e) => { uploadFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
            <input
              ref={(el) => { folderInput.current = el; el?.setAttribute("webkitdirectory", ""); }}
              type="file"
              multiple
              className="hidden"
              onChange={(e) => { uploadFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }}
            />
            <Button size="sm" disabled={disabled} onClick={() => filesInput.current?.click()}><FileUp size={14} /> Choose files</Button>
            <Button size="sm" disabled={disabled} onClick={() => folderInput.current?.click()}><FolderUp size={14} /> Choose folder</Button>
          </div>
          {disabled && <p className="mt-2 text-xs text-warn">Backend not connected.</p>}
          {note && <p className="mt-2 text-xs text-warn">{note}</p>}
        </div>
      </div>

      {rows > 0 && (
        <ul className="divide-y divide-border border-t border-border">
          {text.length > 0 && (
            <li className={cn("flex items-center gap-3 px-4 py-2.5", !usePasted && "opacity-55")}>
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-surface-2 text-muted"><FileText size={16} /></span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium">Pasted text</div>
                <div className="text-xs text-muted">{fmtBytes(pastedBytes)} · edit it in “Paste or pick a sample” below</div>
              </div>
              <Toggle checked={usePasted} onChange={setUsePasted} label="Use the pasted text for training" />
            </li>
          )}
          {uploads.map((u) => <UploadRow key={u.id} u={u} />)}
          {docs.map((d) => <DocRow key={d.id} d={d} />)}
        </ul>
      )}
    </Card>
  );
}

function PasteCard() {
  const text = useStudio((s) => s.text);
  const setText = useStudio((s) => s.setText);
  const setUsePasted = useStudio((s) => s.setUsePasted);
  return (
    <Card>
      <details>
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold">
          Paste or pick a sample <span className="ml-1 text-xs font-normal text-muted">({text.length.toLocaleString()} characters)</span>
        </summary>
        <div className="space-y-3 px-4 pb-4">
          <div className="flex flex-wrap gap-1.5">
            {SAMPLES.map((s) => (
              <Button key={s.id} size="sm" title={s.blurb} onClick={() => { setText(s.text, s.probe); setUsePasted(true); }}>{s.name}</Button>
            ))}
          </div>
          <textarea
            value={text}
            onChange={(e) => { setText(e.target.value); setUsePasted(true); }}
            spellCheck={false}
            rows={9}
            aria-label="Pasted training text"
            placeholder="Paste text here, or pick a sample above…"
            className="w-full resize-y rounded-lg border border-border bg-surface px-3 py-2 font-mono text-xs leading-relaxed outline-none focus:border-accent"
          />
          {text.length > MAX_PASTE && <p className="text-xs text-danger">Pasted text is limited to 2 MB — upload it as a file instead.</p>}
        </div>
      </details>
    </Card>
  );
}

function BuildProgressBar() {
  const p = useStudio((s) => s.progress);
  const imported = useStudio((s) => s.tokenizer.kind === "imported");
  if (!p) return null;
  return (
    <div role="status" className="rounded-xl border border-border bg-surface px-4 py-3 shadow-card">
      <div className="flex items-center gap-2 text-sm font-medium"><Loader2 size={15} className="animate-spin text-accent" /> {p.stage === "tokenizer" ? (imported ? "Loading the tokenizer" : "Learning the vocabulary") : "Tokenizing your data"}</div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-accent transition-[width] duration-200" style={{ width: `${Math.max(3, p.pct * 100)}%` }} /></div>
      <div className="mt-1 text-xs tabular-nums text-muted">{p.message}</div>
    </div>
  );
}

function TokenizerCard() {
  const merges = useStudio((s) => s.merges);
  const info = useStudio((s) => s.dataInfo);
  const progress = useStudio((s) => s.progress);
  const error = useStudio((s) => s.dataError);
  const conn = useStudio((s) => s.conn);
  const seqLen = useStudio((s) => s.hyper.seqLen);
  const setMerges = useStudio((s) => s.setMerges);
  const tok = useStudio((s) => s.tokenizer);
  const imported = tok.kind === "imported";
  const [wantImport, setWantImport] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const file = useRef<HTMLInputElement>(null);
  const mode = imported || wantImport ? "import" : "default";

  async function pick(f: File) {
    setUploading(true);
    setUploadError("");
    const err = await importTokenizer(f);
    setUploading(false);
    if (err) setUploadError(err);
  }

  return (
    <Card>
      <SectionHeader
        title="Tokenizer"
        sub={imported ? "An imported Hugging Face tokenizer.json. Its vocabulary is cut down to the tokens your data actually uses." : "Byte-level BPE trained on your data. Text is split into words first, then frequent letter pairs are merged inside each word."}
        right={progress ? <Loader2 size={15} className="animate-spin text-muted" /> : undefined}
      />
      <div className="space-y-4 px-4 pb-4">
        <Segmented
          value={mode}
          onChange={(v) => {
            setUploadError("");
            if (v === "default") {
              setWantImport(false);
              if (imported) void resetTokenizer();
            } else setWantImport(true);
          }}
          options={[
            { value: "default", label: "Default tokenizer", title: "Byte-level BPE learned from your data" },
            { value: "import", label: "Import tokenizer.json", title: "Use a Hugging Face tokenizer.json (BPE, Unigram, WordPiece, WordLevel)" },
          ]}
        />
        <input
          ref={file}
          type="file"
          accept=".json,application/json"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (f) void pick(f);
          }}
        />
        {mode === "import" && (
          <div className="rounded-lg border border-border bg-surface-2/50 px-3 py-3">
            {imported ? (
              <div className="flex flex-wrap items-center gap-2">
                <FileJson size={18} className="shrink-0 text-accent" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium" title={tok.name}>{tok.name}</div>
                  <div className="text-xs text-muted">{tok.model} · {tok.fullVocab.toLocaleString()} token ids{info ? <> · your data uses <b className="text-fg">{info.vocabSize.toLocaleString()}</b> of them, which become the model’s vocabulary</> : null}</div>
                </div>
                <Button size="sm" disabled={uploading} onClick={() => file.current?.click()}>{uploading ? <Loader2 size={13} className="animate-spin" /> : <Upload size={13} />} Replace</Button>
              </div>
            ) : (
              <div className="flex flex-wrap items-center gap-3">
                <Button variant="primary" disabled={uploading || conn !== "connected"} onClick={() => file.current?.click()}>
                  {uploading ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />} Choose tokenizer.json
                </Button>
                <p className="min-w-0 flex-1 text-xs text-muted">Any Hugging Face <code>tokenizer.json</code> — for example the one in a Gemma, Llama, Qwen or GPT-2 model folder. Until you choose one, the default tokenizer is used.</p>
              </div>
            )}
            {uploadError && <p role="alert" className="mt-2 text-xs text-danger">{uploadError}</p>}
          </div>
        )}
        {!imported && <div className="grid grid-cols-[1fr_88px] items-end gap-3">
          <Field label={`BPE merges: ${merges}`} help="0 = plain bytes (one token per character for ASCII).">
            <input type="range" min={0} max={1000} step={10} value={merges} onChange={(e) => setMerges(Number(e.target.value))} className="w-full" aria-label="BPE merges" />
          </Field>
          <NumberInput value={merges} onChange={setMerges} min={0} max={1000} />
        </div>}
        {!imported && <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-muted">Presets:</span>
          {[
            [0, "Characters", "One token per byte — no merging"],
            [100, "Small", "Common short words and endings"],
            [200, "Subword", "Frequent words become single tokens"],
            [500, "Large", "As many merges as the text supports"],
          ].map(([n, label, help]) => (
            <Button key={n} size="sm" variant={merges === n ? "primary" : "soft"} title={String(help)} onClick={() => setMerges(Number(n))}>{label}</Button>
          ))}
        </div>}

        {conn !== "connected" && <p className="text-xs text-warn">Backend not connected — start it to see tokenization.</p>}
        {error && !progress && <p className="rounded-lg border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-warn">{error}</p>}
        {info && info.tokens < seqLen + 2 && (
          <p className="rounded-lg border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-warn">
            After tokenization there are only <b>{info.tokens}</b> tokens — fewer than the context length ({seqLen}) + 2, so training can’t start. {imported ? "Add more text" : "Lower the BPE merges (very repetitive text collapses into a few giant tokens), add more text"}, or shorten the context in the Training tab.
          </p>
        )}
        {info?.truncated && (
          <p className="rounded-lg border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-warn">
            The data is larger than the server’s token limit, so only its beginning ({fmtCount(info.tokens)} tokens) is used for training.
          </p>
        )}
        {info && (
          <>
            <div className="grid grid-cols-2 gap-2 text-center sm:grid-cols-4">
              {[
                ["Source text", fmtBytes(info.bytes)],
                ["Tokens", fmtCount(info.tokens)],
                [imported ? "Vocabulary (used)" : "Vocabulary", info.vocabSize.toLocaleString()],
                ["Chars / token", info.compression.toFixed(2)],
              ].map(([k, v]) => (
                <div key={k} className="rounded-lg bg-surface-2 px-2 py-2">
                  <div className="text-sm font-semibold tabular-nums">{v}</div>
                  <div className="text-[11px] text-muted">{k}</div>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted">
              {info.sources} source{info.sources === 1 ? "" : "s"} · built in {fmtSeconds(info.buildMs)}
              {!imported && info.merges < merges && <> · learned {info.merges} of {merges} merges — the data has no more letter pairs that repeat, so more merges wouldn’t change anything. Words that appear only once stay split into pieces.</>}
            </p>
            <div>
              <div className="mb-1.5 text-xs font-medium text-muted">How the start of your data is split (first {info.preview.length} tokens; ␣ = space, ↵ = newline)</div>
              <div className="flex flex-wrap gap-1">
                {info.preview.map((t, i) => (
                  <span
                    key={i}
                    title={`id ${t.id}`}
                    className="rounded px-1.5 py-0.5 font-mono text-[11px]"
                    style={{ background: `color-mix(in srgb, ${TINTS[i % TINTS.length]} 18%, transparent)`, color: `color-mix(in srgb, ${TINTS[i % TINTS.length]} 70%, var(--fg))` }}
                  >
                    {vis(t.text)}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </Card>
  );
}

export function DataPanel(): ReactNode {
  return (
    <div className="space-y-4 p-4">
      <SourcesCard />
      <QaCard />
      <SummaryCard />
      <PasteCard />
      <BuildProgressBar />
      <TokenizerCard />
    </div>
  );
}
