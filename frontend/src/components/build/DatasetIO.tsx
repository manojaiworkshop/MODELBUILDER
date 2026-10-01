"use client";
import { Download, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { exportDataset, importDataset, type DatasetKind, type ExportFormat, type ExportStyle, type ImportStyle } from "@/lib/sync";
import { useStudio } from "@/lib/store";
import { Button, Field, SelectInput, Segmented } from "../ui/kit";

const STYLES: Record<DatasetKind, { value: ExportStyle; label: string }[]> = {
  qa: [
    { value: "hf", label: "Hugging Face · SQuAD (context, question, answers)" },
    { value: "simple", label: "Simple (context, question, answer)" },
    { value: "instruction", label: "Instruction · Alpaca (instruction, input, output)" },
    { value: "chat", label: "Chat (messages: system / user / assistant)" },
  ],
  summary: [
    { value: "hf", label: "Hugging Face · CNN/DailyMail (article, highlights)" },
    { value: "simple", label: "Simple (document, summary)" },
    { value: "instruction", label: "Instruction · Alpaca (instruction, input, output)" },
    { value: "chat", label: "Chat (messages: system / user / assistant)" },
  ],
  // no real alternate schema (unlike SQuAD/CNN-DailyMail/Alpaca/chat) — just the sample's own
  // {state, questions, targets} shape, so there is nothing to pick.
  sysone: [{ value: "simple", label: "System One samples (state, questions, targets)" }],
};

/** Export the generated pairs as Hugging Face-style JSONL, or import such a file, in three styles. */
export function DatasetIO({ kind, count, busy }: { kind: DatasetKind; count: number; busy: boolean }) {
  const conn = useStudio((s) => s.conn);
  const [style, setStyle] = useState<ExportStyle>(STYLES[kind][0].value);
  const [format, setFormat] = useState<ExportFormat>("jsonl");
  const [system, setSystem] = useState("");
  const [auto, setAuto] = useState(true);
  const [note, setNote] = useState<{ text: string; bad: boolean } | null>(null);
  const [working, setWorking] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const plural = (n: number) =>
    kind === "qa" ? `${n.toLocaleString()} pair${n === 1 ? "" : "s"}` : kind === "summary" ? `${n.toLocaleString()} summar${n === 1 ? "y" : "ies"}` : `${n.toLocaleString()} sample${n === 1 ? "" : "s"}`;
  const hasStyleChoice = STYLES[kind].length > 1;

  async function doExport() {
    setNote(null);
    setWorking(true);
    const err = await exportDataset(kind, style, format, system);
    setWorking(false);
    setNote(err ? { text: err, bad: true } : { text: `Saved ${plural(count)}${hasStyleChoice ? ` as ${style}` : ""} (${format}).`, bad: false });
  }

  async function doImport(f: File) {
    setNote(null);
    setWorking(true);
    const r = await importDataset(kind, auto ? "auto" : (style as ImportStyle), f);
    setWorking(false);
    if ("error" in r) return setNote({ text: r.error, bad: true });
    setNote({
      text: `Imported ${plural(r.added)}${r.skipped ? `, skipped ${r.skipped} that didn't fit the style` : ""}${r.capped ? " (limit reached)" : ""}.`,
      bad: r.added === 0,
    });
  }

  return (
    <div className="space-y-2 rounded-lg border border-border bg-surface-2/40 px-3 py-3">
      {hasStyleChoice && (
        <Field label="Hugging Face format" help="Used for export, and for import unless “detect automatically” is on. One JSON object per line, loadable with datasets.load_dataset('json', data_files=…).">
          <SelectInput value={style} onChange={(v) => setStyle(v as ExportStyle)} disabled={working} options={STYLES[kind]} />
        </Field>
      )}
      {style === "chat" && (
        <input
          value={system}
          disabled={working}
          onChange={(e) => setSystem(e.target.value)}
          maxLength={2000}
          placeholder="System prompt to add on export (optional)"
          aria-label="System prompt for export"
          className="h-8 w-full rounded-md border border-border bg-surface px-2 text-xs outline-none focus:border-accent disabled:opacity-50"
        />
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Segmented value={format} onChange={setFormat} options={[{ value: "jsonl", label: "JSONL" }, { value: "parquet", label: "Parquet" }]} />
        <Button size="sm" disabled={conn !== "connected" || working || count === 0} onClick={doExport} title={count === 0 ? "Nothing to export yet" : `Download as ${format.toUpperCase()}`}>
          <Download size={13} /> Export
        </Button>
        <Button size="sm" disabled={conn !== "connected" || working || busy} onClick={() => file.current?.click()} title="Add records from a .json / .jsonl / .parquet file">
          <Upload size={13} /> Import
        </Button>
        {hasStyleChoice && (
          <label className="ml-auto flex cursor-pointer items-center gap-1.5 text-xs text-muted">
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} className="accent-[var(--accent)]" /> Detect style on import
          </label>
        )}
        <input
          ref={file}
          type="file"
          accept=".json,.jsonl,.ndjson,.parquet,application/json"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (f) void doImport(f);
          }}
        />
      </div>
      {note && <p role="status" className={note.bad ? "text-xs text-danger" : "text-xs text-muted"}>{note.text}</p>}
    </div>
  );
}
