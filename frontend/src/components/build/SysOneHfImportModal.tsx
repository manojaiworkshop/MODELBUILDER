"use client";
import { AlertTriangle, Download, Loader2, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { fetchSysOneHfInfo, importSysOneHf } from "@/lib/sync";
import { useStudio } from "@/lib/store";
import type { HfDatasetInfo } from "@/lib/types";
import { Button, Field, Modal, NumberInput, SelectInput } from "../ui/kit";

const EXAMPLES = ["dair-ai/emotion", "fancyzhx/ag_news", "SetFit/sst5"];
/** Matches the backend's `sysone::generate::MAX_SAMPLES` — the real ceiling is how many samples a
 * session can hold in total (shared with the other two generation modes), not a smaller
 * import-specific limit. */
const MAX_ROWS = 20_000;

/** Imports a slice of a real, public Hugging Face classification dataset, converting rows into
 * Choice-question samples (gold class + random distractors) — no LLM involved, just real labeled
 * data. Uses HF's public datasets-server API, so no token or auth is needed. */
export function SysOneHfImportModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const conn = useStudio((s) => s.conn);
  const [dataset, setDataset] = useState("");
  const [looking, setLooking] = useState(false);
  const [info, setInfo] = useState<HfDatasetInfo | null>(null);
  const [lookupError, setLookupError] = useState("");

  const [splitKey, setSplitKey] = useState(""); // "config\u0000split"
  const [textCol, setTextCol] = useState("");
  const [labelCol, setLabelCol] = useState("");
  const [optionsPerQ, setOptionsPerQ] = useState(4);
  const [total, setTotal] = useState(200);
  const [questionText, setQuestionText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const split = useMemo(() => info?.splits.find((s) => `${s.config}\u0000${s.split}` === splitKey), [info, splitKey]);
  const labelColumn = useMemo(() => info?.columns.find((c) => c.name === labelCol), [info, labelCol]);
  const classCount = labelColumn?.classNames?.length ?? 0;

  async function lookup() {
    setLookupError("");
    setInfo(null);
    setLooking(true);
    const res = await fetchSysOneHfInfo(dataset.trim());
    setLooking(false);
    if (!res.ok) {
      setLookupError(res.error);
      return;
    }
    setInfo(res.info);
    const firstSplit = res.info.splits.find((s) => s.split === "train") ?? res.info.splits[0];
    setSplitKey(firstSplit ? `${firstSplit.config}\u0000${firstSplit.split}` : "");
    const text = res.info.columns.find((c) => c.dtype === "string" && !c.classNames);
    const label = res.info.columns.find((c) => c.classNames && c.classNames.length > 0);
    setTextCol(text?.name ?? res.info.columns[0]?.name ?? "");
    setLabelCol(label?.name ?? "");
    setTotal(Math.min(200, firstSplit?.numRows ?? 200));
  }

  const ready = !!split && !!textCol && !!labelCol && classCount >= 2 && total > 0;

  async function doImport() {
    if (!split) return;
    setError("");
    setBusy(true);
    const err = await importSysOneHf({
      dataset: dataset.trim(),
      config: split.config,
      split: split.split,
      textCol,
      labelCol,
      optionsPerQ,
      total,
      questionText: questionText.trim() || undefined,
    });
    setBusy(false);
    if (err) setError(err);
    else onClose();
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Import from Hugging Face"
      sub="Pull real labeled rows from a public classification dataset and convert them into Choice-question samples — gold class plus a few random distractors, no LLM involved."
      icon={<Download size={16} className="text-accent" />}
    >
      <div className="space-y-4">
        <Field label="Dataset repo id" help='e.g. "dair-ai/emotion" — the part after huggingface.co/datasets/'>
          <div className="flex gap-2">
            <input
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && dataset.trim() && lookup()}
              placeholder="dair-ai/emotion"
              spellCheck={false}
              className="h-9 min-w-0 flex-1 rounded-lg border border-border bg-surface px-3 font-mono text-sm outline-none focus:border-accent"
            />
            <Button size="sm" disabled={!dataset.trim() || looking || conn !== "connected"} onClick={lookup}>
              {looking ? <Loader2 size={13} className="animate-spin" /> : <Search size={13} />} Look up
            </Button>
          </div>
          <p className="mt-1 flex flex-wrap gap-x-1.5 text-[11px] text-muted">
            Try: {EXAMPLES.map((d, i) => (
              <span key={d}>
                <button type="button" onClick={() => setDataset(d)} className="underline decoration-dotted hover:text-accent">{d}</button>
                {i < EXAMPLES.length - 1 ? "," : ""}
              </span>
            ))}
          </p>
        </Field>

        {lookupError && (
          <div className="flex items-center gap-2 rounded-lg border border-danger/40 bg-danger/5 px-3 py-2 text-xs text-danger">
            <AlertTriangle size={14} className="shrink-0" /> {lookupError}
          </div>
        )}

        {info && (
          <>
            <Field label="Split">
              <SelectInput
                value={splitKey}
                onChange={setSplitKey}
                options={info.splits.map((s) => ({ value: `${s.config}\u0000${s.split}`, label: `${s.config !== s.split ? `${s.config} / ` : ""}${s.split} (${s.numRows.toLocaleString()} rows)` }))}
              />
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Text column" help="Which column holds the text to use as the situation.">
                <SelectInput value={textCol} onChange={setTextCol} options={info.columns.map((c) => ({ value: c.name, label: `${c.name} (${c.dtype})` }))} />
              </Field>
              <Field label="Label column" help="Which column holds the class to predict.">
                <SelectInput value={labelCol} onChange={setLabelCol} options={info.columns.map((c) => ({ value: c.name, label: c.classNames ? `${c.name} (${c.classNames.length} classes)` : `${c.name} (${c.dtype})` }))} />
              </Field>
            </div>

            {labelColumn && !labelColumn.classNames && (
              <p className="text-xs text-warn">
                <AlertTriangle size={12} className="mr-1 inline" />
                &quot;{labelColumn.name}&quot; doesn&apos;t look like a classification label column (no fixed set of classes) — pick a different one.
              </p>
            )}
            {labelColumn?.classNames && (
              <p className="text-[11px] text-muted">Classes: {labelColumn.classNames.join(", ")}</p>
            )}

            <Field label="Question text (optional)" help='Defaults to "which category best fits this?"'>
              <input
                value={questionText}
                onChange={(e) => setQuestionText(e.target.value)}
                placeholder="which category best fits this?"
                className="h-9 w-full rounded-lg border border-border bg-surface px-3 text-sm outline-none focus:border-accent"
              />
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Options per question" help="Gold class plus this many random distractors (so the model learns to decide within a subset, not always see every class).">
                <NumberInput value={optionsPerQ} onChange={setOptionsPerQ} min={2} max={Math.max(2, Math.min(10, classCount || 10))} />
              </Field>
              <Field label="Rows to import" help="Converted into one sample each. Fetched a few hundred rows per second from Hugging Face; the progress bar tracks each page.">
                <NumberInput value={total} onChange={setTotal} min={1} max={Math.min(MAX_ROWS, split?.numRows ?? MAX_ROWS)} />
              </Field>
            </div>
          </>
        )}

        {error && <p className="text-xs text-danger">{error}</p>}

        <div className="flex justify-between border-t border-border pt-3">
          <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
          <Button variant="primary" size="sm" disabled={!ready || busy || conn !== "connected"} onClick={doImport}>
            <Download size={13} /> Import {total.toLocaleString()} row{total === 1 ? "" : "s"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
