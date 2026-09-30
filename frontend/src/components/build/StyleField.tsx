"use client";
import type { DatasetKind } from "@/lib/sync";
import type { DataStyle } from "@/lib/types";
import { Field, SelectInput } from "../ui/kit";

const OPTIONS: Record<DatasetKind, { value: DataStyle; label: string; hint: string }[]> = {
  qa: [
    { value: "simple", label: "Simple", hint: "Free-form 1–3 sentence answers. Trains on “Question: … Answer: …”." },
    { value: "instruction", label: "Instruction (Alpaca)", hint: "Questions as instructions, full responses. Trains on “### Instruction / ### Input / ### Response”." },
    { value: "hf", label: "Hugging Face · SQuAD", hint: "Answers are short spans copied from the page. Trains on “Context: … Question: … Answer: …”." },
    { value: "chat", label: "Chat", hint: "Questions phrased for an assistant, conversational replies. Trains on <|user|> / <|assistant|> turns." },
  ],
  summary: [
    { value: "simple", label: "Simple", hint: "A short prose summary. Trains on “Text: … Summary: …”." },
    { value: "instruction", label: "Instruction (Alpaca)", hint: "The response to “Summarize the following text.” Trains on “### Instruction / ### Input / ### Response”." },
    { value: "hf", label: "Hugging Face · CNN/DailyMail", hint: "Short one-line highlights. Trains on “Article: … Highlights: …”." },
    { value: "chat", label: "Chat", hint: "An assistant's reply to “Summarize the following text”. Trains on <|user|> / <|assistant|> turns." },
  ],
};

/** What the model should write when generating, which also decides how the pairs are laid out for training. */
export function StyleField({ kind, value, onChange, system, onSystem, disabled }: { kind: DatasetKind; value: DataStyle; onChange: (v: DataStyle) => void; system: string; onSystem: (v: string) => void; disabled?: boolean }) {
  const opts = OPTIONS[kind];
  return (
    <Field label="Style" help="The format the language model writes in. Training then uses the matching text layout, so the model learns exactly this style.">
      <SelectInput value={value} onChange={(v) => onChange(v as DataStyle)} disabled={disabled} options={opts.map((o) => ({ value: o.value, label: o.label }))} />
      <p className="mt-1 text-[11px] text-muted">{opts.find((o) => o.value === value)?.hint}</p>
      {value === "chat" && (
        <input
          value={system}
          disabled={disabled}
          onChange={(e) => onSystem(e.target.value)}
          maxLength={2000}
          placeholder="Optional system prompt, e.g. You are a helpful assistant."
          aria-label="System prompt"
          className="mt-2 h-8 w-full rounded-md border border-border bg-surface px-2 text-xs outline-none focus:border-accent disabled:opacity-50"
        />
      )}
    </Field>
  );
}
