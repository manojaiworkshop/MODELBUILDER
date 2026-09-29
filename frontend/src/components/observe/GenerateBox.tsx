"use client";
import { Sparkles, Square } from "lucide-react";
import { useState } from "react";
import { useStudio } from "@/lib/store";
import { generateText, stopGenerate } from "@/lib/sync";
import { Badge, Button, Card, Field, NumberInput, SectionHeader } from "../ui/kit";

export function GenerateBox() {
  const probe = useStudio((s) => s.probe);
  const gen = useStudio((s) => s.gen);
  const samples = useStudio((s) => s.samples);
  const trainedStep = useStudio((s) => (s.inspect?.trained ? s.inspect.step : null) ?? (s.samples[0]?.step ?? null));
  const [prompt, setPrompt] = useState<string | null>(null);
  const [temp, setTemp] = useState(0.8);
  const [topK, setTopK] = useState(0);
  const [max, setMax] = useState(160);
  const p = prompt ?? probe;
  const ready = trainedStep !== null;

  return (
    <Card>
      <SectionHeader
        title="Generate text"
        sub="Sample from the weights of the latest run."
        right={<Badge tone={ready ? "ok" : "neutral"}>{ready ? `weights @ step ${trainedStep}` : "no weights yet"}</Badge>}
      />
      <div className="space-y-3 px-4 pb-4">
        <textarea
          value={p}
          onChange={(e) => setPrompt(e.target.value)}
          rows={2}
          spellCheck={false}
          aria-label="Generation prompt"
          className="w-full resize-y rounded-lg border border-border bg-surface px-3 py-2 font-mono text-xs outline-none focus:border-accent"
        />
        <div className="grid grid-cols-3 gap-3">
          <Field label="Temperature" help="0 = always the most likely token."><NumberInput value={temp} onChange={setTemp} min={0} max={5} step={0.1} float /></Field>
          <Field label="Top-k" help="0 = consider every token."><NumberInput value={topK} onChange={setTopK} min={0} max={500} /></Field>
          <Field label="New tokens"><NumberInput value={max} onChange={setMax} min={1} max={1000} step={20} /></Field>
        </div>
        <div className="flex items-center gap-2">
          {gen.running ? (
            <Button variant="danger" size="sm" onClick={() => stopGenerate()}><Square size={12} fill="currentColor" /> Stop</Button>
          ) : (
            <Button variant="primary" size="sm" disabled={!ready} onClick={() => generateText(p, { maxNew: max, temperature: temp, topK })}>
              <Sparkles size={13} /> Generate
            </Button>
          )}
          {!ready && <span className="text-xs text-muted">Train the model first.</span>}
        </div>
        {gen.error && <p className="text-xs text-danger">{gen.error}</p>}
        {(gen.text || gen.running) && (
          <div className="whitespace-pre-wrap break-words rounded-lg border border-border bg-surface-2/50 p-3 font-mono text-xs leading-relaxed">
            <span className="text-muted">{p}</span>
            <span className="text-accent">{gen.text}</span>
            {gen.running && <span className="ml-0.5 inline-block h-3 w-1.5 animate-pulse bg-accent align-middle" />}
          </div>
        )}
        {samples.length > 0 && (
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">Samples during training (from your trace sentence)</div>
            <div className="space-y-1.5">
              {samples.slice(0, 4).map((s, i) => (
                <div key={s.step} className={`rounded-lg border border-border px-3 py-2 text-xs ${i === 0 ? "fade-in" : "opacity-70"}`}>
                  <span className="mr-2 rounded bg-surface-2 px-1.5 py-0.5 text-[10px] tabular-nums text-muted">step {s.step}</span>
                  <span className="whitespace-pre-wrap break-words font-mono"><span className="text-muted">{s.prompt}</span>{s.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
