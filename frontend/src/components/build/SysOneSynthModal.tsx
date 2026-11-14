"use client";
import { AlertTriangle, Sparkles, Wand2 } from "lucide-react";
import { useState } from "react";
import { generateSysOneSynthetic } from "@/lib/sync";
import { useStudio } from "@/lib/store";
import { Button, Field, Modal, NumberInput } from "../ui/kit";

/** Domain + free-form seed context, a target question-type mix, and a sample count — the model
 * invents entirely new situations instead of grounding them in an uploaded document's pages. */
export function SysOneSynthModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const llm = useStudio((s) => s.llm);
  const setSettingsOpen = useStudio((s) => s.setSettingsOpen);
  const [domain, setDomain] = useState("");
  const [context, setContext] = useState("");
  const [noul, setNoul] = useState(34);
  const [choice, setChoice] = useState(33);
  const [score, setScore] = useState(33);
  const [total, setTotal] = useState(20);
  const [perBatch, setPerBatch] = useState(5);
  const [parallel, setParallel] = useState(4);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const llmReady = !!(llm.baseUrl.trim() && llm.model.trim());
  const mixSum = noul + choice + score;
  const ready = domain.trim().length > 0 && mixSum > 0 && total > 0;

  async function start() {
    setError("");
    setBusy(true);
    const err = await generateSysOneSynthetic(domain.trim(), context.trim(), { noul, choice, score }, total, perBatch, parallel);
    setBusy(false);
    if (err) setError(err);
    else onClose();
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Generate synthetic data"
      sub="No document needed — describe a domain and the model invents realistic situations, questions and calibrated answers from scratch."
      icon={<Wand2 size={16} className="text-accent" />}
    >
      <div className="space-y-4">
        <Field label="Domain" help='Short label for the kind of situations to invent, e.g. "networking", "marketing", "customer support".'>
          <input
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            placeholder="networking"
            spellCheck={false}
            className="h-9 w-full rounded-lg border border-border bg-surface px-3 text-sm outline-none focus:border-accent"
          />
        </Field>
        <Field label="Context (optional)" help="Anything more specific to seed the generation — example scenarios, a style, constraints to follow.">
          <textarea
            value={context}
            onChange={(e) => setContext(e.target.value)}
            rows={3}
            placeholder="e.g. outage and incident tickets for a small ISP's network operations team"
            className="w-full resize-none rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
          />
        </Field>

        <div>
          <div className="mb-1.5 flex items-center justify-between text-xs font-medium text-fg">
            <span>Question type mix</span>
            <span className={mixSum > 0 ? "text-muted" : "text-danger"}>{noul}:{choice}:{score} {mixSum > 0 && <>({Math.round((noul / mixSum) * 100)}% / {Math.round((choice / mixSum) * 100)}% / {Math.round((score / mixSum) * 100)}%)</>}</span>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Yes/no">
              <NumberInput value={noul} onChange={setNoul} min={0} max={100} />
            </Field>
            <Field label="Choice">
              <NumberInput value={choice} onChange={setChoice} min={0} max={100} />
            </Field>
            <Field label="Score">
              <NumberInput value={score} onChange={setScore} min={0} max={100} />
            </Field>
          </div>
          <p className="mt-1 text-[11px] text-muted">Relative weights — they don&apos;t need to add up to 100, the model treats them as a ratio for the whole batch.</p>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <Field label="Total samples" help="Split into requests of “per request” size below.">
            <NumberInput value={total} onChange={setTotal} min={1} max={500} />
          </Field>
          <Field label="Per request" help="Samples asked for in one model call.">
            <NumberInput value={perBatch} onChange={setPerBatch} min={1} max={10} />
          </Field>
          <Field label="Parallel requests">
            <NumberInput value={parallel} onChange={setParallel} min={1} max={16} />
          </Field>
        </div>

        {!llmReady && (
          <div className="flex items-center gap-2 rounded-lg border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-warn">
            <AlertTriangle size={14} className="shrink-0" />
            <span className="flex-1">Connect a model server first (URL and model name).</span>
            <Button
              size="sm"
              onClick={() => {
                // only one modal can be usably on top at a time (both use the same overlay z-index,
                // and this one mounts later in the DOM, so it would otherwise sit above Settings
                // and swallow its clicks) — close this one, the user reopens it after saving.
                onClose();
                setSettingsOpen(true);
              }}
            >
              Settings
            </Button>
          </div>
        )}
        {error && <p className="text-xs text-danger">{error}</p>}

        <div className="flex justify-between border-t border-border pt-3">
          <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
          <Button variant="primary" size="sm" disabled={!ready || !llmReady || busy} onClick={start}>
            <Sparkles size={13} /> Generate ≈ {total.toLocaleString()} sample{total === 1 ? "" : "s"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
