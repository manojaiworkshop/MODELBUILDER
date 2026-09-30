"use client";
import { CheckCircle2, Eye, EyeOff, Loader2, Settings2, XCircle } from "lucide-react";
import { useState } from "react";
import { DEFAULT_LLM, useStudio } from "@/lib/store";
import { testLlmConnection } from "@/lib/sync";
import type { LlmConfig } from "@/lib/types";
import { Button, Field, Modal } from "../ui/kit";

type TestState = { state: "idle" | "testing" | "ok" | "err"; message?: string };

export function LlmSettingsButton() {
  const open = useStudio((s) => s.settingsOpen);
  const setOpen = useStudio((s) => s.setSettingsOpen);
  return (
    <>
      <button
        onClick={() => setOpen(true)}
        aria-label="Model connection settings"
        title="Model connection settings"
        className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-fg"
      >
        <Settings2 size={15} />
      </button>
      <LlmSettingsModal open={open} onClose={() => setOpen(false)} />
    </>
  );
}

export function LlmSettingsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const saved = useStudio((s) => s.llm);
  const setLlm = useStudio((s) => s.setLlm);
  const [draft, setDraft] = useState<LlmConfig>(saved);
  const [showKey, setShowKey] = useState(false);
  const [test, setTest] = useState<TestState>({ state: "idle" });
  // Re-seed the draft from the last saved config each time the modal opens (not on every
  // render): https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes
  const [wasOpen, setWasOpen] = useState(open);
  if (open !== wasOpen) {
    setWasOpen(open);
    if (open) {
      setDraft(saved);
      setTest({ state: "idle" });
    }
  }

  const run = async () => {
    setTest({ state: "testing" });
    const res = await testLlmConnection(draft);
    setTest(res.ok ? { state: "ok", message: res.models?.length ? `Reachable — ${res.models.length} model${res.models.length === 1 ? "" : "s"} listed.` : "Reachable." } : { state: "err", message: res.error });
  };

  const save = () => {
    setLlm(draft);
    onClose();
  };

  const dirty = JSON.stringify(draft) !== JSON.stringify(saved);

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Model connection"
      sub="Where the Learn tab's chat sends your questions. An OpenAI-compatible /v1/chat/completions server — e.g. a self-hosted proxy in front of Claude — not a public cloud endpoint."
      icon={<Settings2 size={16} className="text-accent" />}
    >
      <div className="space-y-4">
        <Field label="API base URL" help="Everything before /chat/completions. No trailing slash needed.">
          <input
            value={draft.baseUrl}
            onChange={(e) => setDraft((d) => ({ ...d, baseUrl: e.target.value }))}
            placeholder="http://192.168.1.6:18090/v1"
            spellCheck={false}
            className="h-9 w-full rounded-lg border border-border bg-surface px-3 font-mono text-sm outline-none focus:border-accent"
          />
        </Field>
        <Field label="API key" help="Sent as an Authorization: Bearer header, only to the base URL above.">
          <div className="relative">
            <input
              value={draft.apiKey}
              onChange={(e) => setDraft((d) => ({ ...d, apiKey: e.target.value }))}
              type={showKey ? "text" : "password"}
              placeholder="sk-…"
              spellCheck={false}
              autoComplete="off"
              className="h-9 w-full rounded-lg border border-border bg-surface px-3 pr-9 font-mono text-sm outline-none focus:border-accent"
            />
            <button type="button" onClick={() => setShowKey((v) => !v)} aria-label={showKey ? "Hide API key" : "Show API key"} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted hover:text-fg">
              {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
            </button>
          </div>
        </Field>
        <Field label="Model name" help="Passed through as the request's “model” field.">
          <input
            value={draft.model}
            onChange={(e) => setDraft((d) => ({ ...d, model: e.target.value }))}
            placeholder="claude-sonnet-5"
            spellCheck={false}
            className="h-9 w-full rounded-lg border border-border bg-surface px-3 font-mono text-sm outline-none focus:border-accent"
          />
        </Field>

        <div className="flex items-center gap-2">
          <Button size="sm" onClick={run} disabled={test.state === "testing" || !draft.baseUrl.trim()}>
            {test.state === "testing" ? <Loader2 size={13} className="animate-spin" /> : null} Test connection
          </Button>
          {test.state === "ok" && <span className="flex items-center gap-1 text-xs text-ok"><CheckCircle2 size={14} /> {test.message}</span>}
          {test.state === "err" && <span className="flex items-center gap-1 text-xs text-danger"><XCircle size={14} /> {test.message}</span>}
        </div>

        <p className="text-[11px] leading-relaxed text-muted">
          Stored only in this browser (localStorage), never on the server. Changing the IP later — a new network, a rebuilt proxy — just means
          updating the base URL here.
        </p>

        <div className="flex justify-between border-t border-border pt-3">
          <Button variant="ghost" size="sm" onClick={() => setDraft(DEFAULT_LLM)}>Reset</Button>
          <div className="flex gap-2">
            <Button size="sm" onClick={onClose}>Cancel</Button>
            <Button variant="primary" size="sm" onClick={save} disabled={!dirty}>Save</Button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
