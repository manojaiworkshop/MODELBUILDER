"use client";
import { AlertTriangle, RotateCcw, Send, Sparkles, Trash2 } from "lucide-react";
import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import { askLearn, retryLearn } from "@/lib/sync";
import { blockLabel } from "@/lib/blocks";
import { useStudio } from "@/lib/store";
import type { LearnChatMsg } from "@/lib/types";
import { cn } from "../ui/kit";
import { DiagramCanvas } from "./DiagramCanvas";
import { Formula } from "./pieces";
import { LlmSettingsButton } from "./LlmSettings";

const SUGGESTIONS = ["Why does attention need multiple heads?", "What's the difference between GQA and MQA?", "Why RMSNorm instead of LayerNorm?"];

/** `**bold**` and `` `code` `` only — matches exactly what the system prompt asks the model to use. */
function renderInline(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return parts.map((p, i) => {
    if (p.startsWith("**") && p.endsWith("**")) return <strong key={i}>{p.slice(2, -2)}</strong>;
    if (p.startsWith("`") && p.endsWith("`")) return <code key={i} className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12px]">{p.slice(1, -1)}</code>;
    return <Fragment key={i}>{p}</Fragment>;
  });
}

function Prose({ text }: { text: string }) {
  const paragraphs = text.split(/\n{2,}/).filter((p) => p.trim());
  return (
    <div className="space-y-2 text-[13px] leading-relaxed">
      {paragraphs.map((p, i) => (
        <p key={i} className="whitespace-pre-wrap">{renderInline(p)}</p>
      ))}
    </div>
  );
}

function AssistantBubble({ msg }: { msg: LearnChatMsg }) {
  if (msg.pending) {
    return (
      <div className="flex items-center gap-1.5 rounded-xl rounded-bl-sm border border-border bg-surface-2/60 px-4 py-3">
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.3s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.15s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted" />
      </div>
    );
  }
  if (msg.error) {
    return (
      <div className="space-y-2 rounded-xl rounded-bl-sm border border-danger/30 bg-danger/5 px-4 py-3">
        <div className="flex items-start gap-2 text-[13px] text-danger"><AlertTriangle size={15} className="mt-0.5 shrink-0" /> {msg.error}</div>
        <button onClick={() => retryLearn(msg.id)} className="flex items-center gap-1.5 text-xs font-medium text-accent hover:underline">
          <RotateCcw size={12} /> Retry
        </button>
      </div>
    );
  }
  const a = msg.answer;
  if (!a) return null;
  return (
    <div className="max-w-full space-y-3 rounded-xl rounded-bl-sm border border-border bg-surface-2/60 px-4 py-3">
      <Prose text={a.text} />
      {a.formulas.length > 0 && (
        <div className="space-y-2">
          {a.formulas.map((f, i) => <Formula key={i} label={f.label ?? undefined}>{f.expr}</Formula>)}
        </div>
      )}
      {a.diagram && a.diagram.nodes.length > 0 && <DiagramCanvas spec={a.diagram} />}
      {a.followUps.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pt-0.5">
          {a.followUps.map((q, i) => (
            <button key={i} onClick={() => askLearn(q)} className="rounded-full border border-accent/30 bg-accent-soft px-2.5 py-1 text-[11px] font-medium text-accent hover:bg-accent/20">
              {q}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function LearnChat() {
  const messages = useStudio((s) => s.learnChat);
  const busy = useStudio((s) => s.learnChatBusy);
  const clear = useStudio((s) => s.clearLearnChat);
  const blocks = useStudio((s) => s.blocks);
  const learnBlockId = useStudio((s) => s.learnBlockId);
  const llmConfigured = useStudio((s) => !!s.llm.baseUrl.trim() && !!s.llm.model.trim());
  const [input, setInput] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const activeIdx = blocks.findIndex((b) => b.id === learnBlockId);
  const activeLabel = activeIdx >= 0 ? blockLabel(blocks, activeIdx) : null;

  const send = () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    askLearn(q);
  };

  return (
    <div className="flex shrink-0 flex-col border-t border-border bg-surface">
      <div className="flex items-center gap-2 px-3 py-2">
        <Sparkles size={14} className="text-accent" />
        <span className="text-xs font-semibold">Ask about {activeLabel ? <span className="text-accent">{activeLabel}</span> : "the model"}</span>
        <div className="flex-1" />
        {messages.length > 0 && (
          <button onClick={clear} aria-label="Clear conversation" title="Clear conversation" className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-danger">
            <Trash2 size={14} />
          </button>
        )}
        <LlmSettingsButton />
      </div>

      {messages.length > 0 && (
        <div ref={listRef} className="max-h-72 space-y-3 overflow-y-auto border-t border-border px-3 py-3">
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="flex justify-end">
                <div className="max-w-[85%] rounded-xl rounded-br-sm bg-accent px-4 py-2 text-sm text-accent-fg">{m.question}</div>
              </div>
            ) : (
              <div key={m.id} className="flex justify-start"><div className="w-full"><AssistantBubble msg={m} /></div></div>
            ),
          )}
        </div>
      )}

      {messages.length === 0 && (
        <div className="flex flex-wrap gap-1.5 border-t border-border px-3 py-2">
          {SUGGESTIONS.map((q) => (
            <button key={q} onClick={() => askLearn(q)} className="rounded-full border border-border bg-surface-2 px-2.5 py-1 text-[11px] text-muted hover:border-accent hover:text-accent">
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="border-t border-border p-2.5">
        {!llmConfigured && (
          <p className="mb-2 text-[11px] text-warn">
            Not connected — click the gear <span aria-hidden>⚙</span> to set the model server before asking.
          </p>
        )}
        <div className="flex items-end gap-2">
          <textarea
            ref={taRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            rows={1}
            placeholder="Ask anything about this model… (Enter to send)"
            className={cn("max-h-28 flex-1 resize-none rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent")}
          />
          <button
            onClick={send}
            disabled={!input.trim() || busy}
            aria-label="Send"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-fg disabled:opacity-40"
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
