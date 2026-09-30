"use client";
import { Check, Code2, Copy } from "lucide-react";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { generate, type Lang } from "@/lib/codegen";
import { useStudio } from "@/lib/store";
import { BlockPlayground } from "../info/BlockPlayground";
import { Segmented, cn } from "../ui/kit";

const KEYWORDS: Record<Lang, Set<string>> = {
  python: new Set(["import", "class", "def", "self", "return", "super", "None", "True", "False", "for", "in", "if", "else", "as", "from"]),
  rust: new Set(["pub", "struct", "impl", "fn", "let", "self", "Self", "mut", "use", "for", "in", "if", "else", "true", "false", "usize", "f32", "Int"]),
  json: new Set(["true", "false", "null"]),
};

const TOKEN = /(#.*$|\/\/.*$)|("(?:[^"\\]|\\.)*")|(\b\d[\d_]*(?:\.\d+)?(?:f32)?\b)|(\b[A-Za-z_][A-Za-z0-9_]*\b)/g;

function Highlight({ text, lang }: { text: string; lang: Lang }) {
  const out: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const [tok, comment, str, num, ident] = m;
    if (comment && lang !== "json") out.push(<span key={m.index} className="text-slate-500 italic">{tok}</span>);
    else if (str) out.push(<span key={m.index} className={lang === "json" && text.slice(m.index + tok.length).trimStart().startsWith(":") ? "text-sky-300" : "text-emerald-300"}>{tok}</span>);
    else if (num) out.push(<span key={m.index} className="text-amber-300">{tok}</span>);
    else if (ident && KEYWORDS[lang].has(ident)) out.push(<span key={m.index} className="text-violet-300">{tok}</span>);
    else if (ident && /^[A-Z]/.test(ident)) out.push(<span key={m.index} className="text-cyan-300">{tok}</span>);
    else out.push(tok);
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return <>{out.map((n, i) => <Fragment key={i}>{n}</Fragment>)}</>;
}

export function CodePanel() {
  const blocks = useStudio((s) => s.blocks);
  const groupOf = useStudio((s) => s.groupOf);
  const groups = useStudio((s) => s.groups);
  const selectedId = useStudio((s) => s.selectedId);
  const hoverId = useStudio((s) => s.hoverId);
  const [lang, setLang] = useState<Lang>("python");
  const [copied, setCopied] = useState(false);
  const lines = useMemo(() => generate(lang, blocks, groupOf, groups), [lang, blocks, groupOf, groups]);
  const focus = hoverId ?? selectedId;
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!selectedId) return;
    ref.current?.querySelector<HTMLElement>(`[data-block="${selectedId}"]`)?.scrollIntoView({ block: "nearest" });
  }, [selectedId, lang]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(lines.map((l) => l.text).join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {}
  };

  return (
    <>
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2.5">
        <span className="hidden text-xs text-muted md:inline">{blocks.length} block{blocks.length === 1 ? "" : "s"} · {lines.length} lines</span>
        <div className="flex-1" />
        <Segmented<Lang>
          value={lang}
          onChange={setLang}
          options={[
            { value: "python", label: "Python" },
            { value: "rust", label: "Rust" },
            { value: "json", label: "Spec", title: "The exact JSON the backend receives" },
          ]}
        />
        <button onClick={copy} aria-label="Copy code" className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-fg">
          {copied ? <Check size={15} className="text-ok" /> : <Copy size={15} />}
        </button>
      </div>
      <div ref={ref} className="min-h-0 flex-1 overflow-auto bg-code font-mono text-[12px] leading-[1.65] text-slate-200">
        {blocks.length === 0 ? (
          <div className="flex h-full items-center justify-center p-8 text-center text-sm text-slate-500">
            Add blocks on the Model tab and the equivalent code appears here.
          </div>
        ) : (
          <div className="min-w-max py-2">
            {lines.map((l, i) => (
              <div
                key={i}
                data-block={l.blockId}
                className={cn("flex px-3 transition-colors", l.blockId && l.blockId === focus ? "bg-indigo-400/20 shadow-[inset_2px_0_0_0_#818cf8]" : "")}
              >
                <span className="mr-4 w-6 shrink-0 select-none text-right text-slate-600">{i + 1}</span>
                <span className="whitespace-pre"><Highlight text={l.text} lang={lang} /></span>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

type Mode = "model" | "block";

/** Owns the "Live code" tab: a mode toggle between the whole generated model (read-only) and one
 * block's editable, runnable Playground. */
export function LiveCodeTab() {
  const mode = useStudio((s) => s.codeMode);
  const setMode = useStudio((s) => s.setCodeMode);
  return (
    <div className="flex h-full min-h-0 flex-col bg-surface">
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2.5">
        <Code2 size={15} className="text-accent" />
        <h2 className="text-sm font-semibold">Live code</h2>
        <div className="flex-1" />
        <Segmented<Mode>
          value={mode}
          onChange={setMode}
          options={[
            { value: "model", label: "Whole model" },
            { value: "block", label: "Try a block", title: "Edit and run a tiny snippet for one block" },
          ]}
        />
      </div>
      {mode === "model" ? <CodePanel /> : <BlockPlayground />}
    </div>
  );
}
