"use client";
import { BookOpen, MousePointerClick } from "lucide-react";
import { blockLabel, CATALOG } from "@/lib/blocks";
import { useStudio } from "@/lib/store";
import { BlockChips } from "./BlockChips";
import { BlockLearn, type LearnCtx } from "./content";
import { LearnChat } from "./LearnChat";

export function InfoPanel() {
  const blocks = useStudio((s) => s.blocks);
  const learnBlockId = useStudio((s) => s.learnBlockId);
  const openLearn = useStudio((s) => s.openLearn);
  const dModel = useStudio((s) => s.blocks.find((b) => b.type === "embedding")?.d_model ?? 64);
  const vocab = useStudio((s) => s.inspect?.vocabSize ?? s.dataInfo?.vocabSize ?? null);
  const seqLen = useStudio((s) => s.hyper.seqLen);
  const active = blocks.find((b) => b.id === learnBlockId) ?? blocks[0] ?? null;
  const ctx: LearnCtx = { dModel, vocab, seqLen };

  return (
    <div className="flex h-full min-h-0 flex-col bg-bg">
      <div className="flex shrink-0 items-center gap-2 border-b border-border bg-surface px-4 py-2.5">
        <BookOpen size={16} className="text-accent" />
        <h2 className="text-sm font-semibold">Learn</h2>
        <span className="hidden text-xs text-muted md:inline">the concepts and formulas behind each block</span>
      </div>
      {blocks.length === 0 ? (
        <>
          <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center text-sm text-muted">
            <MousePointerClick size={22} />
            <p>Add blocks on the Model tab, then click their <span className="mx-0.5 inline-flex h-4 w-4 items-center justify-center rounded-full border border-border-strong text-[10px]">i</span> button — or ask a general question below.</p>
          </div>
          <LearnChat />
        </>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="sticky top-0 z-10 space-y-1 border-b border-border bg-surface px-3 py-2.5">
            <div className="px-1 text-[11px] font-medium text-muted">Jump to a block</div>
            <BlockChips activeId={learnBlockId} onSelect={openLearn} />
          </div>
          {active && (
            <div className="space-y-5 p-4 pb-8">
              <div className="flex items-center gap-2.5">
                <span
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
                  style={{ background: `color-mix(in srgb, ${CATALOG[active.type].color} 16%, transparent)`, color: CATALOG[active.type].color }}
                >
                  {(() => {
                    const Icon = CATALOG[active.type].icon;
                    return <Icon size={18} />;
                  })()}
                </span>
                <div>
                  <h3 className="text-base font-semibold leading-tight">{blockLabel(blocks, blocks.findIndex((b) => b.id === active.id))}</h3>
                  <p className="text-xs text-muted">{CATALOG[active.type].summary(active)}</p>
                </div>
              </div>
              <BlockLearn block={active} ctx={ctx} />
            </div>
          )}
        </div>
      )}
      {blocks.length > 0 && <LearnChat />}
    </div>
  );
}
