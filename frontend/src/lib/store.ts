"use client";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { makeBlock, newId } from "./blocks";
import { PRESETS } from "./presets";
import { DEFAULT_SAMPLE } from "./samples";
import { seedGroups, uniqueLabel, type BlockGroup, type GroupMembership } from "./transformerGroups";
import type { Block, BlockType, BuildProgress, DataInfo, DocInfo, EvalPoint, Hyper, InspectPayload, LearnChatMsg, LlmConfig, QaPair, QaProgress, SamplePayload, StatusPayload, StepPoint, SummaryPair, SummaryProgress, TokenizerInfo, TrainMode, TrainSource, UploadItem } from "./types";

/** Enough merges for frequent words to become single tokens on a small corpus. */
export const DEFAULT_MERGES = 200;

export const DEFAULT_HYPER: Hyper = {
  steps: 400, batchSize: 16, seqLen: 64, lr: 0.003, warmupSteps: 20, weightDecay: 0.01, gradClip: 1,
  seed: 42, probeEvery: 10, evalEvery: 25, sampleEvery: 50, sampleTokens: 120, temperature: 0.8,
};

export const DEFAULT_TOKENIZER: TokenizerInfo = { kind: "default", name: "", model: "", fullVocab: 0 };

export type Conn = "connecting" | "connected" | "disconnected";

/** Training curves are mutated in place (up to 20k points); `rev` tells charts to redraw. */
export interface Series {
  loss: number[];
  lr: number[];
  tps: number[];
  evals: EvalPoint[];
  rev: number;
}
const emptySeries = (): Series => ({ loss: [], lr: [], tps: [], evals: [], rev: 0 });

export interface Studio {
  conn: Conn;
  setConn: (c: Conn) => void;

  // ---- model
  blocks: Block[];
  /** Purely a canvas-arrangement concept, kept off `Block` itself so the Spec/backend payload
   * stays flat and untouched: which rectangular container (if any) each block currently renders
   * inside. A group's members are always kept contiguous in `blocks` by every action below. */
  groups: BlockGroup[];
  groupOf: GroupMembership;
  selectedId: string | null;
  hoverId: string | null;
  /** Replaces the whole stack (preset load, reset) and re-seeds groups from scratch by
   * pattern-detecting adjacent [Attention, Feed-Forward] pairs — a one-time convenience so
   * presets look organized immediately, not an ongoing source of truth. */
  setBlocks: (b: Block[]) => void;
  addBlock: (type: BlockType, index?: number) => void;
  /** Same as `addBlock`, but for drops resolved to a container: `targetGroupId` (null = top level)
   * and `before` (id to land immediately before; null = append to that scope's end). */
  addBlockTo: (type: BlockType, targetGroupId: string | null, before: string | null) => void;
  /** Inserts one fresh Self-Attention + Feed-Forward pair, already wrapped in a new Transformer
   * Block container — the pre-norm/residual math lives inside each block's own settings, so the
   * container is purely visual (see Attn::forward's `x + out`, not `Norm(x) + out`). */
  addTransformerBlock: (index?: number) => void;
  removeBlock: (id: string) => void;
  /** Removes several blocks at once (by id) in a single update — used for whole-group removal,
   * where removing one at a time would shift the others' indices mid-operation. */
  removeBlocks: (ids: string[]) => void;
  duplicateBlock: (id: string) => void;
  /** Moves one block to a new spot: `targetGroupId` is the container it should end up inside
   * (null = top level), `before` is the id of the block it should land immediately before (null =
   * append to the end of that scope). The sole primitive behind all drag-and-drop reordering,
   * regrouping, and ungrouping. */
  moveBlockTo: (blockId: string, targetGroupId: string | null, before: string | null) => void;
  /** Reorders a whole group (all its members, staying contiguous) among the top-level items,
   * inserting it immediately before `beforeAnchorBlockId` (or at the end if null/not found). */
  moveGroupTo: (groupId: string, beforeAnchorBlockId: string | null) => void;
  /** Clones every block in a group (fresh ids) into a brand new group inserted right after it. */
  duplicateGroup: (groupId: string) => void;
  /** Removes a group's container AND every block inside it. */
  removeGroup: (groupId: string) => void;
  /** Dissolves a group's container, leaving its blocks in place at the top level. */
  ungroup: (groupId: string) => void;
  renameGroup: (groupId: string, label: string) => void;
  updateBlock: (id: string, patch: Record<string, unknown>) => void;
  select: (id: string | null) => void;
  hover: (id: string | null) => void;

  // ---- data
  text: string;
  merges: number;
  probe: string;
  hyper: Hyper;
  usePasted: boolean;
  dataInfo: DataInfo | null;
  /** Which tokenizer the backend builds the dataset with (pushed by the server, not persisted). */
  tokenizer: TokenizerInfo;
  /** Set while the backend samples / tokenizes; null when idle. */
  progress: BuildProgress | null;
  /** Why the dataset is unusable (no text, all files failed, …), if it is. */
  dataError: string | null;
  setDataError: (e: string | null) => void;
  dataRev: number;
  docs: DocInfo[];
  uploads: UploadItem[];
  setUsePasted: (b: boolean) => void;
  setDocs: (d: DocInfo[]) => void;
  setProgress: (p: BuildProgress | null) => void;
  addUploads: (u: UploadItem[]) => void;
  patchUpload: (id: string, p: Partial<UploadItem>) => void;
  dropUpload: (id: string) => void;
  setText: (t: string, probe?: string) => void;
  setMerges: (n: number) => void;
  setProbe: (p: string) => void;
  setHyper: (p: Partial<Hyper>) => void;
  setDataInfo: (i: DataInfo | null) => void;
  setTokenizer: (t: TokenizerInfo) => void;

  // ---- Q&A dataset (generated by the model server from uploaded documents)
  qa: QaPair[];
  qaProgress: QaProgress | null;
  setQa: (p: QaPair[]) => void;
  addQa: (p: QaPair[]) => void;
  setQaProgress: (p: QaProgress | null) => void;
  // ---- Summary dataset (same idea, one summary per page instead of several Q&A pairs)
  summaries: SummaryPair[];
  summaryProgress: SummaryProgress | null;
  setSummaries: (p: SummaryPair[]) => void;
  addSummaries: (p: SummaryPair[]) => void;
  setSummaryProgress: (p: SummaryProgress | null) => void;
  /** Train on the text (Data tab), the Q&A pairs, or the summaries; and from scratch or on top of the last run. */
  trainSource: TrainSource;
  trainMode: TrainMode;
  setTrainSource: (s: TrainSource) => void;
  setTrainMode: (m: TrainMode) => void;

  // ---- observation
  inspect: InspectPayload | null;
  inspectBusy: boolean;
  setInspect: (p: InspectPayload | null) => void;
  setInspectBusy: (b: boolean) => void;

  // ---- training
  status: StatusPayload;
  series: Series;
  samples: SamplePayload[];
  gradSnap: InspectPayload | null;
  setGradSnap: (p: InspectPayload | null) => void;
  setStatus: (s: StatusPayload) => void;
  pushSteps: (p: StepPoint[]) => void;
  pushEval: (e: EvalPoint) => void;
  pushSample: (s: SamplePayload) => void;
  resetRun: () => void;

  // ---- generation
  gen: { running: boolean; text: string; error: string | null };
  setGen: (g: Partial<Studio["gen"]>) => void;
  appendGen: (t: string) => void;

  // ---- ui
  splitPct: number;
  setSplit: (n: number) => void;
  /** Which top-level tab the right column shows. */
  rightTab: "observe" | "code" | "learn";
  setRightTab: (t: Studio["rightTab"]) => void;
  /** Block whose concepts the Learn tab is showing. */
  learnBlockId: string | null;
  /** Switches the right column to the Learn tab for this block (and, on phones, brings that column into view). */
  openLearn: (blockId: string) => void;
  /** Whether the Live code tab shows the whole model (read-only, generated) or one block's editable Playground. */
  codeMode: "model" | "block";
  setCodeMode: (m: Studio["codeMode"]) => void;
  /** Block the Playground is showing. */
  codeBlockId: string | null;
  /** Switches the right column to the Live code tab, in Playground mode, for this block. */
  openBlockCode: (blockId: string) => void;
  /** User's edits to a block's Playground snippet, keyed by block id; absent = still the starter code. */
  playgroundCode: Record<string, string>;
  setPlaygroundCode: (blockId: string, code: string) => void;
  resetPlaygroundCode: (blockId: string) => void;
  /** On phones, Build and the right column (Observe/Learn/Live code) are separate screens. */
  mobilePane: "build" | "right";
  setMobilePane: (p: Studio["mobilePane"]) => void;

  // ---- Learn chat
  llm: LlmConfig;
  setLlm: (p: Partial<LlmConfig>) => void;
  learnChat: LearnChatMsg[];
  learnChatBusy: boolean;
  pushLearnMsg: (m: LearnChatMsg) => void;
  updateLearnMsg: (id: string, patch: Partial<LearnChatMsg>) => void;
  clearLearnChat: () => void;
  setLearnChatBusy: (b: boolean) => void;
  settingsOpen: boolean;
  setSettingsOpen: (b: boolean) => void;
}

export const DEFAULT_LLM: LlmConfig = { baseUrl: "", apiKey: "", model: "claude-sonnet-5" };

/** The subset of state kept in localStorage. */
type Persisted = Pick<Studio, "blocks" | "groups" | "groupOf" | "merges" | "probe" | "hyper" | "splitPct" | "usePasted" | "llm"> & { text: string | undefined };

const initialBlocks = () => PRESETS[0].build();

/** Where a new block may go: embedding first, head last, everything else in between. */
function clampInsert(blocks: Block[], type: BlockType, index?: number): number {
  const n = blocks.length;
  if (type === "embedding") return 0;
  if (type === "head") return n;
  const lo = blocks[0]?.type === "embedding" ? 1 : 0;
  const hi = blocks[n - 1]?.type === "head" ? n - 1 : n;
  return Math.min(hi, Math.max(lo, index ?? n));
}

const seedInitial = initialBlocks();
const { groups: initialGroups, groupOf: initialGroupOf } = seedGroups(seedInitial);

export const useStudio = create<Studio>()(
  persist(
    (set) => ({
      conn: "connecting",
      setConn: (conn) => set({ conn }),

      blocks: seedInitial,
      groups: initialGroups,
      groupOf: initialGroupOf,
      selectedId: null,
      hoverId: null,
      setBlocks: (blocks) => set(() => ({ blocks, ...seedGroups(blocks), selectedId: null })),
      addBlock: (type, index) =>
        set((s) => {
          const b = makeBlock(type);
          const blocks = s.blocks.slice();
          const at = clampInsert(blocks, type, index);
          blocks.splice(at, 0, b);
          return { blocks, selectedId: b.id };
        }),
      addBlockTo: (type, targetGroupId, before) =>
        set((s) => {
          const gid = type === "embedding" || type === "head" ? null : targetGroupId;
          const b = makeBlock(type);
          const blocks = s.blocks.slice();
          let at: number;
          if (before) {
            at = blocks.findIndex((x) => x.id === before);
            if (at < 0) at = blocks.length;
          } else if (gid) {
            let last = -1;
            for (let k = 0; k < blocks.length; k++) if (s.groupOf[blocks[k].id] === gid) last = k;
            at = last >= 0 ? last + 1 : blocks.length;
          } else {
            at = blocks.length;
          }
          at = clampInsert(blocks, type, at);
          blocks.splice(at, 0, b);
          const groupOf = gid ? { ...s.groupOf, [b.id]: gid } : s.groupOf;
          return { blocks, groupOf, selectedId: b.id };
        }),
      addTransformerBlock: (index) =>
        set((s) => {
          const blocks = s.blocks.slice();
          const at = clampInsert(blocks, "attention", index);
          const d = blocks.find((b) => b.type === "embedding");
          const heads = d && d.type === "embedding" && d.d_model % 4 === 0 ? 4 : 1;
          // pre_norm+residual live inside each block, not as separate cards: a standalone Norm
          // card in between would apply BEFORE this block sees x, so its own residual add would
          // skip around only itself, adding back the *normalized* value instead of the true
          // pre-norm x — wrong math (verified against Attn::forward's `x + out`, not `h + out`).
          // The container these two land in is purely a visual grouping on top of that.
          const attn = { ...makeBlock("attention"), n_heads: heads, n_kv_heads: heads } as Block;
          const ffn = makeBlock("ffn");
          blocks.splice(at, 0, attn, ffn);
          const gid = newId();
          const groupOf = { ...s.groupOf, [attn.id]: gid, [ffn.id]: gid };
          const groups = [...s.groups, { id: gid, label: uniqueLabel(`Transformer Block ${s.groups.length + 1}`, s.groups) }];
          return { blocks, groupOf, groups, selectedId: attn.id };
        }),
      removeBlock: (id) =>
        set((s) => {
          const blocks = s.blocks.filter((b) => b.id !== id);
          const groupOf = { ...s.groupOf };
          const gid = groupOf[id];
          delete groupOf[id];
          const groups = gid && !blocks.some((b) => groupOf[b.id] === gid) ? s.groups.filter((g) => g.id !== gid) : s.groups;
          return { blocks, groupOf, groups, selectedId: s.selectedId === id ? null : s.selectedId };
        }),
      removeBlocks: (ids) =>
        set((s) => {
          const gone = new Set(ids);
          const blocks = s.blocks.filter((b) => !gone.has(b.id));
          const groupOf = { ...s.groupOf };
          const touched = new Set<string>();
          for (const id of ids) {
            const gid = groupOf[id];
            if (gid) touched.add(gid);
            delete groupOf[id];
          }
          const groups = s.groups.filter((g) => !touched.has(g.id) || blocks.some((b) => groupOf[b.id] === g.id));
          return { blocks, groupOf, groups, selectedId: s.selectedId && gone.has(s.selectedId) ? null : s.selectedId };
        }),
      duplicateBlock: (id) =>
        set((s) => {
          const i = s.blocks.findIndex((b) => b.id === id);
          if (i < 0 || s.blocks[i].type === "embedding" || s.blocks[i].type === "head") return s;
          const copy = { ...s.blocks[i], id: newId() } as Block;
          const blocks = s.blocks.slice();
          blocks.splice(i + 1, 0, copy);
          const gid = s.groupOf[id];
          const groupOf = gid ? { ...s.groupOf, [copy.id]: gid } : s.groupOf;
          return { blocks, groupOf, selectedId: copy.id };
        }),
      moveBlockTo: (blockId, targetGroupId, before) =>
        set((s) => {
          if (before === blockId) return s; // "insert before myself" is a no-op, not "move to the end"
          const i = s.blocks.findIndex((b) => b.id === blockId);
          if (i < 0) return s;
          const moving = s.blocks[i];
          // embedding/head are structural singletons (always first/last) — never groupable
          if (targetGroupId && (moving.type === "embedding" || moving.type === "head")) return s;
          const blocks = s.blocks.slice();
          blocks.splice(i, 1);
          let at: number;
          if (before) {
            at = blocks.findIndex((b) => b.id === before);
            if (at < 0) at = blocks.length;
          } else if (targetGroupId) {
            let last = -1;
            for (let k = 0; k < blocks.length; k++) if (s.groupOf[blocks[k].id] === targetGroupId) last = k;
            at = last >= 0 ? last + 1 : blocks.length;
          } else {
            at = blocks.length;
          }
          at = clampInsert(blocks, moving.type, at);
          blocks.splice(at, 0, moving);
          const groupOf = { ...s.groupOf };
          if (targetGroupId) groupOf[blockId] = targetGroupId;
          else delete groupOf[blockId];
          const oldGroupId = s.groupOf[blockId];
          let groups = s.groups;
          if (oldGroupId && oldGroupId !== targetGroupId && !blocks.some((b) => groupOf[b.id] === oldGroupId)) {
            groups = s.groups.filter((g) => g.id !== oldGroupId);
          }
          return { blocks, groupOf, groups, selectedId: blockId };
        }),
      moveGroupTo: (groupId, beforeAnchorBlockId) =>
        set((s) => {
          const memberIds = new Set(s.blocks.filter((b) => s.groupOf[b.id] === groupId).map((b) => b.id));
          if (memberIds.size === 0) return s;
          const members = s.blocks.filter((b) => memberIds.has(b.id));
          const rest = s.blocks.filter((b) => !memberIds.has(b.id));
          let at = beforeAnchorBlockId ? rest.findIndex((b) => b.id === beforeAnchorBlockId) : -1;
          if (at < 0) at = rest.length;
          const lo = rest[0]?.type === "embedding" ? 1 : 0;
          const hi = rest[rest.length - 1]?.type === "head" ? rest.length - 1 : rest.length;
          at = Math.min(hi, Math.max(lo, at));
          rest.splice(at, 0, ...members);
          return { blocks: rest, selectedId: members[0].id };
        }),
      duplicateGroup: (groupId) =>
        set((s) => {
          const idxs = s.blocks.map((b, i) => (s.groupOf[b.id] === groupId ? i : -1)).filter((i) => i >= 0);
          if (idxs.length === 0) return s;
          const end = idxs[idxs.length - 1];
          const members = idxs.map((i) => s.blocks[i]);
          const copies = members.map((b) => ({ ...b, id: newId() }) as Block);
          const blocks = s.blocks.slice();
          blocks.splice(end + 1, 0, ...copies);
          const newGid = newId();
          const groupOf = { ...s.groupOf };
          for (const c of copies) groupOf[c.id] = newGid;
          const original = s.groups.find((g) => g.id === groupId);
          const groups = [...s.groups, { id: newGid, label: uniqueLabel(original?.label ?? "Transformer Block", s.groups) }];
          return { blocks, groupOf, groups, selectedId: copies[0].id };
        }),
      removeGroup: (groupId) =>
        set((s) => {
          const gone = new Set(s.blocks.filter((b) => s.groupOf[b.id] === groupId).map((b) => b.id));
          const blocks = s.blocks.filter((b) => !gone.has(b.id));
          const groupOf = { ...s.groupOf };
          for (const id of gone) delete groupOf[id];
          const groups = s.groups.filter((g) => g.id !== groupId);
          return { blocks, groupOf, groups, selectedId: s.selectedId && gone.has(s.selectedId) ? null : s.selectedId };
        }),
      ungroup: (groupId) =>
        set((s) => {
          const groupOf = { ...s.groupOf };
          for (const b of s.blocks) if (groupOf[b.id] === groupId) delete groupOf[b.id];
          const groups = s.groups.filter((g) => g.id !== groupId);
          return { groupOf, groups };
        }),
      renameGroup: (groupId, label) => set((s) => ({ groups: s.groups.map((g) => (g.id === groupId ? { ...g, label } : g)) })),
      updateBlock: (id, patch) => set((s) => ({ blocks: s.blocks.map((b) => (b.id === id ? ({ ...b, ...patch } as Block) : b)) })),
      select: (selectedId) => set({ selectedId }),
      hover: (hoverId) => set({ hoverId }),

      text: DEFAULT_SAMPLE.text,
      merges: DEFAULT_MERGES,
      probe: DEFAULT_SAMPLE.probe,
      hyper: DEFAULT_HYPER,
      usePasted: true,
      dataInfo: null,
      tokenizer: DEFAULT_TOKENIZER,
      progress: null,
      dataError: null,
      setDataError: (dataError) => set({ dataError }),
      dataRev: 0,
      docs: [],
      uploads: [],
      setUsePasted: (usePasted) => set({ usePasted }),
      setDocs: (docs) => set({ docs }),
      setProgress: (progress) => set({ progress }),
      addUploads: (u) => set((s) => ({ uploads: [...s.uploads, ...u] })),
      patchUpload: (id, p) => set((s) => ({ uploads: s.uploads.map((x) => (x.id === id ? { ...x, ...p } : x)) })),
      dropUpload: (id) => set((s) => ({ uploads: s.uploads.filter((x) => x.id !== id) })),
      setText: (text, probe) => set((s) => ({ text, probe: probe ?? s.probe })),
      setMerges: (merges) => set({ merges }),
      setProbe: (probe) => set({ probe }),
      setHyper: (p) => set((s) => ({ hyper: { ...s.hyper, ...p } })),
      setTokenizer: (tokenizer) => set({ tokenizer }),
      setDataInfo: (dataInfo) => set((s) => ({ dataInfo, dataRev: s.dataRev + 1 })),

      qa: [],
      qaProgress: null,
      setQa: (qa) => set({ qa }),
      addQa: (p) => set((s) => ({ qa: [...s.qa, ...p] })),
      setQaProgress: (qaProgress) => set({ qaProgress }),
      summaries: [],
      summaryProgress: null,
      setSummaries: (summaries) => set({ summaries }),
      addSummaries: (p) => set((s) => ({ summaries: [...s.summaries, ...p] })),
      setSummaryProgress: (summaryProgress) => set({ summaryProgress }),
      trainSource: "text",
      trainMode: "new",
      setTrainSource: (trainSource) => set({ trainSource }),
      setTrainMode: (trainMode) => set({ trainMode }),

      inspect: null,
      inspectBusy: false,
      setInspect: (inspect) => set({ inspect }),
      setInspectBusy: (inspectBusy) => set({ inspectBusy }),

      status: { state: "idle", message: "", step: 0, total: 0 },
      series: emptySeries(),
      samples: [],
      gradSnap: null,
      setGradSnap: (gradSnap) => set({ gradSnap }),
      setStatus: (status) => set({ status }),
      pushSteps: (pts) =>
        set((s) => {
          const ser = s.series;
          for (const p of pts) {
            ser.loss[p.step - 1] = p.loss;
            ser.lr[p.step - 1] = p.lr;
            ser.tps[p.step - 1] = p.tokPerSec;
          }
          ser.rev++;
          return { series: ser };
        }),
      pushEval: (e) =>
        set((s) => {
          s.series.evals.push(e);
          s.series.rev++;
          return { series: s.series };
        }),
      pushSample: (sm) => set((s) => ({ samples: [sm, ...s.samples].slice(0, 12) })),
      resetRun: () => set({ series: emptySeries(), samples: [], gradSnap: null, status: { state: "preparing", message: "Starting…", step: 0, total: 0 } }),

      gen: { running: false, text: "", error: null },
      setGen: (g) => set((s) => ({ gen: { ...s.gen, ...g } })),
      appendGen: (t) => set((s) => ({ gen: { ...s.gen, text: s.gen.text + t } })),

      splitPct: 48,
      setSplit: (splitPct) => set({ splitPct: Math.min(75, Math.max(25, splitPct)) }),

      rightTab: "observe",
      setRightTab: (rightTab) => set({ rightTab }),
      learnBlockId: null,
      openLearn: (learnBlockId) => set({ rightTab: "learn", learnBlockId, mobilePane: "right" }),
      codeMode: "model",
      setCodeMode: (codeMode) => set({ codeMode }),
      codeBlockId: null,
      openBlockCode: (codeBlockId) => set({ rightTab: "code", codeMode: "block", codeBlockId, mobilePane: "right" }),
      playgroundCode: {},
      setPlaygroundCode: (blockId, code) => set((s) => ({ playgroundCode: { ...s.playgroundCode, [blockId]: code } })),
      resetPlaygroundCode: (blockId) =>
        set((s) => {
          const next = { ...s.playgroundCode };
          delete next[blockId];
          return { playgroundCode: next };
        }),
      mobilePane: "build",
      setMobilePane: (mobilePane) => set({ mobilePane }),

      llm: DEFAULT_LLM,
      setLlm: (p) => set((s) => ({ llm: { ...s.llm, ...p } })),
      learnChat: [],
      learnChatBusy: false,
      pushLearnMsg: (m) => set((s) => ({ learnChat: [...s.learnChat, m] })),
      updateLearnMsg: (id, patch) => set((s) => ({ learnChat: s.learnChat.map((m) => (m.id === id ? { ...m, ...patch } : m)) })),
      clearLearnChat: () => set({ learnChat: [] }),
      setLearnChatBusy: (learnChatBusy) => set({ learnChatBusy }),
      settingsOpen: false,
      setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
    }),
    {
      name: "mb.studio.v1",
      version: 3,
      migrate: (persisted, from) => {
        let p = persisted as Persisted;
        // v1 defaulted to 60 merges, which leaves most words as single letters
        if (from < 2 && p.merges === 60) p = { ...p, merges: DEFAULT_MERGES };
        // v2 predates group containers — backfill by pattern-detecting existing [Attention, Ffn] pairs
        if (from < 3 && !p.groups) p = { ...p, ...seedGroups(p.blocks ?? []) };
        return p;
      },
      partialize: (s) => ({
        blocks: s.blocks,
        groups: s.groups,
        groupOf: s.groupOf,
        merges: s.merges,
        usePasted: s.usePasted,
        llm: s.llm,
        probe: s.probe,
        hyper: s.hyper,
        splitPct: s.splitPct,
        // big uploads would blow the ~5 MB localStorage quota
        text: s.text.length < 400_000 ? s.text : undefined,
      }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<Studio>;
        return {
          ...current,
          ...p,
          hyper: { ...current.hyper, ...(p.hyper ?? {}) },
          llm: { ...current.llm, ...(p.llm ?? {}) },
          text: p.text ?? current.text,
          groups: p.groups ?? current.groups,
          groupOf: p.groupOf ?? current.groupOf,
        };
      },
    },
  ),
);

export const isRunning = (s: StatusPayload) => s.state === "preparing" || s.state === "running";
