"use client";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { makeBlock } from "./blocks";
import { PRESETS } from "./presets";
import { DEFAULT_SAMPLE } from "./samples";
import type { Block, BlockType, BuildProgress, DataInfo, DocInfo, EvalPoint, Hyper, InspectPayload, SamplePayload, StatusPayload, StepPoint, UploadItem } from "./types";

/** Enough merges for frequent words to become single tokens on a small corpus. */
export const DEFAULT_MERGES = 200;

export const DEFAULT_HYPER: Hyper = {
  steps: 400, batchSize: 16, seqLen: 64, lr: 0.003, warmupSteps: 20, weightDecay: 0.01, gradClip: 1,
  seed: 42, probeEvery: 10, evalEvery: 25, sampleEvery: 50, sampleTokens: 120, temperature: 0.8,
};

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
  selectedId: string | null;
  hoverId: string | null;
  setBlocks: (b: Block[]) => void;
  addBlock: (type: BlockType, index?: number) => void;
  addLayer: (index?: number) => void;
  removeBlock: (id: string) => void;
  duplicateBlock: (id: string) => void;
  moveBlock: (from: number, to: number) => void;
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
}

/** The subset of state kept in localStorage. */
type Persisted = Pick<Studio, "blocks" | "merges" | "probe" | "hyper" | "splitPct" | "usePasted"> & { text: string | undefined };

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

export const useStudio = create<Studio>()(
  persist(
    (set) => ({
      conn: "connecting",
      setConn: (conn) => set({ conn }),

      blocks: initialBlocks(),
      selectedId: null,
      hoverId: null,
      setBlocks: (blocks) => set({ blocks, selectedId: null }),
      addBlock: (type, index) =>
        set((s) => {
          const b = makeBlock(type);
          const blocks = s.blocks.slice();
          const at = clampInsert(blocks, type, index);
          blocks.splice(at, 0, b);
          return { blocks, selectedId: b.id };
        }),
      addLayer: (index) =>
        set((s) => {
          const blocks = s.blocks.slice();
          const at = clampInsert(blocks, "attention", index);
          const d = blocks.find((b) => b.type === "embedding");
          const heads = d && d.type === "embedding" && d.d_model % 4 === 0 ? 4 : 1;
          const attn = { ...makeBlock("attention"), n_heads: heads, n_kv_heads: heads } as Block;
          blocks.splice(at, 0, attn, makeBlock("ffn"));
          return { blocks, selectedId: attn.id };
        }),
      removeBlock: (id) => set((s) => ({ blocks: s.blocks.filter((b) => b.id !== id), selectedId: s.selectedId === id ? null : s.selectedId })),
      duplicateBlock: (id) =>
        set((s) => {
          const i = s.blocks.findIndex((b) => b.id === id);
          if (i < 0 || s.blocks[i].type === "embedding" || s.blocks[i].type === "head") return s;
          const copy = { ...s.blocks[i], id: makeBlock("dropout").id } as Block;
          const blocks = s.blocks.slice();
          blocks.splice(i + 1, 0, copy);
          return { blocks, selectedId: copy.id };
        }),
      moveBlock: (from, to) =>
        set((s) => {
          const n = s.blocks.length;
          if (from === to || from < 0 || to < 0 || from >= n || to >= n) return s;
          const lo = s.blocks[0]?.type === "embedding" ? 1 : 0;
          const hi = s.blocks[n - 1]?.type === "head" ? n - 2 : n - 1;
          const dest = Math.min(hi, Math.max(lo, to));
          if (dest === from) return s;
          const blocks = s.blocks.slice();
          const [b] = blocks.splice(from, 1);
          blocks.splice(dest, 0, b);
          return { blocks };
        }),
      updateBlock: (id, patch) => set((s) => ({ blocks: s.blocks.map((b) => (b.id === id ? ({ ...b, ...patch } as Block) : b)) })),
      select: (selectedId) => set({ selectedId }),
      hover: (hoverId) => set({ hoverId }),

      text: DEFAULT_SAMPLE.text,
      merges: DEFAULT_MERGES,
      probe: DEFAULT_SAMPLE.probe,
      hyper: DEFAULT_HYPER,
      usePasted: true,
      dataInfo: null,
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
      setDataInfo: (dataInfo) => set((s) => ({ dataInfo, dataRev: s.dataRev + 1 })),

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
    }),
    {
      name: "mb.studio.v1",
      version: 2,
      // v1 defaulted to 60 merges, which leaves most words as single letters
      migrate: (persisted, from) => {
        const p = persisted as Persisted;
        return from < 2 && p.merges === 60 ? { ...p, merges: DEFAULT_MERGES } : p;
      },
      partialize: (s) => ({
        blocks: s.blocks,
        merges: s.merges,
        usePasted: s.usePasted,
        probe: s.probe,
        hyper: s.hyper,
        splitPct: s.splitPct,
        // big uploads would blow the ~5 MB localStorage quota
        text: s.text.length < 400_000 ? s.text : undefined,
      }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<Studio>;
        return { ...current, ...p, hyper: { ...current.hyper, ...(p.hyper ?? {}) }, text: p.text ?? current.text };
      },
    },
  ),
);

export const isRunning = (s: StatusPayload) => s.state === "preparing" || s.state === "running";
