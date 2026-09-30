// Shapes shared with the Rust backend (see backend/src/spec.rs and events.rs).

export type NormKind = "none" | "rms" | "layer";
export type Activation = "gelu" | "silu" | "relu";
export type FfnKind = "glu" | "mlp" | "moe";

export type BlockParams =
  | { type: "embedding"; d_model: number; scale: boolean }
  | { type: "pos_encoding" }
  | { type: "norm"; kind: NormKind }
  | {
      type: "attention";
      n_heads: number;
      n_kv_heads: number;
      qk_norm: boolean;
      rope: boolean;
      rope_theta: number;
      window: number;
      sink: boolean;
      causal: boolean;
      residual: boolean;
      pre_norm: NormKind;
      dropout: number;
    }
  | {
      type: "ffn";
      kind: FfnKind;
      d_ff: number;
      activation: Activation;
      n_experts: number;
      top_k: number;
      residual: boolean;
      pre_norm: NormKind;
      dropout: number;
    }
  | { type: "dropout"; p: number }
  | { type: "head"; tied: boolean };

export type BlockType = BlockParams["type"];
export type Block = BlockParams & { id: string };

export interface Hyper {
  steps: number;
  batchSize: number;
  seqLen: number;
  lr: number;
  warmupSteps: number;
  weightDecay: number;
  gradClip: number;
  seed: number;
  probeEvery: number;
  evalEvery: number;
  sampleEvery: number;
  sampleTokens: number;
  temperature: number;
}

// ---------------------------------------------------------------- payloads

export interface Heat {
  rows: number;
  cols: number;
  data: number[];
}
export interface Issue {
  blockId: string | null;
  level: "error" | "warning" | "info";
  message: string;
}
export interface ShapeInfo {
  blockId: string;
  outShape: [number, number];
  params: number;
}
export interface Stats {
  mean: number;
  std: number;
  absMax: number;
}
export interface AttnTrace {
  heads: Heat[];
  sink_mass: number[] | null;
}
export interface MoeTrace {
  nExperts: number;
  gates: Heat;
  usage: number[];
}
export interface LayerTrace {
  index: number;
  kind: string;
  outShape: [number, number];
  params: number;
  stats: Stats;
  act: Heat;
  attn: AttnTrace | null;
  moe: MoeTrace | null;
  gradNorm: number | null;
  weightNorm: number | null;
}
export interface TokenInfo {
  id: number;
  text: string;
}
export interface NextToken extends TokenInfo {
  p: number;
}
export interface MapPoint extends TokenInfo {
  x: number;
  y: number;
  count: number;
}
export interface InspectPayload {
  ok: boolean;
  error: string | null;
  issues: Issue[];
  shapes: ShapeInfo[];
  totalParams: number;
  vocabSize: number;
  seqLen: number;
  trained: boolean;
  step: number | null;
  tokens: TokenInfo[];
  layers: LayerTrace[];
  nextTokens: NextToken[];
  probeLoss: number | null;
  map: MapPoint[];
}
/** Which tokenizer the dataset is built with: the built-in BPE, or an imported Hugging Face tokenizer.json. */
export interface TokenizerInfo {
  kind: "default" | "imported";
  /** File name of the imported tokenizer.json. */
  name: string;
  /** "BPE", "Unigram", "WordPiece" or "WordLevel". */
  model: string;
  /** Ids the imported tokenizer defines. */
  fullVocab: number;
}
export interface DataInfo {
  ok: boolean;
  error: string | null;
  chars: number;
  tokens: number;
  /** Model vocabulary: for an imported tokenizer, the ids the data uses (out of `fullVocab`). */
  vocabSize: number;
  fullVocab: number;
  merges: number;
  compression: number;
  preview: TokenInfo[];
  /** Pasted text + documents that went into the dataset. */
  sources: number;
  /** Bytes of source text. */
  bytes: number;
  /** The corpus was cut at the server's token limit. */
  truncated: boolean;
  buildMs: number;
}

export type DocState = "extracting" | "ready" | "error";
export interface DocInfo {
  id: string;
  name: string;
  size: number;
  state: DocState;
  kind: string | null;
  textBytes: number;
  note: string | null;
  error: string | null;
  enabled: boolean;
}
export interface BuildProgress {
  stage: "tokenizer" | "encode" | "done";
  pct: number;
  message: string;
}
/** A file the browser is still sending to the backend. */
export interface UploadItem {
  id: string;
  name: string;
  size: number;
  loaded: number;
  state: "queued" | "uploading" | "error";
  error?: string;
}
// ------------------------------------------------------------- Q&A dataset

/** How a dataset is written: LLM output style, training layout and file schema. */
export type DataStyle = "hf" | "simple" | "instruction" | "chat";

export interface QaPair {
  id: number;
  docId: string;
  doc: string;
  page: number;
  style: DataStyle;
  /** System prompt (chat style), empty when none. */
  system: string;
  question: string;
  answer: string;
  /** First ~300 characters of the page the pair came from (empty for imports without context). */
  contextHead: string;
}
export interface QaPageInfo {
  n: number;
  chars: number;
  preview: string;
}
export interface QaPageList {
  pages: QaPageInfo[];
  /** No real pages in this format: these are fixed-size sections. */
  synthetic: boolean;
  truncated: boolean;
}
export interface QaProgress {
  state: "running" | "done" | "stopped" | "error";
  done: number;
  total: number;
  pairs: number;
  failed: number;
  message: string;
}
// ------------------------------------------------------------ summary dataset

export interface SummaryPair {
  id: number;
  docId: string;
  doc: string;
  page: number;
  style: DataStyle;
  system: string;
  /** The page text the summary was written from (same clipped text the model saw). */
  text: string;
  summary: string;
}
export interface SummaryProgress {
  state: "running" | "done" | "stopped" | "error";
  done: number;
  total: number;
  summaries: number;
  failed: number;
  message: string;
}
/** What a training run learns from, and whether it starts from scratch. */
export type TrainSource = "text" | "qa" | "summary";
export type TrainMode = "new" | "continue";

export type RunState = "idle" | "preparing" | "running" | "stopped" | "done" | "error";
export interface StatusPayload {
  state: RunState;
  message: string;
  step: number;
  total: number;
}
export interface StepPoint {
  step: number;
  loss: number;
  lr: number;
  tokPerSec: number;
}
export interface EvalPoint {
  step: number;
  valLoss: number;
  perplexity: number;
}
export interface SamplePayload {
  step: number;
  prompt: string;
  text: string;
}

// -------------------------------------------------------------- Learn chat

export interface FormulaItem {
  label: string | null;
  expr: string;
}
export interface DiagramNode {
  id: string;
  label: string;
  sublabel: string | null;
  level: number | null;
}
export interface DiagramEdge {
  from: string;
  to: string;
  label: string | null;
}
export interface DiagramSpec {
  title: string | null;
  nodes: DiagramNode[];
  edges: DiagramEdge[];
}
export interface LearnAnswer {
  text: string;
  formulas: FormulaItem[];
  diagram: DiagramSpec | null;
  followUps: string[];
}
export interface LlmConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}
export interface LearnChatMsg {
  id: string;
  role: "user" | "assistant";
  /** set when role is "user" */
  question?: string;
  /** set when role is "assistant" and it succeeded */
  answer?: LearnAnswer;
  pending?: boolean;
  error?: string;
}
