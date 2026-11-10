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

// ------------------------------------------------------- System One ("JEV model")
// A Jev-style decision model: reads a state + typed questions, answers all of them with
// calibrated probabilities in one forward pass. A fixed bidirectional-encoder architecture,
// not a stack of drag-and-drop blocks, so it has its own config/data/observe UI.

export type SysOneQuestion =
  | { type: "noul"; text: string }
  | { type: "choice"; text: string; options: string[] }
  | { type: "score"; text: string; levels: string[] };

export type SysOneTarget = { type: "noul"; p: number } | { type: "choice"; probs: number[] } | { type: "score"; probs: number[] };

export type SysOneAnswer =
  | { type: "noul"; p: number; confidence: number }
  | { type: "choice"; probs: number[]; best: number; confidence: number }
  | { type: "score"; probs: number[]; score: number; confidence: number };

export interface SysOneSample {
  id: number;
  state: string;
  questions: SysOneQuestion[];
  targets: SysOneTarget[];
  source: string;
  /** Tool-calling samples: the full call the questions were derived from. */
  call?: { tool: string; arguments: unknown } | null;
}

export interface SysOneGenProgress {
  state: "running" | "done" | "stopped" | "error";
  done: number;
  total: number;
  samples: number;
  failed: number;
  message: string;
}

/** Target ratio of question types for synthetic generation — weights, not required to sum to 100. */
export interface SysOneMix {
  noul: number;
  choice: number;
  score: number;
}

/** Mirrors the Rust `TrainHyper` field-for-field (camelCase on the wire). */
export interface SysOneHyper {
  epochs: number;
  batchSize: number;
  lr: number;
  warmupSteps: number;
  confWeight: number;
  gradClip: number;
  seed: number;
  merges: number;
  dModel: number;
  nHeads: number;
  nKvHeads: number;
  nLayers: number;
  dFf: number;
  dropout: number;
  maxSeqLen: number;
  /** Where training runs; the trained model is always served from the CPU. */
  device: "cpu" | "gpu";
  /** "scratch" = random weights; "laya" = fine-tune the pretrained Laya checkpoint (GPU only). */
  base: "scratch" | "laya";
  /** Laya: encoder layers (from the top, of 28) that are fine-tuned; the rest stay frozen. */
  layaLayers: number;
  /** Laya: tokens per question sequence. */
  layaMaxLen: number;
  /** Laya: questions drawn from each sample per epoch. */
  layaQuestions: number;
}

/** State of the local copy of Laya (pushed by the server). */
export interface LayaStatus {
  state: "missing" | "downloading" | "ready" | "error";
  done: number;
  total: number;
  message: string;
}

export interface SysOneStep {
  step: number;
  loss: number;
  noul: number;
  opt: number;
  conf: number;
  lr: number;
}

export interface SysOneEvalReport {
  nNoul: number;
  nChoice: number;
  nScore: number;
  noulAcc: number;
  choiceAcc: number;
  scoreAcc: number;
  noulBrier: number;
  choiceBrier: number;
  scoreBrier: number;
  scoreMae: number;
  probEce: number;
  confEce: number;
}

export interface SysOneEval {
  epoch: number;
  report: SysOneEvalReport;
  best: boolean;
}

// ---------------------------------------------- import a real HF dataset

export interface HfColumn {
  name: string;
  dtype: string;
  /** Present when this is a Hugging Face `ClassLabel` column — the full label vocabulary. */
  classNames: string[] | null;
}
export interface HfSplitInfo {
  config: string;
  split: string;
  numRows: number;
}
export interface HfDatasetInfo {
  dataset: string;
  splits: HfSplitInfo[];
  columns: HfColumn[];
}

// ------------------------------------------- Tool calling with the trained model

export interface ToolCandidate {
  name: string;
  relevance: number;
  choice: number | null;
}
export interface ToolDecision {
  path: string;
  question: string;
  answer: string;
  p: number;
  confidence: number;
  used: boolean;
}
export interface ToolMissing {
  path: string;
  required: boolean;
  reason: string;
}
export interface ToolCallResult {
  tool: string;
  toolConfidence: number;
  candidates: ToolCandidate[];
  specificity: string | null;
  arguments: Record<string, unknown>;
  decisions: ToolDecision[];
  missing: ToolMissing[];
  /** OpenAI shape: `[{function: {arguments: "<json string>", name}, id, type: "function"}]`. */
  toolCalls: unknown[];
}

// ------------------------------------------- Tool-calling dataset (MCP servers)

export type McpTransport = "stdio" | "sse" | "http";

/** An MCP server the user connected. Kept in the browser (localStorage); the backend only
 * receives it when asked to connect. */
export interface McpServer {
  id: string;
  /** Also the prefix of qualified tool names: `<name>__<tool>`. */
  name: string;
  transport: McpTransport;
  /** stdio: program to run (no shell) and its arguments. */
  command: string;
  args: string[];
  env: Record<string, string>;
  cwd: string;
  /** sse / http endpoint. */
  url: string;
  timeoutSecs: number | null;
  connectTimeoutSecs: number | null;
  /** Whether this server's tools are part of the tool-calling dataset. */
  enabled: boolean;
}

export interface McpTool {
  name: string;
  title: string | null;
  description: string | null;
  /** JSON Schema of the arguments. */
  inputSchema: Record<string, unknown>;
  outputSchema: Record<string, unknown> | null;
  readOnly: boolean | null;
}

/** Last connection attempt for one server (not persisted). */
export interface McpStatus {
  busy: "refresh" | "test" | null;
  ok: boolean | null;
  message: string;
  serverName?: string | null;
  serverVersion?: string | null;
  elapsedMs?: number;
}
