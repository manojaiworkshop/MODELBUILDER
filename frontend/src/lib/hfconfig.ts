// Turns a Hugging Face `config.json` into this app's block stack.
// Pure functions: no network, no store. The UI (HfModelModal) feeds them.
import { makeBlock } from "./blocks";
import type { Activation, Block, BlockType, FfnKind, NormKind } from "./types";

type Cfg = Record<string, unknown>;

/** Backend limits (backend/src/spec.rs): 40 blocks, d_model 512 → at most 18 layers of attention+FFN. */
export const MAX_LAYERS = 18;
export const MAX_D_MODEL = 512;

export interface Arch {
  name: string;
  modelType: string;
  hidden: number;
  heads: number;
  kvHeads: number;
  layers: number;
  /** Sliding-window size per original layer (0 = full attention). */
  windows: number[];
  ffn: FfnKind;
  dff: number;
  act: Activation;
  experts: number;
  topK: number;
  norm: NormKind;
  rope: boolean;
  ropeTheta: number;
  qkNorm: boolean;
  sink: boolean;
  scaleEmb: boolean;
  tied: boolean;
  learnedPos: boolean;
  vocab: number;
  /** Set when the file is this app's own exported config.json (already a block list). */
  nativeBlocks?: Block[];
}

export interface FitOptions {
  dModel: number;
  layers: number;
  maxExperts: number;
  maxWindow: number;
}

const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
const str = (v: unknown): string => (typeof v === "string" ? v : "");
const first = (...v: unknown[]): number | undefined => v.map(num).find((x) => x !== undefined);

const MLP_TYPES = /^(gpt2|gpt_neo|gptj|gpt_neox|opt|bloom|falcon|bert|phi|starcoder2|codegen|xglm|mpt)$/;
const LEARNED_POS = /^(gpt2|gpt_neo|opt|bloom|bert|xglm|codegen)$/;
const LAYERNORM_TYPES = /^(gpt2|gpt_neo|gptj|gpt_neox|opt|bloom|falcon|bert|phi|starcoder2|codegen|xglm|mpt)$/;

export function parseHfConfig(raw: unknown, fallbackName = "HF model"): Arch {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("config.json must be a JSON object.");
  const top = raw as Cfg;

  // This app's own export: `blocks` is already the stack.
  if (Array.isArray(top.blocks) && top.model_type === "modelbuilder") {
    const blocks = (top.blocks as Cfg[]).map((b) => ({ ...b, id: makeBlock("dropout").id })) as unknown as Block[];
    return {
      name: fallbackName, modelType: "modelbuilder", hidden: 0, heads: 0, kvHeads: 0, layers: 0, windows: [], ffn: "glu", dff: 0, act: "gelu",
      experts: 0, topK: 0, norm: "rms", rope: true, ropeTheta: 1e4, qkNorm: false, sink: false, scaleEmb: false, tied: true, learnedPos: false, vocab: 0, nativeBlocks: blocks,
    };
  }

  // Multimodal configs keep the language model under `text_config`.
  const sub = top.text_config && typeof top.text_config === "object" ? (top.text_config as Cfg) : {};
  const c: Cfg = { ...top, ...sub };
  const modelType = str(c.model_type) || str(top.model_type);

  const hidden = first(c.hidden_size, c.n_embd, c.d_model, c.dim);
  const heads = first(c.num_attention_heads, c.n_head, c.n_heads, c.num_heads);
  const layers = first(c.num_hidden_layers, c.n_layer, c.num_layers, c.n_layers);
  if (!hidden || !heads || !layers) {
    throw new Error("This doesn't look like a transformer language-model config (need hidden_size, num_attention_heads and num_hidden_layers).");
  }
  const kvHeads = first(c.num_key_value_heads, c.n_head_kv, c.num_kv_heads, c.multi_query_group_num) ?? (c.multi_query === true ? 1 : heads);

  // ---- feed-forward
  const experts = first(c.num_local_experts, c.num_experts, c.n_routed_experts, c.moe_num_experts) ?? 0;
  const topK = first(c.num_experts_per_tok, c.moe_top_k, c.top_k) ?? 2;
  const ffn: FfnKind = experts > 1 ? "moe" : MLP_TYPES.test(modelType) ? "mlp" : "glu";
  const dff = (experts > 1 ? first(c.moe_intermediate_size, c.intermediate_size, c.ffn_dim) : first(c.intermediate_size, c.n_inner, c.ffn_dim, c.ffn_hidden_size)) ?? hidden * 4;
  const actName = (str(c.hidden_act) || str(c.hidden_activation) || str(c.activation_function) || "gelu").toLowerCase();
  const act: Activation = /silu|swish/.test(actName) ? "silu" : /relu/.test(actName) ? "relu" : "gelu";

  // ---- normalisation
  const norm: NormKind = c.rms_norm_eps !== undefined || c.rms_norm_epsilon !== undefined ? "rms" : LAYERNORM_TYPES.test(modelType) || c.layer_norm_eps !== undefined || c.layer_norm_epsilon !== undefined ? "layer" : "rms";

  // ---- positions
  const learnedPos = LEARNED_POS.test(modelType);
  const rp = (c.rope_parameters ?? c.rope_scaling) as Cfg | undefined;
  const ropeTheta = first(c.rope_theta, rp?.rope_theta, (rp?.full_attention as Cfg | undefined)?.rope_theta, c.rotary_emb_base) ?? 1e4;

  // ---- sliding windows
  const sw = num(c.sliding_window) ?? 0;
  let windows: number[] = Array(layers).fill(0);
  if (Array.isArray(c.layer_types)) {
    windows = (c.layer_types as unknown[]).slice(0, layers).map((t) => (/sliding/i.test(str(t)) ? sw || 128 : 0));
    while (windows.length < layers) windows.push(0);
  } else if (sw > 0 && c.use_sliding_window !== false) {
    if (/^gemma3/.test(modelType)) {
      const pat = num(c.sliding_window_pattern) ?? 6; // every `pat`-th layer is global
      windows = windows.map((_, i) => ((i + 1) % pat === 0 ? 0 : sw));
    } else if (/^gemma2/.test(modelType)) {
      windows = windows.map((_, i) => (i % 2 === 0 ? sw : 0));
    } else {
      windows = windows.map(() => sw);
    }
  }

  const name = str(c.name_or_path) || str(top._name_or_path) || fallbackName;
  return {
    name,
    modelType: modelType || "unknown",
    hidden, heads, kvHeads, layers, windows, ffn, dff, act, experts, topK, norm,
    rope: !learnedPos,
    ropeTheta,
    qkNorm: /^(gemma3|qwen3|olmo2)/.test(modelType),
    sink: /^gpt_oss/.test(modelType),
    scaleEmb: /^gemma/.test(modelType),
    tied: typeof c.tie_word_embeddings === "boolean" ? c.tie_word_embeddings : modelType === "gpt2",
    learnedPos,
    vocab: num(c.vocab_size) ?? 0,
  };
}

/** Sensible starting size: small enough to train on a CPU in minutes, shaped like the original. */
export function defaultFit(a: Arch): FitOptions {
  return {
    dModel: Math.min(a.hidden, 128),
    layers: Math.min(a.layers, 4),
    maxExperts: 4,
    maxWindow: 32,
  };
}

const divisors = (n: number) => Array.from({ length: n }, (_, i) => i + 1).filter((d) => n % d === 0);

/** Head count near dModel/32 (≤ original) that divides dModel and leaves an even head size for RoPE. */
function fitHeads(a: Arch, dModel: number): { heads: number; kv: number } {
  const target = Math.max(1, Math.min(a.heads, Math.round(dModel / 32)));
  const ok = divisors(dModel).filter((h) => h <= Math.max(target, 1) && (dModel / h) % 2 === 0);
  const heads = ok.length ? Math.max(...ok) : 1;
  const group = Math.max(1, Math.round(a.heads / Math.max(a.kvHeads, 1))); // query heads per KV head
  const kvWanted = Math.max(1, Math.round(heads / group));
  const kv = divisors(heads).reduce((best, d) => (Math.abs(d - kvWanted) < Math.abs(best - kvWanted) ? d : best), 1);
  return { heads, kv };
}

export function buildBlocks(a: Arch, o: FitOptions): { blocks: Block[]; notes: string[] } {
  if (a.nativeBlocks) return { blocks: a.nativeBlocks, notes: ["This is a ModelBuilder export — the saved block list is loaded as is."] };
  const B = (type: BlockType, patch: Record<string, unknown> = {}): Block => ({ ...makeBlock(type), ...patch }) as Block;
  const notes: string[] = [];

  const dModel = Math.max(8, Math.min(MAX_D_MODEL, o.dModel));
  const nLayers = Math.max(1, Math.min(MAX_LAYERS, o.layers, a.layers));
  const { heads, kv } = fitHeads(a, dModel);
  const dff = Math.min(4096, Math.max(8, Math.round(((dModel * a.dff) / a.hidden) / 8) * 8));
  const experts = a.ffn === "moe" ? Math.max(2, Math.min(a.experts, o.maxExperts)) : 0;
  const topK = Math.max(1, Math.min(a.topK, experts || 1));

  // Evenly spaced original layers, so the sliding/global mix is preserved.
  const picks = Array.from({ length: nLayers }, (_, i) => (nLayers === 1 ? 0 : Math.round((i * (a.layers - 1)) / (nLayers - 1))));

  const blocks: Block[] = [B("embedding", { d_model: dModel, scale: a.scaleEmb })];
  if (a.learnedPos) {
    blocks.push(B("pos_encoding"));
    notes.push("The original uses learned position embeddings; sinusoidal positions are used instead.");
  }
  for (const idx of picks) {
    const w = a.windows[idx] ?? 0;
    blocks.push(
      B("attention", {
        n_heads: heads, n_kv_heads: kv, qk_norm: a.qkNorm, rope: a.rope, rope_theta: a.ropeTheta,
        window: w ? Math.min(w, o.maxWindow) : 0, sink: a.sink, causal: true, residual: true, pre_norm: a.norm, dropout: 0,
      }),
      B("ffn", {
        kind: a.ffn, d_ff: dff, activation: a.act, n_experts: experts || 4, top_k: topK, residual: true, pre_norm: a.norm, dropout: 0,
      }),
    );
  }
  blocks.push(B("norm", { kind: a.norm }), B("head", { tied: a.tied }));

  if (a.hidden !== dModel || a.layers !== nLayers) {
    notes.push(`Scaled down from ${a.hidden}-wide × ${a.layers} layers to ${dModel}-wide × ${nLayers} layers; the design (heads, GQA, FFN type, norm, windows) is kept.`);
  }
  if (a.heads !== heads) notes.push(`Attention heads ${a.heads} → ${heads} (KV heads ${a.kvHeads} → ${kv}).`);
  if (a.experts > experts && experts) notes.push(`Mixture-of-experts reduced from ${a.experts} to ${experts} experts (top-${topK}).`);
  if (a.windows.some((w) => w > o.maxWindow)) notes.push(`Sliding windows capped at ${o.maxWindow} tokens (this app's context is ≤ 256).`);
  if (a.vocab) notes.push(`Vocabulary (${a.vocab.toLocaleString()} tokens in the original) comes from the tokenizer you train with, not the config.`);
  notes.push("Weights are not downloaded — the architecture is rebuilt and trained from scratch here.");
  return { blocks, notes };
}

/** `org/name`, a huggingface.co URL (with /tree/<rev>, /blob/…), or a direct link to a config.json. */
export function configUrlFor(input: string): { url: string; id: string } | null {
  const s = input.trim();
  if (!s) return null;
  if (/^https?:\/\/.+config\.json(\?.*)?$/i.test(s)) return { url: s.replace("/blob/", "/resolve/"), id: s.split("/").slice(-3, -1).join("/") };
  const m = s.replace(/^https?:\/\/(www\.)?huggingface\.co\//i, "").replace(/^\/+/, "").split(/[?#]/)[0].split("/");
  if (m.length < 2 || !m[0] || !m[1]) return null;
  const id = `${m[0]}/${m[1]}`;
  const rev = (m[2] === "tree" || m[2] === "resolve" || m[2] === "blob") && m[3] ? m[3] : "main";
  return { url: `https://huggingface.co/${id}/resolve/${encodeURIComponent(rev)}/config.json`, id };
}

export async function fetchHfConfig(input: string): Promise<{ cfg: unknown; id: string }> {
  const t = configUrlFor(input);
  if (!t) throw new Error('Enter a model like "google/gemma-3-270m" or a huggingface.co URL.');
  let res: Response;
  try {
    res = await fetch(t.url);
  } catch {
    throw new Error("Couldn't reach huggingface.co from the browser. Download config.json and upload it instead.");
  }
  if (res.status === 401 || res.status === 403) throw new Error(`"${t.id}" is gated — accept its licence on huggingface.co, download config.json and upload it here.`);
  if (res.status === 404) throw new Error(`No config.json found for "${t.id}" (check the name; is it a model, not a dataset?).`);
  if (!res.ok) throw new Error(`Hugging Face returned HTTP ${res.status}.`);
  try {
    return { cfg: await res.json(), id: t.id };
  } catch {
    throw new Error("config.json wasn't valid JSON.");
  }
}

/** From a picked folder or loose files: the shallowest config.json. */
export async function readConfigFromFiles(files: FileList | File[]): Promise<{ cfg: unknown; id: string }> {
  const list = Array.from(files).filter((f) => f.name === "config.json" || (files.length === 1 && f.name.endsWith(".json")));
  if (!list.length) throw new Error("No config.json found in the selection.");
  const rel = (f: File) => (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
  list.sort((x, y) => rel(x).split("/").length - rel(y).split("/").length);
  const f = list[0];
  try {
    const parts = rel(f).split("/");
    return { cfg: JSON.parse(await f.text()), id: parts.length > 1 ? parts[0] : f.name };
  } catch {
    throw new Error(`${rel(f)} isn't valid JSON.`);
  }
}
