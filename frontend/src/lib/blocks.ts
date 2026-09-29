import {
  Binary, Droplets, Eye, Layers, Target, SlidersHorizontal, Waves, type LucideIcon,
} from "lucide-react";
import type { Block, BlockParams, BlockType } from "./types";

let counter = 0;
export const newId = () => `b${Date.now().toString(36)}${(counter++).toString(36)}`;

export type Field =
  | { key: string; label: string; kind: "int"; min: number; max: number; step?: number; help?: string; when?: (b: Block) => boolean }
  | { key: string; label: string; kind: "float"; min: number; max: number; step: number; help?: string; when?: (b: Block) => boolean }
  | { key: string; label: string; kind: "bool"; help?: string; when?: (b: Block) => boolean }
  | { key: string; label: string; kind: "select"; options: { value: string; label: string }[]; help?: string; when?: (b: Block) => boolean };

const NORMS = [
  { value: "rms", label: "RMSNorm" },
  { value: "layer", label: "LayerNorm" },
  { value: "none", label: "None" },
];

export interface BlockDef {
  type: BlockType;
  title: string;
  blurb: string;
  /** CSS color used for the accent stripe / icon. */
  color: string;
  icon: LucideIcon;
  defaults: () => BlockParams;
  fields: Field[];
  summary: (b: Block) => string;
  /** Where this idea comes from in the transformer_rs reference project. */
  origin: string;
}

const isMoe = (b: Block) => b.type === "ffn" && b.kind === "moe";

export const CATALOG: Record<BlockType, BlockDef> = {
  embedding: {
    type: "embedding",
    title: "Embedding",
    blurb: "Turns each token id into a vector. Must be first.",
    color: "#0ea5e9",
    icon: Binary,
    origin: "Gemma-style: scaled by √d_model, init std 1/√d",
    defaults: () => ({ type: "embedding", d_model: 64, scale: true }),
    fields: [
      { key: "d_model", label: "d_model", kind: "int", min: 8, max: 512, step: 8, help: "Width of every vector flowing through the model." },
      { key: "scale", label: "Scale by √d_model", kind: "bool", help: "Gemma multiplies embeddings by √d_model so they match the layers' scale." },
    ],
    summary: (b) => (b.type === "embedding" ? `d_model ${b.d_model}${b.scale ? " · scaled" : ""}` : ""),
  },
  pos_encoding: {
    type: "pos_encoding",
    title: "Positional Encoding",
    blurb: "Adds fixed sine/cosine position signals to the embeddings.",
    color: "#14b8a6",
    icon: Waves,
    origin: "Sinusoidal positions (the ATIS encoder)",
    defaults: () => ({ type: "pos_encoding" }),
    fields: [],
    summary: () => "sinusoidal",
  },
  norm: {
    type: "norm",
    title: "Normalization",
    blurb: "Rescales activations so training stays stable.",
    color: "#64748b",
    icon: SlidersHorizontal,
    origin: "RMSNorm (Gemma) or LayerNorm",
    defaults: () => ({ type: "norm", kind: "rms" }),
    fields: [{ key: "kind", label: "Kind", kind: "select", options: NORMS.filter((n) => n.value !== "none") }],
    summary: (b) => (b.type === "norm" ? (b.kind === "layer" ? "LayerNorm" : "RMSNorm") : ""),
  },
  attention: {
    type: "attention",
    title: "Self-Attention",
    blurb: "Every token looks at earlier tokens and mixes in what matters.",
    color: "#8b5cf6",
    icon: Eye,
    origin: "GQA + QK-norm + RoPE + sliding window (Gemma) · attention sink (gpt-oss)",
    defaults: () => ({
      type: "attention", n_heads: 4, n_kv_heads: 2, qk_norm: true, rope: true, rope_theta: 10000,
      window: 0, sink: false, causal: true, residual: true, pre_norm: "rms", dropout: 0,
    }),
    fields: [
      { key: "n_heads", label: "Heads", kind: "int", min: 1, max: 32, help: "Parallel attention patterns. Must divide d_model." },
      { key: "n_kv_heads", label: "KV heads", kind: "int", min: 1, max: 32, help: "Grouped-query attention: fewer key/value heads shared by the query heads. Must divide Heads." },
      { key: "causal", label: "Causal mask", kind: "bool", help: "Off = each token can see the future, including the answer. Loss will collapse (cheating)." },
      { key: "rope", label: "RoPE positions", kind: "bool", help: "Rotary position embedding applied to queries and keys." },
      { key: "rope_theta", label: "RoPE base", kind: "int", min: 100, max: 2_000_000, step: 1000, when: (b) => b.type === "attention" && b.rope },
      { key: "qk_norm", label: "QK-norm", kind: "bool", help: "RMS-normalise each head's queries/keys before attention (Gemma-3)." },
      { key: "window", label: "Sliding window", kind: "int", min: 0, max: 256, help: "0 = attend to the whole past; N = only the last N tokens." },
      { key: "sink", label: "Attention sink", kind: "bool", help: "A learned 'attend to nothing' slot per head (gpt-oss)." },
      { key: "pre_norm", label: "Pre-norm", kind: "select", options: NORMS },
      { key: "residual", label: "Residual add", kind: "bool", help: "Output = input + attention(input)." },
      { key: "dropout", label: "Dropout", kind: "float", min: 0, max: 0.9, step: 0.05 },
    ],
    summary: (b) => {
      if (b.type !== "attention") return "";
      const bits = [`${b.n_heads}h`, b.n_kv_heads !== b.n_heads ? `${b.n_kv_heads}kv` : ""];
      if (b.window) bits.push(`win ${b.window}`);
      if (b.sink) bits.push("sink");
      if (!b.causal) bits.push("NON-CAUSAL");
      return bits.filter(Boolean).join(" · ");
    },
  },
  ffn: {
    type: "ffn",
    title: "Feed-Forward",
    blurb: "Per-token MLP. Gated (GeGLU), plain, or a mixture of experts.",
    color: "#f59e0b",
    icon: Layers,
    origin: "GeGLU (Gemma) · top-k Mixture of Experts (gpt-oss)",
    defaults: () => ({
      type: "ffn", kind: "glu", d_ff: 128, activation: "gelu", n_experts: 4, top_k: 2,
      residual: true, pre_norm: "rms", dropout: 0,
    }),
    fields: [
      {
        key: "kind", label: "Type", kind: "select",
        options: [
          { value: "glu", label: "Gated GLU (GeGLU/SwiGLU)" },
          { value: "mlp", label: "Plain MLP" },
          { value: "moe", label: "Mixture of Experts" },
        ],
      },
      { key: "d_ff", label: "Hidden size", kind: "int", min: 8, max: 4096, step: 8 },
      {
        key: "activation", label: "Activation", kind: "select",
        options: [{ value: "gelu", label: "GELU" }, { value: "silu", label: "SiLU" }, { value: "relu", label: "ReLU" }],
      },
      { key: "n_experts", label: "Experts", kind: "int", min: 2, max: 16, when: isMoe },
      { key: "top_k", label: "Top-k routed", kind: "int", min: 1, max: 16, help: "Each token uses only this many experts.", when: isMoe },
      { key: "pre_norm", label: "Pre-norm", kind: "select", options: NORMS },
      { key: "residual", label: "Residual add", kind: "bool" },
      { key: "dropout", label: "Dropout", kind: "float", min: 0, max: 0.9, step: 0.05 },
    ],
    summary: (b) => {
      if (b.type !== "ffn") return "";
      if (b.kind === "moe") return `MoE ${b.top_k}/${b.n_experts} · ff ${b.d_ff}`;
      return `${b.kind === "glu" ? "GLU" : "MLP"} · ff ${b.d_ff}`;
    },
  },
  dropout: {
    type: "dropout",
    title: "Dropout",
    blurb: "Randomly zeroes activations while training to reduce overfitting.",
    color: "#94a3b8",
    icon: Droplets,
    origin: "Regularisation",
    defaults: () => ({ type: "dropout", p: 0.1 }),
    fields: [{ key: "p", label: "Probability", kind: "float", min: 0, max: 0.9, step: 0.05 }],
    summary: (b) => (b.type === "dropout" ? `p ${b.p}` : ""),
  },
  head: {
    type: "head",
    title: "LM Head",
    blurb: "Scores every vocabulary token for what comes next. Must be last.",
    color: "#f43f5e",
    icon: Target,
    origin: "Weight-tied to the embedding by default (Gemma)",
    defaults: () => ({ type: "head", tied: true }),
    fields: [{ key: "tied", label: "Tie with embedding", kind: "bool", help: "Reuse the embedding matrix as the output projection (fewer parameters)." }],
    summary: (b) => (b.type === "head" ? (b.tied ? "tied weights" : "separate weights") : ""),
  },
};

export const PALETTE_ORDER: BlockType[] = ["embedding", "pos_encoding", "attention", "ffn", "norm", "dropout", "head"];

export function makeBlock(type: BlockType): Block {
  return { ...CATALOG[type].defaults(), id: newId() } as Block;
}

/** Stable, human name for a block in the stack ("Self-Attention 2"). */
export function blockLabel(blocks: Block[], index: number): string {
  const b = blocks[index];
  const n = blocks.slice(0, index + 1).filter((x) => x.type === b.type).length;
  const same = blocks.filter((x) => x.type === b.type).length;
  return same > 1 ? `${CATALOG[b.type].title} ${n}` : CATALOG[b.type].title;
}

export const fmtParams = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${n}`);
