import { makeBlock } from "./blocks";
import type { Block, BlockType } from "./types";

type Patch = Record<string, unknown>;
const B = (type: BlockType, patch: Patch = {}): Block => ({ ...makeBlock(type), ...patch }) as Block;

export interface Preset {
  id: string;
  name: string;
  blurb: string;
  build: () => Block[];
}

export const PRESETS: Preset[] = [
  {
    id: "tiny-gpt",
    name: "Tiny GPT",
    blurb: "Two standard layers: attention then feed-forward.",
    build: () => [
      B("embedding", { d_model: 64 }),
      B("attention", { n_heads: 4, n_kv_heads: 4, dropout: 0.1 }),
      B("ffn", { kind: "glu", d_ff: 128, dropout: 0.1 }),
      B("attention", { n_heads: 4, n_kv_heads: 4, dropout: 0.1 }),
      B("ffn", { kind: "glu", d_ff: 128, dropout: 0.1 }),
      B("norm"),
      B("head"),
    ],
  },
  {
    id: "gemma-mini",
    name: "Gemma-mini",
    blurb: "GQA 4:1, QK-norm, GeGLU, local sliding-window layers + a global one.",
    build: () => [
      B("embedding", { d_model: 64 }),
      B("attention", { n_heads: 4, n_kv_heads: 1, window: 16 }),
      B("ffn", { kind: "glu", d_ff: 192 }),
      B("attention", { n_heads: 4, n_kv_heads: 1, window: 0, rope_theta: 1_000_000 }),
      B("ffn", { kind: "glu", d_ff: 192 }),
      B("norm"),
      B("head"),
    ],
  },
  {
    id: "gpt-oss-mini",
    name: "GPT-OSS-mini",
    blurb: "Attention sinks + a 4-expert top-2 mixture-of-experts FFN.",
    build: () => [
      B("embedding", { d_model: 64 }),
      B("attention", { n_heads: 4, n_kv_heads: 1, window: 16, sink: true }),
      B("ffn", { kind: "moe", d_ff: 64, n_experts: 4, top_k: 2 }),
      B("attention", { n_heads: 4, n_kv_heads: 1, sink: true, rope_theta: 1_000_000 }),
      B("ffn", { kind: "moe", d_ff: 64, n_experts: 4, top_k: 2 }),
      B("norm"),
      B("head"),
    ],
  },
  {
    id: "classic",
    name: "Classic (2017)",
    blurb: "Sinusoidal positions, LayerNorm, ReLU MLP, untied head.",
    build: () => [
      B("embedding", { d_model: 64, scale: true }),
      B("pos_encoding"),
      B("attention", { n_heads: 4, n_kv_heads: 4, rope: false, qk_norm: false, pre_norm: "layer" }),
      B("ffn", { kind: "mlp", d_ff: 128, activation: "relu", pre_norm: "layer" }),
      B("norm", { kind: "layer" }),
      B("head", { tied: false }),
    ],
  },
  {
    id: "bigram",
    name: "Bigram (no attention)",
    blurb: "Baseline: predicts each token from itself only. Compare its loss!",
    build: () => [B("embedding", { d_model: 64 }), B("ffn", { kind: "glu", d_ff: 128 }), B("head")],
  },
  {
    id: "empty",
    name: "Start from scratch",
    blurb: "An empty canvas — drag blocks in yourself.",
    build: () => [],
  },
];
