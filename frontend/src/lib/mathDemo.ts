// Small, pure math helpers so the Learn tab's worked examples compute real numbers from the
// block's actual formula, instead of hand-typed (and easy to get subtly wrong) prose numbers.

export const round = (x: number, d = 3) => Math.round(x * 10 ** d) / 10 ** d;
export const r2 = (x: number) => round(x, 2);
export const r3 = (x: number) => round(x, 3);
export const r4 = (x: number) => round(x, 4);

export function softmax(xs: number[]): number[] {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function rmsnorm(xs: number[], eps = 1e-6): number[] {
  const ms = xs.reduce((a, x) => a + x * x, 0) / xs.length;
  const scale = 1 / Math.sqrt(ms + eps);
  return xs.map((x) => x * scale);
}

/** Keeps the `k` largest values of `p`, zeroes the rest, and renormalizes the kept ones to sum to 1. */
export function topKGate(p: number[], k: number): number[] {
  const threshold = [...p].sort((a, b) => b - a)[Math.min(k, p.length) - 1];
  const kept = p.map((v) => (v >= threshold ? v : 0));
  const sum = kept.reduce((a, b) => a + b, 0) || 1;
  return kept.map((v) => v / sum);
}

/** A deterministic, reproducible-looking "router output" for `n` experts (not random — same n always gives the same demo numbers). */
export function demoRouterProbs(n: number): number[] {
  const raw = Array.from({ length: n }, (_, i) => Math.exp(Math.sin(i * 2.4 + 1) * 1.7));
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map((v) => v / sum);
}

export interface RopePair {
  i: number;
  /** θᵢ = base^(-2i/head_dim): the per-step rotation rate this pair turns at, in radians/position. */
  theta: number;
  /** Total angle at position m = m · θᵢ, in radians. */
  angleRad: number;
  angleDeg: number;
  cos: number;
  sin: number;
}

/** RoPE's rotation angle for pair index `i` (0-indexed) at sequence position `m`. */
export function ropePair(base: number, headDim: number, i: number, m: number): RopePair {
  const theta = Math.pow(base, (-2 * i) / headDim);
  const angle = m * theta;
  return { i, theta: r4(theta), angleRad: r3(angle), angleDeg: r2((angle * 180) / Math.PI), cos: r4(Math.cos(angle)), sin: r4(Math.sin(angle)) };
}
