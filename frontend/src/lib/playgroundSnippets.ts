import type { Block } from "./types";

const helpers = (
  fns: string,
) => `// Available: print(...), dot, matvec, matmul, transpose, softmax, rmsnorm, gelu, silu, relu\n// (${fns})\n\n`;

function embeddingSnippet(b: Extract<Block, { type: "embedding" }>): string {
  const d = Math.min(b.d_model, 4);
  return `${helpers("this block: Embedding")}// A tiny lookup table (real one is vocab × d_model = ${b.d_model})
const table = [
  [0.02, -0.15, 0.09, -0.03],  // token 0
  [0.11,  0.04, -0.22, 0.08],  // token 1
  [-0.05, 0.19,  0.01, -0.14], // token 2
];
const tokenId = 1;
const row = table[tokenId];
print("row " + tokenId + ":", row);
${b.scale ? `
const dModel = ${d};
const scale = Math.sqrt(dModel);
print("× sqrt(d_model)=" + scale.toFixed(2) + ":", row.map(x => x * scale));` : `
// this block's "Scale by √d_model" is off, so nothing further happens here`}
`;
}

function posEncodingSnippet(): string {
  return `${helpers("this block: Positional Encoding")}function pe(pos, dModel) {
  const out = [];
  for (let i = 0; i < dModel / 2; i++) {
    const theta = pos / Math.pow(10000, (2 * i) / dModel);
    out.push(Math.sin(theta), Math.cos(theta));
  }
  return out;
}
const dModel = 4;
print("PE(pos=0):", pe(0, dModel));
print("PE(pos=1):", pe(1, dModel));
print("PE(pos=2):", pe(2, dModel));
// try a bigger position and watch the low dimensions oscillate fast, the high ones barely move
print("PE(pos=50):", pe(50, dModel));
`;
}

function attentionSnippet(b: Extract<Block, { type: "attention" }>): string {
  const causal = b.causal;
  return `${helpers("this block: Self-Attention")}// 3 tiny tokens, 1 head, head_dim=2 (real: ${b.n_heads} heads, head_dim=${64 / b.n_heads || "?"})
const Q = [[1, 0], [0, 1], [1, 1]];
const K = [[1, 0], [0, 1], [1, 1]];
const V = [[1, 2], [3, 4], [5, 6]];
const headDim = 2;

let scores = matmul(Q, transpose(K)).map(row => row.map(x => x / Math.sqrt(headDim)));
print("scores = QKᵀ/√head_dim:", scores);
${causal ? `
// causal mask: token i can't see token j > i
scores = scores.map((row, i) => row.map((x, j) => (j <= i ? x : -Infinity)));
print("after causal mask:", scores);` : `
// causal is OFF on this block — every token sees every other token, including "the future"`}

const attn = scores.map(softmax);
print("attention weights (softmax):", attn);

const out = matmul(attn, V);
print("output = attn @ V:", out);
`;
}

function ffnSnippet(b: Extract<Block, { type: "ffn" }>): string {
  const act = b.activation === "silu" ? "silu" : b.activation === "relu" ? "relu" : "gelu";
  if (b.kind === "moe") {
    return `${helpers("this block: Feed-Forward (Mixture of Experts)")}// A router picking top-${b.top_k} of ${b.n_experts} experts for one tiny token
const x = [1, -2];
const routerLogits = [0.9, -0.3, 1.4, 0.1].slice(0, ${b.n_experts});
const p = softmax(routerLogits);
print("router probabilities:", p);

const k = ${b.top_k};
const sorted = [...p].sort((a, b) => b - a);
const threshold = sorted[k - 1];
const kept = p.map(v => (v >= threshold ? v : 0));
const sum = kept.reduce((a, b) => a + b, 0);
const gate = kept.map(v => v / sum);
print("kept top-" + k + ", renormalized:", gate);

// each expert is its own tiny GeGLU (same shape as the plain GLU block below) — skipped here for brevity
print("only experts with a nonzero gate above actually run on this token");
`;
  }
  if (b.kind === "glu") {
    return `${helpers("this block: Feed-Forward (gated GLU)")}const x = [1, -2];
const Wgate = [[0.5, -0.3], [0.2, 0.8]];
const Wup   = [[1.0,  0.5], [-0.4, 0.6]];
const Wdown = [[0.7, -0.2], [0.3,  0.9]];

const gate = matvec(Wgate, x).map(${act});
const up = matvec(Wup, x);
print("${act}(gate):", gate);
print("up:", up);

const h = gate.map((g, i) => g * up[i]);
print("h = gate ⊙ up:", h);
print("y = h @ Wdown:", matvec(Wdown, h));
`;
  }
  return `${helpers("this block: Feed-Forward (plain MLP)")}const x = [1, -2];
const Wup   = [[1.0,  0.5], [-0.4, 0.6]];
const Wdown = [[0.7, -0.2], [0.3,  0.9]];

const h = matvec(Wup, x).map(${act});
print("${act}(x @ Wup):", h);
print("y = h @ Wdown:", matvec(Wdown, h));
`;
}

function normSnippet(b: Extract<Block, { type: "norm" }>): string {
  if (b.kind === "layer") {
    return `${helpers("this block: LayerNorm")}const x = [2, -4, 1, 3];
const mean = x.reduce((a, v) => a + v, 0) / x.length;
const variance = x.reduce((a, v) => a + (v - mean) ** 2, 0) / x.length;
print("mean:", mean, " variance:", variance);
const y = x.map(v => (v - mean) / Math.sqrt(variance + 1e-6));
print("LayerNorm(x):", y);
`;
  }
  return `${helpers("this block: RMSNorm")}const x = [2, -4, 1, 3];
print("x:", x);
print("RMSNorm(x):", rmsnorm(x));
// try changing x — the output always ends up at unit root-mean-square, same direction as x
`;
}

function dropoutSnippet(b: Extract<Block, { type: "dropout" }>): string {
  return `${helpers("this block: Dropout")}const x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
const p = ${b.p};
const keepProb = 1 - p;
const y = x.map(v => (Math.random() < keepProb ? v / keepProb : 0));
print("input:", x);
print("after dropout(p=" + p + "):", y);
// run again (▶) — it's random every time, exactly like real training. Roughly ${Math.round(b.p * 100)}% become 0.
`;
}

function headSnippet(b: Extract<Block, { type: "head" }>): string {
  return `${helpers("this block: LM Head")}const h = [0.4, -0.1, 0.8]; // final hidden state for one token (tiny: 3 dims)
const E = [ // ${b.tied ? "tied: this IS the embedding table" : "untied: its own matrix"} — 5 tokens × 3 dims
  [0.1,  0.2, -0.3],
  [0.5, -0.4,  0.1],
  [-0.2, 0.3,  0.6],
  [0.0,  0.1, -0.1],
  [0.7, -0.2,  0.2],
];
const logits = E.map(row => dot(row, h));
print("logits:", logits);
print("probs = softmax(logits):", softmax(logits));
`;
}

/** Starter code for a block's Playground, tied to its real settings (activation, causal, dropout p, …). */
export function starterCode(block: Block): string {
  switch (block.type) {
    case "embedding":
      return embeddingSnippet(block);
    case "pos_encoding":
      return posEncodingSnippet();
    case "attention":
      return attentionSnippet(block);
    case "ffn":
      return ffnSnippet(block);
    case "norm":
      return normSnippet(block);
    case "dropout":
      return dropoutSnippet(block);
    case "head":
      return headSnippet(block);
  }
}
