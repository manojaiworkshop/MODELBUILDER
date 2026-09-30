import { loopPlan, type BlockGroup, type GroupMembership, type LoopPlan } from "./transformerGroups";
import type { Block } from "./types";

export interface CodeLine {
  text: string;
  blockId?: string;
}
export type Lang = "python" | "rust" | "json";

const q = (s: string) => JSON.stringify(s);
const py = (v: unknown): string => (typeof v === "boolean" ? (v ? "True" : "False") : typeof v === "string" ? q(v) : String(v));

/** name like attn_2 — numbering per block type, so code names match card titles. */
function names(blocks: Block[]): string[] {
  const base: Record<string, string> = {
    embedding: "embed", pos_encoding: "pos", norm: "norm", attention: "attn", ffn: "ffn", dropout: "drop", head: "head",
  };
  const seen: Record<string, number> = {};
  const total: Record<string, number> = {};
  blocks.forEach((b) => (total[b.type] = (total[b.type] ?? 0) + 1));
  return blocks.map((b) => {
    seen[b.type] = (seen[b.type] ?? 0) + 1;
    return total[b.type] > 1 ? `${base[b.type]}_${seen[b.type]}` : base[b.type];
  });
}

function pyCtor(b: Block, d: number): string {
  switch (b.type) {
    case "embedding": return `Embedding(vocab_size, d_model=${b.d_model}, scale=${py(b.scale)})`;
    case "pos_encoding": return `SinusoidalPositions(d_model=${d})`;
    case "norm": return `${b.kind === "layer" ? "LayerNorm" : "RMSNorm"}(${d})`;
    case "attention": {
      const a = [`d_model=${d}`, `heads=${b.n_heads}`, `kv_heads=${b.n_kv_heads}`, `rope=${py(b.rope)}`];
      if (b.rope) a.push(`rope_theta=${b.rope_theta}`);
      a.push(`qk_norm=${py(b.qk_norm)}`, `window=${b.window || "None"}`, `sink=${py(b.sink)}`, `causal=${py(b.causal)}`, `pre_norm=${py(b.pre_norm)}`, `residual=${py(b.residual)}`);
      if (b.dropout) a.push(`dropout=${b.dropout}`);
      return `Attention(${a.join(", ")})`;
    }
    case "ffn": {
      const a = [`d_model=${d}`, `kind=${py(b.kind)}`, `d_ff=${b.d_ff}`, `activation=${py(b.activation)}`];
      if (b.kind === "moe") a.push(`experts=${b.n_experts}`, `top_k=${b.top_k}`);
      a.push(`pre_norm=${py(b.pre_norm)}`, `residual=${py(b.residual)}`);
      if (b.dropout) a.push(`dropout=${b.dropout}`);
      return `FeedForward(${a.join(", ")})`;
    }
    case "dropout": return `nn.Dropout(${b.p})`;
    case "head": return b.tied ? "LMHead(tied_to=self.embed)" : `LMHead(d_model=${d}, vocab_size=vocab_size)`;
  }
}

/** `x = self.name(src)`, right-padded with a shape comment — the one forward line every Python
 * renderer (flat or looped) needs, so both paths format it identically. */
function pyForwardLine(name: string, b: Block, src: string, d: number, vocabDst: boolean): CodeLine {
  const shape = vocabDst && b.type === "head" ? "[batch, seq, vocab]" : `[batch, seq, ${d}]`;
  const dst = vocabDst && b.type === "head" ? "logits" : "x";
  return { text: `        ${dst} = self.${name}(${src})`.padEnd(46) + `# ${shape}`, blockId: b.id };
}

function pythonFlat(blocks: Block[]): CodeLine[] {
  const nm = names(blocks);
  const emb = blocks.find((b) => b.type === "embedding");
  const d = emb && emb.type === "embedding" ? emb.d_model : 0;
  const L: CodeLine[] = [];
  L.push({ text: "import torch.nn as nn" });
  L.push({ text: "" });
  L.push({ text: "class Transformer(nn.Module):" });
  L.push({ text: "    def __init__(self, vocab_size):" });
  L.push({ text: "        super().__init__()" });
  if (!blocks.length) L.push({ text: "        # drag an Embedding block onto the canvas…" });
  blocks.forEach((b, i) => L.push({ text: `        self.${nm[i]} = ${pyCtor(b, d)}`, blockId: b.id }));
  L.push({ text: "" });
  L.push({ text: "    def forward(self, ids):            # ids: [batch, seq]" });
  blocks.forEach((b, i) => L.push(pyForwardLine(nm[i], b, i === 0 ? "ids" : "x", d, true)));
  if (blocks.length && blocks[blocks.length - 1].type === "head") {
    L.push({ text: "        return logits                  # softmax → next-token probabilities" });
  }
  return L;
}

/** Same computation as `pythonFlat`, but a repeated, identical [Attention, Feed-Forward] run is
 * factored into its own `TransformerBlock` class stacked in a loop — the way real model code
 * (nanoGPT, HF Transformers, …) is actually structured, instead of every layer unrolled by hand. */
function pythonLooped(blocks: Block[], plan: LoopPlan): CodeLine[] {
  const emb = blocks.find((b) => b.type === "embedding");
  const d = emb && emb.type === "embedding" ? emb.d_model : 0;
  const { before, attn, ffn, count, after } = plan;
  const beforeNames = names(before);
  const afterNames = names(after);
  const L: CodeLine[] = [];
  L.push({ text: "import torch.nn as nn" });
  L.push({ text: "" });
  L.push({ text: "class TransformerBlock(nn.Module):" });
  L.push({ text: "    def __init__(self, vocab_size):" });
  L.push({ text: "        super().__init__()" });
  L.push({ text: `        self.attn = ${pyCtor(attn, d)}`, blockId: attn.id });
  L.push({ text: `        self.ffn = ${pyCtor(ffn, d)}`, blockId: ffn.id });
  L.push({ text: "" });
  L.push({ text: "    def forward(self, x):" });
  L.push({ text: "        x = self.attn(x)               # [batch, seq, " + d + "]", blockId: attn.id });
  L.push({ text: "        x = self.ffn(x)                # [batch, seq, " + d + "]", blockId: ffn.id });
  L.push({ text: "        return x" });
  L.push({ text: "" });
  L.push({ text: "" });
  L.push({ text: "class Transformer(nn.Module):" });
  L.push({ text: "    def __init__(self, vocab_size):" });
  L.push({ text: "        super().__init__()" });
  before.forEach((b, i) => L.push({ text: `        self.${beforeNames[i]} = ${pyCtor(b, d)}`, blockId: b.id }));
  L.push({ text: `        self.blocks = nn.ModuleList([TransformerBlock(vocab_size) for _ in range(${count})])  # ×${count}, all identical` });
  after.forEach((b, i) => L.push({ text: `        self.${afterNames[i]} = ${pyCtor(b, d)}`, blockId: b.id }));
  L.push({ text: "" });
  L.push({ text: "    def forward(self, ids):            # ids: [batch, seq]" });
  before.forEach((b, i) => L.push(pyForwardLine(beforeNames[i], b, i === 0 ? "ids" : "x", d, false)));
  L.push({ text: "        for block in self.blocks:              # the ×" + count + " Transformer Block loop" });
  L.push({ text: "            x = block(x)                       # [batch, seq, " + d + "]" });
  after.forEach((b, i) => L.push(pyForwardLine(afterNames[i], b, "x", d, true)));
  if (after.length && after[after.length - 1].type === "head") {
    L.push({ text: "        return logits                  # softmax → next-token probabilities" });
  }
  return L;
}

export function pythonCode(blocks: Block[], groupOf: GroupMembership, groups: BlockGroup[]): CodeLine[] {
  const plan = loopPlan(blocks, groupOf, groups);
  return plan ? pythonLooped(blocks, plan) : pythonFlat(blocks);
}

function rsType(b: Block): string {
  switch (b.type) {
    case "embedding": return "Embedding<B>";
    case "pos_encoding": return "PositionalEncoding<B>";
    case "norm": return b.kind === "layer" ? "LayerNorm<B>" : "RmsNorm<B>";
    case "attention": return "Attention<B>";
    case "ffn": return b.kind === "moe" ? "MoeFfn<B>" : b.kind === "glu" ? "GluFfn<B>" : "MlpFfn<B>";
    case "dropout": return "Dropout";
    case "head": return b.tied ? "TiedHead" : "Linear<B>";
  }
}

function rsInit(b: Block, d: number): string {
  switch (b.type) {
    case "embedding": return `EmbeddingConfig::new(vocab, ${b.d_model})${b.scale ? ".scaled()" : ""}.init(dev)`;
    case "pos_encoding": return `PositionalEncodingConfig::new(${d}).init(dev)`;
    case "norm": return b.kind === "layer" ? `LayerNormConfig::new(${d}).init(dev)` : `RmsNormConfig::new(${d}).init(dev)`;
    case "attention": {
      let s = `AttentionConfig::new(${d}, ${b.n_heads}).kv_heads(${b.n_kv_heads})`;
      if (b.rope) s += `.rope(${b.rope_theta}f32)`;
      if (b.qk_norm) s += ".qk_norm()";
      if (b.window) s += `.window(${b.window})`;
      if (b.sink) s += ".sink()";
      if (!b.causal) s += ".bidirectional()";
      return `${s}.init(dev)`;
    }
    case "ffn": {
      const s = b.kind === "moe" ? `MoeConfig::new(${d}, ${b.d_ff}, ${b.n_experts}).top_k(${b.top_k})` : `${b.kind === "glu" ? "GluConfig" : "MlpConfig"}::new(${d}, ${b.d_ff})`;
      return `${s}.init(dev)`;
    }
    case "dropout": return `DropoutConfig::new(${b.p}).init()`;
    case "head": return b.tied ? "TiedHead" : `LinearConfig::new(${d}, vocab).with_bias(false).init(dev)`;
  }
}

function rustFlat(blocks: Block[]): CodeLine[] {
  const nm = names(blocks);
  const emb = blocks.find((b) => b.type === "embedding");
  const d = emb && emb.type === "embedding" ? emb.d_model : 0;
  const L: CodeLine[] = [];
  L.push({ text: "#[derive(Module, Debug)]" });
  L.push({ text: "pub struct Transformer<B: Backend> {" });
  blocks.forEach((b, i) => L.push({ text: `    ${nm[i]}: ${rsType(b)},`, blockId: b.id }));
  L.push({ text: "}" });
  L.push({ text: "" });
  L.push({ text: "impl<B: Backend> Transformer<B> {" });
  L.push({ text: "    pub fn new(vocab: usize, dev: &B::Device) -> Self {" });
  L.push({ text: "        Self {" });
  blocks.forEach((b, i) => L.push({ text: `            ${nm[i]}: ${rsInit(b, d)},`, blockId: b.id }));
  L.push({ text: "        }" });
  L.push({ text: "    }" });
  L.push({ text: "" });
  L.push({ text: "    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {" });
  blocks.forEach((b, i) => {
    const src = b.type === "embedding" ? "ids" : "x";
    const last = i === blocks.length - 1;
    const call = b.type === "head" && b.tied ? `self.${nm[i]}.forward(x, &self.embed)` : `self.${nm[i]}.forward(${src})`;
    L.push({ text: last ? `        ${call}` : `        let x = ${call};`, blockId: b.id });
  });
  L.push({ text: "    }" });
  L.push({ text: "}" });
  return L;
}

/** Rust counterpart of `pythonLooped`: a `TransformerBlock<B>` struct held in a `Vec`, built and
 * driven by a loop, instead of one field per repeated layer. */
function rustLooped(blocks: Block[], plan: LoopPlan): CodeLine[] {
  const emb = blocks.find((b) => b.type === "embedding");
  const d = emb && emb.type === "embedding" ? emb.d_model : 0;
  const { before, attn, ffn, count, after } = plan;
  const beforeNames = names(before);
  const afterNames = names(after);
  const L: CodeLine[] = [];
  L.push({ text: "#[derive(Module, Debug)]" });
  L.push({ text: "pub struct TransformerBlock<B: Backend> {" });
  L.push({ text: `    attn: ${rsType(attn)},`, blockId: attn.id });
  L.push({ text: `    ffn: ${rsType(ffn)},`, blockId: ffn.id });
  L.push({ text: "}" });
  L.push({ text: "" });
  L.push({ text: "impl<B: Backend> TransformerBlock<B> {" });
  L.push({ text: "    pub fn new(dev: &B::Device) -> Self {" });
  L.push({ text: `        Self { attn: ${rsInit(attn, d)}, ffn: ${rsInit(ffn, d)} }`, blockId: attn.id });
  L.push({ text: "    }" });
  L.push({ text: "    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {" });
  L.push({ text: "        self.ffn.forward(self.attn.forward(x))" });
  L.push({ text: "    }" });
  L.push({ text: "}" });
  L.push({ text: "" });
  L.push({ text: "#[derive(Module, Debug)]" });
  L.push({ text: "pub struct Transformer<B: Backend> {" });
  before.forEach((b, i) => L.push({ text: `    ${beforeNames[i]}: ${rsType(b)},`, blockId: b.id }));
  L.push({ text: "    blocks: Vec<TransformerBlock<B>>," });
  after.forEach((b, i) => L.push({ text: `    ${afterNames[i]}: ${rsType(b)},`, blockId: b.id }));
  L.push({ text: "}" });
  L.push({ text: "" });
  L.push({ text: "impl<B: Backend> Transformer<B> {" });
  L.push({ text: "    pub fn new(vocab: usize, dev: &B::Device) -> Self {" });
  L.push({ text: "        Self {" });
  before.forEach((b, i) => L.push({ text: `            ${beforeNames[i]}: ${rsInit(b, d)},`, blockId: b.id }));
  L.push({ text: `            blocks: (0..${count}).map(|_| TransformerBlock::new(dev)).collect(),  // ×${count}, all identical` });
  after.forEach((b, i) => L.push({ text: `            ${afterNames[i]}: ${rsInit(b, d)},`, blockId: b.id }));
  L.push({ text: "        }" });
  L.push({ text: "    }" });
  L.push({ text: "" });
  L.push({ text: "    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {" });
  before.forEach((b, i) => L.push({ text: `        let x = self.${beforeNames[i]}.forward(${i === 0 ? "ids" : "x"});`, blockId: b.id }));
  L.push({ text: "        let x = self.blocks.iter().fold(x, |x, block| block.forward(x));" });
  after.forEach((b, i) => {
    const last = i === after.length - 1;
    const call = b.type === "head" && b.tied ? `self.${afterNames[i]}.forward(x, &self.embed)` : `self.${afterNames[i]}.forward(x)`;
    L.push({ text: last ? `        ${call}` : `        let x = ${call};`, blockId: b.id });
  });
  L.push({ text: "    }" });
  L.push({ text: "}" });
  return L;
}

export function rustCode(blocks: Block[], groupOf: GroupMembership, groups: BlockGroup[]): CodeLine[] {
  const plan = loopPlan(blocks, groupOf, groups);
  return plan ? rustLooped(blocks, plan) : rustFlat(blocks);
}

export function specJson(blocks: Block[]): CodeLine[] {
  const spec = { blocks: blocks.map((b) => ({ ...b })) };
  const lines: CodeLine[] = [];
  lines.push({ text: "{" });
  lines.push({ text: '  "blocks": [' });
  spec.blocks.forEach((b, i) => {
    const one = JSON.stringify(b, null, 2).split("\n");
    one.forEach((l, j) => lines.push({ text: "    " + l + (j === one.length - 1 && i < spec.blocks.length - 1 ? "," : ""), blockId: b.id }));
  });
  lines.push({ text: "  ]" });
  lines.push({ text: "}" });
  return lines;
}

export const generate = (lang: Lang, blocks: Block[], groupOf: GroupMembership, groups: BlockGroup[]) =>
  lang === "python" ? pythonCode(blocks, groupOf, groups) : lang === "rust" ? rustCode(blocks, groupOf, groups) : specJson(blocks);
