"use client";
import type { Block } from "@/lib/types";
import { demoRouterProbs, r2, r3, r4, rmsnorm, ropePair, softmax, topKGate } from "@/lib/mathDemo";
import { Example, Formula, M, Note, Section, Why } from "./pieces";

export interface LearnCtx {
  dModel: number;
  /** null when no dataset has been tokenized yet. */
  vocab: number | null;
  seqLen: number;
}

const fmtVec = (xs: number[], d = 3) => `[${xs.map((x) => x.toFixed(d)).join(", ")}]`;

// -------------------------------------------------------------- embedding

function EmbeddingLearn({ block, ctx }: { block: Extract<Block, { type: "embedding" }>; ctx: LearnCtx }) {
  const vocab = ctx.vocab ?? 320;
  const d = block.d_model;
  const std = r4(1 / Math.sqrt(d));
  const raw = [0.02, -0.15, 0.09, -0.03];
  const scale = Math.sqrt(d);
  const scaled = raw.map((x) => r2(x * scale));
  return (
    <>
      <Section title="What it is">
        <p>
          A lookup table <M>E</M> with one row per vocabulary token, <M>vocab × d_model</M> numbers in total. Looking a token up is not a computation
          at all — it’s just reading one row out of the table. Every other block in the stack works on the <M>d_model</M>-dimensional vectors this
          block produces.
        </p>
      </Section>
      <Formula label="forward">{`x = E[token_id]              # one row, shape [d_model]${block.scale ? "\nx = x * sqrt(d_model)        # rescale (this block's \"scale\" option)" : ""}`}</Formula>
      <Why>
        <p>
          The table starts with small random numbers (this backend initializes each row with standard deviation <M>1/√d_model</M>, so a row’s
          values are already close to unit length overall). {block.scale ? (
            <>
              Multiplying by <M>√d_model</M> afterwards brings each individual number up to roughly unit variance too — matching the scale later
              blocks (RMSNorm, attention) are built to expect. This is Gemma’s convention; the original 2017 Transformer didn’t do this.
            </>
          ) : (
            <>
              With scaling off, embeddings stay at that small initial scale relative to later layers — closer to the original 2017 Transformer,
              which also didn’t scale. Toggle “Scale by √d_model” on to use Gemma’s convention instead.
            </>
          )}
        </p>
      </Why>
      <Example>
        <p>Token id 42, in a vocabulary of {vocab.toLocaleString()}, with d_model = {d} (so std ≈ {std} at initialization):</p>
        <p>row 42 (4 of {d} numbers shown) ≈ <M>{fmtVec(raw)}</M></p>
        {block.scale && (
          <p>
            × √{d} = {r2(scale)} → <M>{fmtVec(scaled)}</M> — each number moved from roughly ±{std} to roughly ±1.
          </p>
        )}
      </Example>
    </>
  );
}

// ---------------------------------------------------------- positional encoding

function PosEncodingLearn({ ctx }: { ctx: LearnCtx }) {
  const d = ctx.dModel;
  const m = 5;
  const fast = { sin: r4(Math.sin(m)), cos: r4(Math.cos(m)) };
  const i = Math.floor(d / 4);
  const theta = Math.pow(10000, (2 * i) / d);
  const angle = m / theta;
  const slow = { sin: r4(Math.sin(angle)), cos: r4(Math.cos(angle)) };
  return (
    <>
      <Section title="What it is">
        <p>
          Self-attention on its own is <em>permutation-equivariant</em>: shuffle the input tokens and the output just shuffles the same way — nothing
          in the computation actually tells the model which token came first. This block fixes that by adding a unique, fixed pattern to each
          position’s embedding, so “the cat sat” and “sat the cat” stop looking identical to the network.
        </p>
      </Section>
      <Formula label="forward">{`PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))\nPE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))\nx = embedding + PE(pos)`}</Formula>
      <Why>
        <p>
          Every position gets a distinct fingerprint, and because <M>sin</M>/<M>cos</M> of a sum expands in terms of the parts, the <em>relative</em>{" "}
          offset between two positions is linearly recoverable from their two fingerprints — part of why the original 2017 Transformer chose exactly
          this form rather than, say, just using the raw position number.
        </p>
      </Why>
      <Example>
        <p>d_model = {d}, position {m}: dimension pair 0 (fastest-varying, i = 0) → <M>sin = {fast.sin}, cos = {fast.cos}</M>.</p>
        <p>dimension pair {i} (slower, deep in the vector) → <M>sin = {slow.sin}, cos = {slow.cos}</M> — barely moved yet at position {m}.</p>
        <p>Low dimensions oscillate quickly from token to token; high dimensions drift slowly — together they encode position at every timescale.</p>
      </Example>
      <Note tone="neutral">
        Most Attention blocks in this app inject position with RoPE instead (see the Self-Attention page), applied fresh inside every layer rather
        than added once up front. Using both isn’t wrong, just redundant — this block mainly matters when every Attention block has RoPE turned off
        (the “Classic (2017)” preset does exactly that).
      </Note>
    </>
  );
}

// ------------------------------------------------------------------- norm

function NormLearn({ block }: { block: Extract<Block, { type: "norm" }> }) {
  const x = [2, -4, 1, 3];
  const y = rmsnorm(x).map(r3);
  const ms = r2(x.reduce((a, v) => a + v * v, 0) / x.length);
  return (
    <>
      <Section title="What it is">
        <p>Rescales each token’s vector so its activations stay in a consistent, well-behaved range as they flow through many stacked layers.</p>
      </Section>
      {block.kind === "layer" ? (
        <Formula label="LayerNorm">{`y = (x - mean(x)) / sqrt(var(x) + eps) * γ + β`}</Formula>
      ) : (
        <Formula label="RMSNorm">{`y = x / sqrt(mean(x²) + eps) * γ`}</Formula>
      )}
      <Why>
        <p>
          Without normalization, activation scale tends to drift — grow or shrink — as it passes through more layers, which makes gradients unstable
          and training unreliable past a handful of layers.{" "}
          {block.kind === "rms" ? (
            <>
              RMSNorm skips LayerNorm’s mean-centering and bias term: recent large models (Gemma, LLaMA, gpt-oss) found that just rescaling the
              magnitude is enough for stability, and it’s cheaper to compute.
            </>
          ) : (
            <>
              LayerNorm also centers the mean and adds a learned bias <M>β</M> — the original, more expensive normalization; RMSNorm (this block’s
              other option) drops both and only rescales magnitude, which most modern LLMs now prefer.
            </>
          )}
        </p>
      </Why>
      <Example>
        <p>4-dimensional toy vector <M>x = {fmtVec(x, 0)}</M>, mean(x²) = {ms}.</p>
        <p>RMSNorm(x) ≈ <M>{fmtVec(y)}</M> — same direction, rescaled so its own root-mean-square is ≈1.</p>
      </Example>
      <Note tone="neutral">
        This is separate from the <M>pre_norm</M> setting built into every Attention/Feed-Forward block, which normalizes just that block’s own
        input. A standalone Normalization block like this one is usually placed once, right before the LM Head, for one last rescale before the
        logits are computed.
      </Note>
    </>
  );
}

// -------------------------------------------------------------- attention

function AttentionLearn({ block, ctx }: { block: Extract<Block, { type: "attention" }>; ctx: LearnCtx }) {
  const headDim = ctx.dModel / block.n_heads;
  const nRep = block.n_heads / block.n_kv_heads;
  return (
    <>
      <Section title="What it is">
        <p>
          Lets each token gather information from <em>other</em> tokens, weighted by how relevant they are, and mix that into its own vector. It’s
          what lets “it” in “the trophy didn’t fit in the suitcase because it was too big” resolve to “trophy” — directly, regardless of distance,
          unlike an RNN which has to carry that information step by step.
        </p>
      </Section>
      <Formula label="forward, per head">{`Q = x Wq,  K = x Wk,  V = x Wv        # shape [seq, ${headDim}] per head\nscores = Q Kᵀ / sqrt(head_dim)        # head_dim = d_model / n_heads = ${ctx.dModel}/${block.n_heads} = ${headDim}\nscores = mask(scores)                 # see "masking" below\nA = softmax(scores)\nout_head = A V\nout = concat(all heads) Wo${block.residual ? "\nx' = x + out                          # residual add" : ""}`}</Formula>
      <Why>
        <p>
          Dividing by <M>√head_dim</M> keeps the dot products from growing with head size (larger dot products would push softmax toward a near
          one-hot, saturated distribution before the network has learned anything useful). Multiple heads run this in parallel with separate
          weights, so different heads can specialize in different kinds of relationships (say, subject↔verb agreement in one head, coreference in
          another).
        </p>
      </Why>

      <Section title="Heads: multi-head vs. grouped-query attention (GQA)">
        {block.n_kv_heads < block.n_heads ? (
          <>
            <Formula label="GQA">{`query head i uses KV head ⌊i / ${nRep}⌋      # n_heads/n_kv_heads = ${nRep}`}</Formula>
            <Note tone="on">
              This block: {block.n_heads} query heads share {block.n_kv_heads} KV head{block.n_kv_heads === 1 ? "" : "s"} — a {nRep}:{block.n_kv_heads} ratio.
              {block.n_heads === 4 && block.n_kv_heads === 1 && " That's exactly real Gemma-3-270M's own ratio."} The K/V projections (and, at
              inference, the KV cache) shrink by {nRep}×, at a small quality cost, because nearby query heads end up sharing the same keys and
              values.
            </Note>
          </>
        ) : (
          <Note tone="off">
            KV heads ({block.n_kv_heads}) equal query heads ({block.n_heads}) here — plain multi-head attention, every head has its own K/V. The
            most expressive option, and the most memory-hungry. Lower “KV heads” below “Heads” to turn this into GQA.
          </Note>
        )}
      </Section>

      <Section title="QK-norm">
        {block.qk_norm ? (
          <>
            <Formula>{`q = RMSNorm(q);  k = RMSNorm(k)     # per head, before RoPE`}</Formula>
            <Note tone="on">
              On for this block. Keeps Q·K dot products from growing unboundedly as training progresses — left unchecked, that drift can push
              softmax toward a near one-hot distribution and destabilize training at depth. A Gemma-3-specific detail most earlier LLaMA-family
              models skip.
            </Note>
          </>
        ) : (
          <Note tone="off">Off for this block — Q and K go straight from their projections into the dot product, unnormalized.</Note>
        )}
      </Section>

      <Section title="Position: RoPE">
        {block.rope ? (
          (() => {
            const fast = ropePair(block.rope_theta, headDim, 0, 5);
            const slowI = Math.max(1, Math.floor(headDim / 4));
            const slow = ropePair(block.rope_theta, headDim, slowI, 5);
            return (
              <>
                <Formula label="rotate each (x₂ᵢ, x₂ᵢ₊₁) pair at position m">{`θᵢ = rope_theta ^ (-2i / head_dim)\nx'₂ᵢ   = x₂ᵢ cos(mθᵢ) − x₂ᵢ₊₁ sin(mθᵢ)\nx'₂ᵢ₊₁ = x₂ᵢ sin(mθᵢ) + x₂ᵢ₊₁ cos(mθᵢ)`}</Formula>
                <p>
                  Rotating both Q and K this way means <M>Q_m · K_n</M> after rotation depends only on the distance <M>m−n</M>, not on <M>m</M> and{" "}
                  <M>n</M> separately — attention becomes relative-position-aware for free, with nothing added to the residual stream (unlike the
                  Positional Encoding block, which adds position once, up front).
                </p>
                <Example>
                  <p>
                    head_dim = {headDim}, rope_theta = {block.rope_theta.toLocaleString()}, position m = 5:
                  </p>
                  <p>
                    pair <M>i=0</M>: θ₀ = rope_theta⁰ = <M>1</M> (always exactly 1, whatever rope_theta is) → angle = 5 × 1 ={" "}
                    <M>{fast.angleRad} rad</M> ({fast.angleDeg}°) → <M>cos = {fast.cos}, sin = {fast.sin}</M>. <em>The first pair always turns 1
                    radian per position.</em>
                  </p>
                  <p>
                    pair <M>i={slowI}</M>, deeper in the vector: θ{slowI} = rope_theta^(−{slowI * 2}/{headDim}) = <M>{slow.theta}</M> → angle = 5 ×{" "}
                    {slow.theta} = <M>{slow.angleRad} rad</M> ({slow.angleDeg}°) — barely moved. A larger rope_theta makes deep pairs like this turn
                    even more slowly, which is what lets the model resolve much longer-range position differences.
                  </p>
                </Example>
              </>
            );
          })()
        ) : (
          <Note tone="off">
            Off for this block — with no RoPE and no separate Positional Encoding block earlier in the stack, this attention has no notion of token
            order at all.
          </Note>
        )}
      </Section>

      <Section title="Masking: what a token is allowed to see">
        <Formula>{`allowed(i, j) = ${block.causal ? "j ≤ i" : "true"}${block.window > 0 ? ` AND (i − j < ${block.window})` : ""}`}</Formula>
        {!block.causal && (
          <Note tone="off">
            Causal is off: token i can see tokens after it, including the one it’s meant to predict. This app’s own validation warns about this —
            for a next-token model, the loss will collapse toward trivially copying the answer instead of learning to predict it. (Off is the right
            choice for a BERT-style encoder that never generates text — just not for this app’s decoder.)
          </Note>
        )}
        {block.window > 0 ? (
          <Example>
            <p>
              window = {block.window}: causal already blocks the future; the window then also blocks anything more than {block.window} steps in
              the past. Token 50 sees only tokens {Math.max(0, 50 - block.window + 1)}–50 ({block.window} tokens) instead of all 50 before it.
            </p>
            <p>
              Cost drops from <M>O(seq²)</M> to <M>O(seq · {block.window})</M>. Most Gemma-3/gpt-oss layers use a window like this; a few “global”
              layers (window = 0) are kept in the mix for long-range recall.
            </p>
          </Example>
        ) : (
          <Note tone="off">window = 0: no limit — this block attends across the entire past (full causal attention).</Note>
        )}
      </Section>

      <Section title="Attention sink">
        {block.sink ? (
          <>
            <Formula>{`scores' = concat(scores, sink_logit)   # one learned scalar per head\nA' = softmax(scores')\nout_head = A'[:, :-1] @ V              # the sink's own share of A' is discarded`}</Formula>
            <Why>
              <p>
                When a sliding window leaves very few real keys visible (e.g. right after the start of a sequence), softmax is forced to dump 100%
                of its probability onto whatever few tokens exist — often an over-confident, unhelpful distribution. The sink gives every head a
                harmless “attend to nothing” option that can soak up that leftover probability instead. From StreamingLLM, adopted by gpt-oss.
              </p>
            </Why>
            <Note tone="on">
              The sink logit starts at exactly 0 (so at initialization it behaves like any other fresh, small Q·K score) — training decides how much
              sink mass each head actually needs. Watch it live per head as the “sink %” badge in the Observe tab’s attention maps.
            </Note>
          </>
        ) : (
          <Note tone="off">Off for this block — no sink slot; softmax must distribute its full probability over real, visible tokens only.</Note>
        )}
      </Section>

      <Section title="Pre-norm and residual">
        <p>
          <M>pre_norm = {block.pre_norm}</M> normalizes this block’s <em>input</em> (not its output) before Q/K/V are computed — see the
          Normalization block’s page for the formula and why. {block.residual ? (
            <>
              <M>residual</M> is on: the block adds its own output back onto its input (<M>x’ = x + Attention(...)</M>) rather than replacing it.
              Without this, gradients have to flow through every sub-layer in sequence instead of having a direct path backward — hard to train past
              a handful of layers.
            </>
          ) : (
            <>
              <M>residual</M> is off here: this block’s output <em>replaces</em> its input outright, with no shortcut path for gradients to skip
              around it.
            </>
          )}
        </p>
      </Section>
    </>
  );
}

// -------------------------------------------------------------------- ffn

function FfnLearn({ block, ctx }: { block: Extract<Block, { type: "ffn" }>; ctx: LearnCtx }) {
  return (
    <>
      <Section title="What it is">
        <p>
          After attention mixes information <em>across</em> tokens, the feed-forward block processes <em>each</em> token independently through a
          small per-token network. This is where most of a transformer’s raw computation — often most of its parameters — actually lives.
        </p>
      </Section>

      {block.kind === "mlp" && (
        <>
          <Formula label="plain MLP">{`h = ${block.activation}(x W_up)      # d_model → d_ff = ${ctx.dModel} → ${block.d_ff}\ny = h W_down                # d_ff → d_model`}</Formula>
          <Why>
            <p>
              The simplest way to give each token a wider, nonlinear per-token transformation than one linear layer could. A wider hidden size (
              <M>d_ff</M>) generally means more capacity per token, at the cost of more compute — this is the original 2017 Transformer’s design.
            </p>
          </Why>
        </>
      )}

      {block.kind === "glu" &&
        (() => {
          const up = [2.0, -0.5];
          const gateAct = [0.8, 0.1];
          const h = up.map((v, i) => r3(v * gateAct[i]));
          return (
            <>
              <Formula label="gated GLU / GeGLU / SwiGLU">{`gate = ${block.activation}(x W_gate)\nup   = x W_up\nh    = gate ⊙ up            # elementwise multiply\ny    = h W_down`}</Formula>
              <Why>
                <p>
                  The gate branch learns, per feature and per token, how much of the <M>up</M> branch to let through — a learned multiplicative
                  switch. This consistently trains better than a plain MLP at equal compute, and is what PaLM, LLaMA, Gemma and gpt-oss all actually
                  use.
                </p>
              </Why>
              <Example>
                <p>
                  For one feature: <M>up = {up[0]}</M>, <M>{block.activation}(gate) = {gateAct[0]}</M> → <M>h = {h[0]}</M>. For another where the gate
                  is nearly shut: <M>up = {up[1]}</M>, <M>{block.activation}(gate) = {gateAct[1]}</M> → <M>h = {h[1]}</M> — almost entirely
                  suppressed, even though its “up” value wasn’t small.
                </p>
              </Example>
            </>
          );
        })()}

      {block.kind === "moe" &&
        (() => {
          const p = demoRouterProbs(block.n_experts).map(r3);
          const g = topKGate(p, block.top_k).map(r3);
          return (
            <>
              <Formula label="mixture of experts">{`p = softmax(router(x))          # one probability per expert, over ${block.n_experts}\ng = top_${block.top_k}(p), renormalized to sum to 1\ny = Σₑ gₑ · Expertₑ(x)          # each Expertₑ is its own GLU block`}</Formula>
              <Why>
                <p>
                  Gives the model far more total parameters ({block.n_experts} separate feed-forward blocks) without paying for all of them on every
                  token — only the top {block.top_k} experts actually contribute to any given token’s output, so compute stays proportional to top_k,
                  not n_experts.
                </p>
              </Why>
              <Example>
                <p>Router output for one token, {block.n_experts} experts:</p>
                <p><M>p = {fmtVec(p, 2)}</M></p>
                <p>
                  keep the top {block.top_k}, renormalize → <M>g = {fmtVec(g, 2)}</M> (the rest become exactly 0 and contribute nothing to this
                  token).
                </p>
              </Example>
              <Note tone="neutral">
                This app computes every expert for every token and multiplies the unselected ones by 0, rather than truly skipping them — correct
                routing behaviour (which experts get used, how balanced the load is, visible live in Observe), just without the compute savings real
                sparse MoE serving gets.
              </Note>
            </>
          );
        })()}

      <Section title="Pre-norm and residual">
        <p>
          Same idea as in Self-Attention: <M>pre_norm = {block.pre_norm}</M> normalizes the block’s input before it runs (see the Normalization
          block’s page), and {block.residual ? (
            <>residual is on — the block’s output is added back onto its input rather than replacing it, giving gradients a direct shortcut path.</>
          ) : (
            <>residual is off — this block’s output replaces its input outright.</>
          )}
        </p>
      </Section>
    </>
  );
}

// --------------------------------------------------------------- dropout

function DropoutLearn({ block }: { block: Extract<Block, { type: "dropout" }> }) {
  const keep = r2(1 - block.p);
  const scale = r3(1 / (1 - block.p || 1));
  return (
    <>
      <Section title="What it is">
        <p>A regularizer: while training, it randomly zeroes some fraction of the activations passing through it, on every step.</p>
      </Section>
      <Formula label="forward, during training only">{`keep = Bernoulli(${keep})   # p = ${block.p} probability of zeroing\ny = (x * keep) / ${keep}          # rescale survivors so the average is unchanged`}</Formula>
      <Why>
        <p>
          Forces the network to not rely too heavily on any single connection, since that connection might vanish on a random step — a classic
          defense against overfitting, especially useful when the dataset is small relative to the model (exactly the “Overfitting” warning this
          app’s own training card can show).
        </p>
      </Why>
      <Example>
        <p>
          p = {block.p}: an activation of <M>1.0</M> that survives becomes <M>1.0 / {keep} = {scale}</M>; on average {r2(block.p * 100)}% of
          activations become exactly <M>0</M> on any given training step, independently, for every token.
        </p>
      </Example>
      <Note tone="on">
        Verified in this backend’s code: burn’s <M>Dropout</M> only activates on an autodiff (training) backend. The Observe tab’s live preview and
        the Generate box both run the model on a plain, non-autodiff backend — so dropout is always a no-op there. You only ever see the
        deterministic network on the right; only real training steps actually drop anything.
      </Note>
    </>
  );
}

// ------------------------------------------------------------------- head

function HeadLearn({ block, ctx }: { block: Extract<Block, { type: "head" }>; ctx: LearnCtx }) {
  const vocab = ctx.vocab ?? 320;
  const logits = [2.1, 0.4, -1.0];
  const probs = softmax(logits).map(r4);
  const saved = vocab * ctx.dModel;
  return (
    <>
      <Section title="What it is">
        <p>
          The very last step: turns the final per-token hidden vector into one score (a “logit”) for every possible next token, so the network can
          say how likely each one is.
        </p>
      </Section>
      <Formula label="forward">{block.tied ? `logits = h @ Eᵀ          # reuses the Embedding block's own matrix\nprobs  = softmax(logits)` : `logits = h @ W_head      # a separate d_model × vocab matrix\nprobs  = softmax(logits)`}</Formula>
      <Why>
        {block.tied ? (
          <p>
            Tying reuses the embedding table instead of learning a second, separate <M>d_model × vocab</M> matrix — an exact parameter saving (see
            below) — and nudges the input and output “meaning spaces” to line up: tokens that behave alike as <em>inputs</em> tend to end up
            predicted alike as <em>outputs</em> too. A mild but real regularizer used by GPT-2, Gemma and most modern open LLMs.
          </p>
        ) : (
          <p>
            An untied head learns its own separate projection instead of reusing the Embedding block’s matrix — strictly more parameters and more
            freedom, at the cost of the parameter savings and the input/output regularization tying gives you. Turn “Tie with embedding” on to use
            Gemma’s convention instead.
          </p>
        )}
      </Why>
      <Example>
        <p>Three example logits (of {vocab.toLocaleString()}): <M>{fmtVec(logits, 1)}</M> → softmax → <M>{fmtVec(probs, 3)}</M> (sums to 1).</p>
        {block.tied && (
          <p>
            With vocab = {vocab.toLocaleString()} and d_model = {ctx.dModel}, tying saves exactly{" "}
            <M>{saved.toLocaleString()}</M> parameters — the size of the head that would otherwise have to be learned separately.
          </p>
        )}
      </Example>
    </>
  );
}

export function BlockLearn({ block, ctx }: { block: Block; ctx: LearnCtx }) {
  switch (block.type) {
    case "embedding":
      return <EmbeddingLearn block={block} ctx={ctx} />;
    case "pos_encoding":
      return <PosEncodingLearn ctx={ctx} />;
    case "norm":
      return <NormLearn block={block} />;
    case "attention":
      return <AttentionLearn block={block} ctx={ctx} />;
    case "ffn":
      return <FfnLearn block={block} ctx={ctx} />;
    case "dropout":
      return <DropoutLearn block={block} />;
    case "head":
      return <HeadLearn block={block} ctx={ctx} />;
  }
}
