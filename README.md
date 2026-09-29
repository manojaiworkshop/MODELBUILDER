# ModelBuilder

Build a transformer by dragging blocks, train it live, and watch what happens inside.

- **Left – build.** Drag blocks (Embedding, Attention, Feed-Forward, …) into a stack, edit their parameters, and see the equivalent Python / Rust code update as you go. Pick or upload training text and set hyper-parameters.
- **Right – observe.** Your sentence becomes tokens → ids → embedding vectors; then a live loss curve, per-block activations, attention maps per head, mixture-of-experts routing, gradient flow, an embedding-space map, and text samples that improve as it trains.

```
frontend/   Next.js 16 · React 19 · Tailwind 4 · dnd-kit · socket.io-client · zustand
backend/    Rust · axum · socketioxide (Socket.IO) · burn (tensors + autograd, CPU)
```

## Run it

Requirements: Rust (stable) and Node 20+.

```bash
./dev.sh                      # builds the backend, then runs backend :4000 + frontend :3000
```

or in two terminals:

```bash
cd backend  && cargo run --release        # http://127.0.0.1:4000
cd frontend && npm install && npm run dev # http://localhost:3000
```

Open <http://localhost:3000>. The pill in the top bar shows the backend connection.

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `PORT` / `HOST` | backend | `4000` / `127.0.0.1` | listen address (use `HOST=0.0.0.0` to expose on the network) |
| `MAX_TRAININGS` | backend | `2` | concurrent training runs across all users |
| `CORS_ORIGIN` | backend | any | comma-separated allowed browser origins |
| `NEXT_PUBLIC_BACKEND_URL` | frontend | `http://<page host>:4000` | where the browser connects (set at build time) |

## Training on your own files

Drop files or whole folders onto the **Data** tab (or use *Choose files / folder*). Files are streamed over HTTP straight to disk (never held in browser or server memory), the text is extracted, and the tokenized dataset is rebuilt in the background with live progress. Each file can be switched on/off or removed; the dataset rebuilds itself.

| Format | Notes |
|---|---|
| Plain text, code, CSV/TSV, JSON, logs, Markdown | any encoding: UTF-8, UTF-16, Latin-1/Windows-1252 |
| PDF | text layer only. PDFs with a damaged index (common after line-ending conversion or truncation) are repaired automatically; scanned/image PDFs have no text (no OCR) and are refused |
| Word `.docx` | paragraphs, tables (tab-separated rows), footnotes |
| PowerPoint `.pptx` | slide text + speaker notes, in slide order |
| Excel `.xlsx .xlsm .xlsb .xls`, `.ods` | every sheet, tab-separated rows |
| OpenDocument `.odt .odp`, `.epub`, `.rtf`, HTML | scripts/styles/comments dropped |
| `.zip`, `.gz` | unpacked recursively (nested archives, mixed contents); unreadable entries are listed and skipped |
| Legacy `.doc` / `.ppt`, images, audio, video, binaries | **not supported** — refused with a reason (re-save as `.docx`/`.pptx`, or export to PDF/TXT) |

The format is detected from the file *content*, not its name, so a `.docx` renamed to `.bin` still works.

How it scales: text is extracted to disk; the BPE vocabulary is learned from a ~48 MB sample spread evenly across all files; the corpus is then encoded in parallel 1 MB chunks into a compact `u16` array (up to 500 M tokens ≈ 1 GB of RAM; beyond that only the beginning is used, and the UI says so). Training samples random windows across the whole array.

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_BYTES` | 1 GiB | largest single file |
| `MAX_SESSION_BYTES` | 4 GiB | disk one browser tab may use (uploads + extracted text) |
| `MAX_DOC_TEXT_BYTES` | 2 GiB | cap on text extracted from one file |
| `MAX_TOKENS` | 500 000 000 | tokens kept for training |
| `MAX_EXTRACTIONS` / `MAX_BUILDS` | 2 / 2 | concurrent extractions / dataset builds |
| `DATA_DIR` | system temp `/modelbuilder` | scratch space; wiped at startup and when a tab disconnects |

Formats that must be loaded whole (PDF, Excel, HTML, RTF) are limited to 512 MB per file. Files are only kept while the tab is open.

## What the blocks are

The backend's block library is taken from the `transformer_rs` project (`~/Documents/NLPWORLD/RUST_TRANFORMER/transformer_rs`):

| Block | Options | Origin |
|---|---|---|
| Embedding | `d_model`, √d scaling | Gemma-style embedding, init std 1/√d |
| Positional Encoding | – | sinusoidal (the ATIS encoder) |
| Self-Attention | heads, KV heads (GQA), QK-norm, RoPE (+base), sliding window, attention **sink**, causal, pre-norm, residual, dropout | `slm::model` (Gemma) + `gptoss::model` (sink) |
| Feed-Forward | gated GLU (GeGLU/SwiGLU), plain MLP, **mixture of experts** (n experts, top-k) | `GemmaMlp`, `GptOssMoeMlp` |
| Normalization | RMSNorm / LayerNorm | – |
| Dropout | p | – |
| LM Head | tied to embedding, or separate | weight tying from `slm::model` |

Tokenizer: byte-level BPE (`slm::bpe`) with GPT-2-style pre-tokenization, trained on your text. Text is split into words first (a word keeps its leading space, e.g. ` thy`) and merges never cross word boundaries, so tokens are word parts rather than byte glue. `merges = 0` is plain bytes; the default of 200 turns frequent words into single tokens. Words seen only once stay split into pieces, and merging stops when no letter pair repeats.

Any order/number of blocks works as long as the stack starts with an Embedding and ends with an LM Head. Bad stacks are explained live (e.g. *d_model must be divisible by n_heads*, *causal off = the model can see its answer*). Try the presets: Tiny GPT, Gemma-mini, GPT-OSS-mini, Classic (2017), and a no-attention **Bigram** baseline to compare loss against.

## Socket.IO protocol

All client→server events use acks. Payloads are camelCase JSON.

| Event | Payload → reply |
|---|---|
| `data:set` | `{text?, merges, usePasted}` → `{ok}`; the dataset is rebuilt in the background |
| `data:remove` / `data:toggle` | `{id}` / `{id, enabled}` |
| `PUT /api/upload?sid=<socket id>&name=<file>` | HTTP, raw body, streamed to disk → `{ok, id}` |
| `model:inspect` | `{spec:{blocks}, probe, seqLen}` → validation issues, shapes, param counts, and (if valid) per-block traces for the probe sentence |
| `train:start` / `train:stop` | `{spec, hyper, probe}` → `{ok, error?}` |
| `probe:set` | `{text}` – change the traced sentence mid-run |
| `generate` / `generate:stop` | `{prompt, maxNew, temperature, topK}` → `{ok, text}` (tokens stream as `generate:token`) |

Server pushes while preparing data: `data:docs` (document list + extraction status), `data:progress`, `data:info` (tokens, vocabulary, source size).

Server pushes during a run: `train:status`, `train:steps` (batched loss/lr/speed), `train:eval` (validation loss), `train:inspect` (full network snapshot with gradient norms), `train:sample`.

The spec is exactly what the **Spec** tab of the live code panel shows.

## Limits (enforced by the backend)

Pasted text ≤ 2 MB (files: see above) · ≤ 40 blocks · `d_model` ≤ 512 · ≤ 25 M parameters · context ≤ 256 · batch ≤ 64 · ≤ 100 000 steps · one run per browser tab.

## Tests

```bash
cd backend  && cargo test --release     # 31 tests, incl. real Word/PDF/Excel/PowerPoint fixtures
cd frontend && npx tsc --noEmit && npm run lint && npm run build
```

Backend tests check, among other things, that the analytic parameter count equals burn's exact count for every block type, that changing a later token never changes earlier logits (causality), that sliding-window/sink masks and top-k routing behave as specified, and that a small Gemma+MoE+sink model's loss actually falls.

## Notes

- Training runs on the CPU (burn `ndarray`), roughly 17 steps/s for the presets on a desktop CPU. The model code is backend-generic, so a GPU backend is a feature-flag away.
- Everything is dense and simple on purpose: attention is recomputed without a KV cache, and all MoE experts run for every token (non-selected ones get weight 0) — matching the reference project.
- The Python / Rust panels are a read-only, illustrative view of the stack; the JSON spec is what actually runs.
# MODELBUILDER
