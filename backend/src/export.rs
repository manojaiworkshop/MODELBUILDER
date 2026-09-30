//! Downloadable model bundle laid out like a Hugging Face model repository:
//!
//! ```text
//! config.json  generation_config.json  model.safetensors
//! tokenizer.json  tokenizer_config.json  special_tokens_map.json
//! chat_template.jinja (only when trained on chat-style data)  README.md
//! ```
//!
//! The tokenizer is a standard `tokenizers` file, so `AutoTokenizer.from_pretrained` works on the
//! folder: the built-in byte-level BPE written out, or the imported `tokenizer.json` unchanged (then
//! `token_map.json` says which tokenizer id each embedding row belongs to). The network is a custom block stack rather than one of the built-in
//! `transformers` architectures: `config.json` carries the full block list and `model.safetensors`
//! the weights under Llama/Gemma-style tensor names, which is everything needed to re-implement it.

use crate::hfdata::Style;
use crate::model::NamedTensor;
use crate::spec::{BlockKind, FfnKind, NormKind};
use crate::tokenizer::{Bpe, BOS_ID, EOS_ID, FIRST_MERGE_ID, PAD_ID, UNK_ID};
use crate::train::Trained;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::Write;

/// GPT-2's byte -> printable character table, which byte-level BPE vocabularies are written in.
fn byte_char(b: u8) -> char {
    let printable = |x: u32| (33..=126).contains(&x) || (161..=172).contains(&x) || (174..=255).contains(&x);
    if printable(b as u32) {
        return char::from_u32(b as u32).unwrap();
    }
    // the remaining bytes are assigned 256, 257, … in byte order
    let n = (0..b).filter(|&x| !printable(x as u32)).count() as u32;
    char::from_u32(256 + n).unwrap()
}

/// `tokenizer.json` for `bpe`, plus how many merges had to get a placeholder token because they
/// spell the same string as an earlier merge (the format keys tokens by string).
pub(crate) fn tokenizer_json(bpe: &Bpe) -> (Value, usize) {
    let mut text: Vec<String> = (0..=255u8).map(|b| byte_char(b).to_string()).collect();
    let specials = [(PAD_ID, "<pad>"), (UNK_ID, "<unk>"), (BOS_ID, "<bos>"), (EOS_ID, "<eos>")];
    let mut vocab = Map::new();
    let mut seen: HashMap<String, u32> = HashMap::new();
    for (id, t) in text.iter().enumerate() {
        vocab.insert(t.clone(), json!(id));
        seen.insert(t.clone(), id as u32);
    }
    for (id, s) in specials {
        vocab.insert(s.to_string(), json!(id));
        text.push(s.to_string());
    }
    let mut merges = Vec::new();
    let mut dropped = 0;
    for (i, &(a, b)) in bpe.merges().iter().enumerate() {
        let id = FIRST_MERGE_ID + i as u32;
        let s = format!("{}{}", text[a as usize], text[b as usize]);
        if seen.contains_key(&s) {
            dropped += 1;
            vocab.insert(format!("<dup_{id}>"), json!(id));
        } else {
            merges.push(format!("{} {}", text[a as usize], text[b as usize]));
            vocab.insert(s.clone(), json!(id));
            seen.insert(s.clone(), id);
        }
        text.push(s);
    }
    let added: Vec<Value> = specials
        .iter()
        .map(|(id, s)| json!({ "id": id, "content": s, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }))
        .collect();
    let v = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true },
        "post_processor": null,
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges,
        },
    });
    (v, dropped)
}

/// The safetensors container: 8-byte header length, JSON header, then the raw little-endian data.
pub fn safetensors(tensors: &[NamedTensor]) -> Vec<u8> {
    let mut header = Map::new();
    header.insert("__metadata__".into(), json!({ "format": "pt" }));
    let mut offset = 0usize;
    for (name, shape, data) in tensors {
        let end = offset + data.len() * 4;
        header.insert(name.clone(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [offset, end] }));
        offset = end;
    }
    let mut h = serde_json::to_vec(&Value::Object(header)).unwrap();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + offset);
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    for (_, _, data) in tensors {
        for x in data {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// Jinja template reproducing the chat layout the model was trained on (`<|user|>` turns).
const CHAT_TEMPLATE: &str = "{%- for message in messages -%}\n\
{%- if message['role'] == 'system' -%}{{ '<|system|>\\n' + message['content'] + '\\n' }}\n\
{%- elif message['role'] == 'user' -%}{{ '<|user|>\\n' + message['content'] + '\\n' }}\n\
{%- elif message['role'] == 'assistant' -%}{{ '<|assistant|>\\n' + message['content'] + '\\n\\n' }}\n\
{%- endif -%}\n\
{%- endfor -%}\n\
{%- if add_generation_prompt -%}{{ '<|assistant|>\\n' }}{%- endif -%}\n";

fn config_json(t: &Trained, vocab: usize, params: usize) -> Value {
    let blocks: Vec<Value> = t.spec.blocks.iter().map(|b| serde_json::to_value(&b.kind).unwrap_or(Value::Null)).collect();
    let d = t.spec.d_model().unwrap_or(0);
    let mut c = json!({
        "architectures": ["ModelBuilderForCausalLM"],
        "model_type": "modelbuilder",
        "vocab_size": vocab,
        "hidden_size": d,
        "max_position_embeddings": t.seq_len,
        "tie_word_embeddings": t.spec.blocks.iter().any(|b| matches!(b.kind, BlockKind::Head { tied: true })),
        "scale_embeddings": t.spec.blocks.iter().any(|b| matches!(b.kind, BlockKind::Embedding { scale: true, .. })),
        "pad_token_id": PAD_ID, "bos_token_id": BOS_ID, "eos_token_id": EOS_ID,
        "torch_dtype": "float32",
        "num_parameters": params,
        "blocks": blocks,
    });
    let o = c.as_object_mut().unwrap();
    if let Some(imp) = t.tok.bpe.imported() {
        // ids belong to the imported tokenizer, and the embedding only has rows for the ones the data used
        for k in ["pad_token_id", "bos_token_id", "eos_token_id"] {
            o.remove(k);
        }
        o.insert("original_vocab_size".into(), json!(imp.tok.full_vocab));
        o.insert("token_map".into(), json!("token_map.json"));
    }
    // the usual summary fields, taken from the first attention / feed-forward block
    let n_attn = t.spec.blocks.iter().filter(|b| matches!(b.kind, BlockKind::Attention { .. })).count();
    o.insert("num_hidden_layers".into(), json!(n_attn));
    if let Some(BlockKind::Attention { n_heads, n_kv_heads, rope_theta, .. }) = t.spec.blocks.iter().map(|b| &b.kind).find(|k| matches!(k, BlockKind::Attention { .. })) {
        o.insert("num_attention_heads".into(), json!(n_heads));
        o.insert("num_key_value_heads".into(), json!(n_kv_heads));
        o.insert("head_dim".into(), json!(if *n_heads > 0 { d / n_heads } else { 0 }));
        o.insert("rope_theta".into(), json!(rope_theta));
    }
    if let Some(BlockKind::Ffn { d_ff, .. }) = t.spec.blocks.iter().map(|b| &b.kind).find(|k| matches!(k, BlockKind::Ffn { .. })) {
        o.insert("intermediate_size".into(), json!(d_ff));
    }
    if let Some(BlockKind::Ffn { n_experts, top_k, .. }) = t.spec.blocks.iter().map(|b| &b.kind).find(|k| matches!(k, BlockKind::Ffn { kind: FfnKind::Moe, .. })) {
        o.insert("num_local_experts".into(), json!(n_experts));
        o.insert("num_experts_per_tok".into(), json!(top_k));
    }
    c
}

fn describe_block(k: &BlockKind) -> String {
    let norm = |n: &NormKind| match n {
        NormKind::None => "none",
        NormKind::Rms => "RMSNorm",
        NormKind::Layer => "LayerNorm",
    };
    match k {
        BlockKind::Embedding { d_model, scale } => format!("Embedding (d_model {d_model}{})", if *scale { ", scaled by √d" } else { "" }),
        BlockKind::PosEncoding => "Positional encoding (sinusoidal)".into(),
        BlockKind::Norm { kind } => format!("Normalization ({})", norm(kind)),
        BlockKind::Attention { n_heads, n_kv_heads, rope, window, sink, causal, pre_norm, .. } => format!(
            "Self-attention ({n_heads} heads, {n_kv_heads} KV heads{}{}{}{}, pre-norm {})",
            if *rope { ", RoPE" } else { "" },
            if *window > 0 { format!(", window {window}") } else { String::new() },
            if *sink { ", sink" } else { "" },
            if *causal { ", causal" } else { ", bidirectional" },
            norm(pre_norm)
        ),
        BlockKind::Ffn { kind, d_ff, n_experts, top_k, pre_norm, .. } => match kind {
            FfnKind::Glu => format!("Feed-forward (gated GLU, d_ff {d_ff}, pre-norm {})", norm(pre_norm)),
            FfnKind::Mlp => format!("Feed-forward (MLP, d_ff {d_ff}, pre-norm {})", norm(pre_norm)),
            FfnKind::Moe => format!("Feed-forward (mixture of experts, {n_experts} experts, top-{top_k}, d_ff {d_ff}, pre-norm {})", norm(pre_norm)),
        },
        BlockKind::Dropout { p } => format!("Dropout (p {p})"),
        BlockKind::Head { tied } => format!("LM head ({})", if *tied { "tied to the embedding" } else { "separate weights" }),
    }
}

fn prompt_examples(t: &Trained) -> String {
    let p = &t.provenance;
    if p.source == "text" || p.styles.is_empty() {
        return "Trained on plain text: give it the beginning of a passage and it continues it.\n".into();
    }
    let qa = p.source == "qa";
    let mut out = String::from("Prompt the model in the layout it was trained on, ending where its answer begins:\n\n");
    let sys = if p.system.is_empty() { String::new() } else { format!("<|system|>\n{}\n", p.system) };
    for style in &p.styles {
        let chat_q = format!("{sys}<|user|>\n<question>\n<|assistant|>\n");
        let chat_s = format!("{sys}<|user|>\nSummarize the following text:\n\n<text>\n<|assistant|>\n");
        let (name, body) = match (style, qa) {
            (Style::Hf, true) => ("Hugging Face / SQuAD", "Context: <page text>\nQuestion: <question>\nAnswer:"),
            (Style::Hf, false) => ("Hugging Face / CNN-DailyMail", "Article: <text>\nHighlights:"),
            (Style::Instruction, true) => ("Instruction (Alpaca)", "### Instruction:\n<question>\n\n### Input:\n<context>\n\n### Response:\n"),
            (Style::Instruction, false) => ("Instruction (Alpaca)", "### Instruction:\nSummarize the following text.\n\n### Input:\n<text>\n\n### Response:\n"),
            (Style::Chat, true) => ("Chat", chat_q.as_str()),
            (Style::Chat, false) => ("Chat", chat_s.as_str()),
            (_, true) => ("Simple", "Question: <question>\nAnswer:"),
            (_, false) => ("Simple", "Text: <text>\nSummary:"),
        };
        out.push_str(&format!("**{name}**\n\n```text\n{body}\n```\n\n"));
    }
    if p.styles.contains(&Style::Chat) {
        out.push_str("`chat_template.jinja` renders `messages` in the chat layout: `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.\n");
    }
    out
}

fn readme(t: &Trained, params: usize, dropped: usize) -> String {
    let m = *t.metrics.lock().unwrap();
    let vocab = t.tok.bpe.vocab_size();
    let fmt = |x: Option<f32>| x.map(|v| format!("{v:.4}")).unwrap_or_else(|| "n/a".into());
    let ppl = m.val_loss.map(|v| format!(" (perplexity {:.1})", v.exp().min(1e6))).unwrap_or_default();
    let data = match t.provenance.source.as_str() {
        "qa" => format!("{} generated question-answer pairs", t.provenance.examples),
        "summary" => format!("{} generated page summaries", t.provenance.examples),
        _ => format!("plain text ({} characters)", t.provenance.examples),
    };
    let vocab_line = match t.tok.bpe.imported() {
        Some(c) => format!("{vocab} token ids, the ones your data used out of the {} in `{}` ({} tokenizer)", c.tok.full_vocab, c.tok.name, c.tok.model),
        None => format!("{vocab} (byte-level BPE, {} merges)", t.tok.bpe.n_merges()),
    };
    let token_ids = if t.tok.bpe.imported().is_some() {
        "\n## Token ids\n\nThe model was trained on an imported tokenizer, keeping only the token ids that occur in the training data (a from-scratch model has no use for an embedding row per id of a 30k-260k vocabulary). Embedding row `i` belongs to tokenizer id `token_map.json[\"tokenizer_ids\"][i]`. To run the model, encode with `tokenizer.json`, map each id to its row (ids not listed were never seen in training), and map sampled rows back the same way.\n".to_string()
    } else {
        String::new()
    };
    let blocks: String = t.spec.blocks.iter().enumerate().map(|(i, b)| format!("{}. {}\n", i + 1, describe_block(&b.kind))).collect();
    let dup = if dropped > 0 { format!("\n{dropped} BPE merge(s) spell the same string as an earlier one; they keep their ids in the weights but appear in `tokenizer.json` as `<dup_N>` placeholders the tokenizer never produces.\n") } else { String::new() };
    format!(
        r#"---
library_name: modelbuilder
pipeline_tag: text-generation
tags:
- modelbuilder
- transformer
- from-scratch
---

# ModelBuilder model

A small language model built block by block in **ModelBuilder** and trained from scratch on the CPU.

- **Parameters:** {params} (float32)
- **Vocabulary:** {vocab_line}
- **Context length:** {seq}
- **Trained on:** {data}
- **Training:** {step} of {steps} steps, batch size {batch}, learning rate {lr}, AdamW (weight decay {wd}, gradient clip {clip}), seed {seed}
- **Final loss:** train {train}, validation {val}{ppl}

## Architecture

{blocks}
## Files

| File | What it is |
|---|---|
| `config.json` | Model config, including the full `blocks` list needed to rebuild the network |
| `model.safetensors` | The weights (float32) |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | The tokenizer in the standard `tokenizers` format (your imported `tokenizer.json` is included unchanged) |
| `generation_config.json` | Default sampling settings |

## Using it

The tokenizer loads directly:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./")
ids = tok("Hello world")["input_ids"]
```

The network is a custom stack of blocks, not one of the built-in `transformers` architectures, so `AutoModel` cannot build it by itself. The weights load with `safetensors`:

```python
from safetensors.torch import load_file
weights = load_file("model.safetensors")
```

Tensor names follow `model.embed_tokens.weight`, `model.blocks.<block number>.self_attn.q_proj.weight`, `model.blocks.<n>.mlp.gate_proj.weight`, `model.blocks.<n>.mlp.experts.<e>.down_proj.weight`, `lm_head.weight` (only when the head is not tied). Linear weights are stored as `[out, in]` like `torch.nn.Linear`. Block numbers are positions in the block list above, starting at 0. Sinusoidal positions and RoPE are computed, not stored.

## Prompt format

{prompts}{token_ids}{dup}"#,
        params = params,
        vocab_line = vocab_line,
        seq = t.seq_len,
        data = data,
        step = t.step,
        steps = t.hyper.steps,
        batch = t.hyper.batch_size,
        lr = t.hyper.lr,
        wd = t.hyper.weight_decay,
        clip = t.hyper.grad_clip,
        seed = t.hyper.seed,
        train = fmt(m.train_loss),
        val = fmt(m.val_loss),
        ppl = ppl,
        blocks = blocks,
        prompts = prompt_examples(t),
        token_ids = token_ids,
        dup = dup,
    )
}

/// The whole bundle as a zip, every file under a `modelbuilder-model/` folder.
pub fn bundle(t: &Trained) -> Result<Vec<u8>, String> {
    let tensors = t.model.named_tensors();
    let params: usize = tensors.iter().map(|(_, _, d)| d.len()).sum();
    let vocab = t.tok.bpe.vocab_size();
    let imported = t.tok.bpe.imported().cloned();
    let (tokenizer, dropped) = match &imported {
        Some(c) => (c.tok.json.clone(), 0),
        None => {
            let (v, dropped) = tokenizer_json(&t.tok.bpe);
            (serde_json::to_vec_pretty(&v).unwrap_or_default(), dropped)
        }
    };
    let pretty = |v: &Value| serde_json::to_vec_pretty(v).unwrap_or_default();

    let mut generation = json!({
        "_from_model_config": true,
        "do_sample": true, "temperature": t.hyper.temperature, "max_new_tokens": t.hyper.sample_tokens,
    });
    let mut tok_config = json!({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": t.seq_len,
        "clean_up_tokenization_spaces": false,
    });
    let special = json!({ "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>" });
    if imported.is_none() {
        // the built-in tokenizer's own special tokens; an imported one keeps whatever it defines
        let g = generation.as_object_mut().unwrap();
        g.insert("bos_token_id".into(), json!(BOS_ID));
        g.insert("eos_token_id".into(), json!(EOS_ID));
        g.insert("pad_token_id".into(), json!(PAD_ID));
        let c = tok_config.as_object_mut().unwrap();
        c.insert("bos_token".into(), json!("<bos>"));
        c.insert("eos_token".into(), json!("<eos>"));
        c.insert("unk_token".into(), json!("<unk>"));
        c.insert("pad_token".into(), json!("<pad>"));
        c.insert("add_prefix_space".into(), json!(false));
    }

    let mut files: Vec<(&str, Vec<u8>, bool)> = vec![
        ("config.json", pretty(&config_json(t, vocab, params)), true),
        ("generation_config.json", pretty(&generation), true),
        ("model.safetensors", safetensors(&tensors), false),
        ("tokenizer.json", tokenizer, true),
        ("tokenizer_config.json", pretty(&tok_config), true),
    ];
    match &imported {
        Some(c) => files.push(("token_map.json", pretty(&json!({ "tokenizer_ids": c.raw_ids(), "original_vocab_size": c.tok.full_vocab })), true)),
        None => files.push(("special_tokens_map.json", pretty(&special), true)),
    }
    if t.provenance.styles.contains(&Style::Chat) {
        files.push(("chat_template.jinja", CHAT_TEMPLATE.as_bytes().to_vec(), true));
    }
    files.push(("README.md", readme(t, params, dropped).into_bytes(), true));

    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, data, compress) in files {
        let method = if compress { zip::CompressionMethod::Deflated } else { zip::CompressionMethod::Stored };
        let opts = zip::write::SimpleFileOptions::default().compression_method(method).large_file(data.len() > u32::MAX as usize);
        zip.start_file(format!("modelbuilder-model/{name}"), opts).map_err(|e| e.to_string())?;
        zip.write_all(&data).map_err(|e| e.to_string())?;
    }
    Ok(zip.finish().map_err(|e| e.to_string())?.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::tests::{run_job, run_job_tok, GEMMA};
    use burn::module::Module;

    fn trained() -> std::sync::Arc<Trained> {
        let (_, slot) = run_job(GEMMA, 10);
        let t = slot.lock().unwrap().clone().unwrap();
        t
    }

    #[test]
    fn byte_alphabet_matches_gpt2() {
        assert_eq!(byte_char(b'a'), 'a');
        assert_eq!(byte_char(b' '), 'Ġ');
        assert_eq!(byte_char(b'\n'), 'Ċ');
        assert_eq!(byte_char(0), 'Ā');
        assert_eq!((0..=255u8).map(byte_char).collect::<std::collections::HashSet<_>>().len(), 256);
    }

    #[test]
    fn every_learned_weight_is_exported_once_with_its_shape() {
        let t = trained();
        let tensors = t.model.named_tensors();
        let total: usize = tensors.iter().map(|(_, s, d)| { assert_eq!(s.iter().product::<usize>(), d.len()); d.len() }).sum();
        assert_eq!(total, t.model.num_params(), "exported elements must equal the model's parameter count");
        let names: std::collections::HashSet<_> = tensors.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(names.len(), tensors.len(), "names are unique");
        for n in ["model.embed_tokens.weight", "model.blocks.1.self_attn.q_proj.weight", "model.blocks.3.self_attn.sinks", "model.blocks.4.mlp.router.weight", "model.blocks.4.mlp.experts.3.down_proj.weight", "model.blocks.5.norm.weight"] {
            assert!(names.contains(n), "missing {n}: {names:?}");
        }
        // [out, in]: q_proj of a 48-dim model with 4 heads is 48x48, k_proj (2 KV heads) is 24x48
        let shape = |n: &str| tensors.iter().find(|x| x.0 == n).unwrap().1.clone();
        assert_eq!(shape("model.blocks.1.self_attn.k_proj.weight"), vec![24, 48]);
        assert_eq!(shape("model.blocks.2.mlp.gate_proj.weight"), vec![96, 48]);
    }

    #[test]
    fn safetensors_layout_is_valid() {
        let t = vec![("a".to_string(), vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]), ("b".to_string(), vec![1], vec![5.0])];
        let bytes = safetensors(&t);
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(n % 8, 0);
        let h: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        assert_eq!(h["a"]["data_offsets"], json!([0, 16]));
        assert_eq!(h["b"]["data_offsets"], json!([16, 20]));
        assert_eq!(h["__metadata__"]["format"], "pt");
        assert_eq!(bytes.len(), 8 + n + 20);
        assert_eq!(f32::from_le_bytes(bytes[8 + n + 16..].try_into().unwrap()), 5.0);
    }

    #[test]
    fn tokenizer_file_has_every_id_and_valid_merges() {
        let bpe = Bpe::train(&"the cat sat on the mat and the dog. ".repeat(40), 30);
        let (v, dropped) = tokenizer_json(&bpe);
        let vocab = v["model"]["vocab"].as_object().unwrap();
        assert_eq!(vocab.len(), bpe.vocab_size());
        let ids: std::collections::HashSet<u64> = vocab.values().map(|x| x.as_u64().unwrap()).collect();
        assert_eq!(ids.len(), bpe.vocab_size(), "ids are unique");
        assert_eq!(v["model"]["merges"].as_array().unwrap().len() + dropped, bpe.n_merges());
        for m in v["model"]["merges"].as_array().unwrap() {
            let (a, b) = m.as_str().unwrap().split_once(' ').unwrap();
            assert!(vocab.contains_key(a) && vocab.contains_key(b) && vocab.contains_key(&format!("{a}{b}")), "{m}");
        }
    }

    #[test]
    fn bundle_contains_the_repo_files() {
        let t = trained();
        let zip = bundle(&t).unwrap();
        let mut a = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
        let names: Vec<String> = (0..a.len()).map(|i| a.by_index(i).unwrap().name().to_string()).collect();
        for f in ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md"] {
            assert!(names.contains(&format!("modelbuilder-model/{f}")), "{f} missing from {names:?}");
        }
        assert!(!names.iter().any(|n| n.ends_with("chat_template.jinja")), "plain-text models get no chat template");
        let mut cfg = String::new();
        std::io::Read::read_to_string(&mut a.by_name("modelbuilder-model/config.json").unwrap(), &mut cfg).unwrap();
        let c: Value = serde_json::from_str(&cfg).unwrap();
        assert_eq!(c["model_type"], "modelbuilder");
        assert_eq!(c["blocks"].as_array().unwrap().len(), 7);
        assert_eq!(c["num_local_experts"], 4);
    }

    /// Writes a tokenizer.json plus the ids our own BPE gives for some texts, so an outside check
    /// (Python `tokenizers`) can confirm the exported file encodes identically:
    /// `EXPORT_PARITY_DIR=/tmp/x cargo test --release parity -- --ignored`.
    #[test]
    #[ignore]
    fn write_tokenizer_parity_files() {
        let Ok(dir) = std::env::var("EXPORT_PARITY_DIR") else { return };
        let corpus = "It's a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.\n\nHowever little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. \"My dear Mr. Bennet,\" said his lady to him one day, \"have you heard that Netherfield Park is let at last?\" Mr. Bennet replied that he had not. They'd 12 apples & 3 pears in 2024; we're done — I'll go.  Extra  spaces\tand tabs. Ünïcödé 日本語 text.\n".repeat(6);
        let bpe = Bpe::train(&corpus, 200);
        let (tok, dropped) = tokenizer_json(&bpe);
        std::fs::write(format!("{dir}/tokenizer.json"), serde_json::to_vec(&tok).unwrap()).unwrap();
        let texts = [
            "It's a truth universally acknowledged, that a single man",
            "\"My dear Mr. Bennet,\" said his lady to him one day",
            "They'd 12 apples & 3 pears in 2024; we're done — I'll go.",
            "  Extra  spaces\tand tabs.\n\nNew paragraph",
            "Ünïcödé 日本語 text, unseen words: zyxwv qqq",
            "don't won't 'tis rock'n'roll 'sun 'll",
            "",
        ];
        let cases: Vec<Value> = texts.iter().map(|t| json!({ "text": t, "ids": bpe.encode(t) })).collect();
        std::fs::write(format!("{dir}/cases.json"), serde_json::to_vec(&json!({ "dropped": dropped, "cases": cases })).unwrap()).unwrap();
    }

    #[test]
    fn an_imported_tokenizer_trains_and_is_exported_unchanged() {
        use crate::tokenizer::{compact_ids, ImportedTok};
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(80);
        let (json, _) = tokenizer_json(&Bpe::train(&text, 30));
        let bytes = serde_json::to_vec(&json).unwrap();
        let tok = std::sync::Arc::new(ImportedTok::parse("mine.json", bytes.clone()).unwrap());
        let (bpe, ids, counts) = compact_ids(&tok, &tok.encode_raw(&text)).unwrap();
        let data = std::sync::Arc::new(crate::tokenizer::Tokenized::from_parts(bpe, ids, text.chars().count(), text.len() as u64, counts, false));
        let (evs, slot) = run_job_tok(GEMMA, 10, data.clone());
        assert!(evs.iter().any(|e| matches!(e, crate::train::Event::Sample(s) if !s.text.is_empty())), "sampling decodes with the imported tokenizer");
        let t = slot.lock().unwrap().clone().unwrap();
        assert_eq!(t.model.vocab_size(), data.bpe.vocab_size());

        let zip = bundle(&t).unwrap();
        let mut a = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
        let read = |a: &mut zip::ZipArchive<std::io::Cursor<Vec<u8>>>, f: &str| {
            let mut v = Vec::new();
            std::io::Read::read_to_end(&mut a.by_name(&format!("modelbuilder-model/{f}")).unwrap(), &mut v).unwrap();
            v
        };
        assert_eq!(read(&mut a, "tokenizer.json"), bytes, "the uploaded file ships unchanged");
        let map: Value = serde_json::from_slice(&read(&mut a, "token_map.json")).unwrap();
        assert_eq!(map["tokenizer_ids"].as_array().unwrap().len(), data.bpe.vocab_size());
        let cfg: Value = serde_json::from_slice(&read(&mut a, "config.json")).unwrap();
        assert_eq!(cfg["original_vocab_size"], tok.full_vocab);
        assert_eq!(cfg["vocab_size"], data.bpe.vocab_size());
        assert!(cfg.get("bos_token_id").is_none(), "special ids belong to the imported tokenizer");
        assert!(String::from_utf8(read(&mut a, "README.md")).unwrap().contains("mine.json"));
    }
}
