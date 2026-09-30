//! Byte-level BPE, ported from transformer_rs `slm/bpe.rs`.
//!
//! The base vocabulary is the 256 byte values, so any text is representable
//! (zero OOV). Learned merges fuse frequent adjacent pairs into single tokens.
//! `merges = 0` therefore gives a plain byte/character-level tokenizer.
//!
//! Changes from the original:
//! - **Pre-tokenization** (the GPT-2 regex): text is first split into words, numbers, punctuation
//!   and whitespace runs, with a word's leading space attached to it (" the"). Merges are
//!   learned and applied *inside* these pieces only, so a token can never straddle two
//!   words ("t thy ", "e in") — the tokens are word parts, not arbitrary byte glue.
//! - Training counts each distinct piece once (weighted by frequency), so it is fast even on
//!   large corpora; `encode` memoises per distinct piece.
//! - Tokens can be rendered for display.

use std::collections::HashMap;
use std::sync::Arc;

pub const PAD_ID: u32 = 256;
pub const UNK_ID: u32 = 257;
pub const BOS_ID: u32 = 258;
pub const EOS_ID: u32 = 259;
const NUM_BYTES: u32 = 256;
pub const FIRST_MERGE_ID: u32 = NUM_BYTES + 4;

/// The tokenizer a dataset and model use: the built-in byte-level BPE (learned from the data), or a
/// Hugging Face `tokenizer.json` the user imported. Both hand out the same kind of ids, so
/// everything downstream (training, sampling, inspection) does not care which one it is.
#[derive(Debug, Clone)]
pub struct Bpe {
    kind: Kind,
}

#[derive(Debug, Clone)]
enum Kind {
    Builtin {
        /// Token id `FIRST_MERGE_ID + i` is the fusion of `merges[i]`.
        merges: Vec<(u32, u32)>,
        /// pair -> (rank, resulting id), for fast encoding.
        rank: HashMap<(u32, u32), (u32, u32)>,
    },
    Imported(Arc<Compact>),
}

/// An imported `tokenizer.json`, as uploaded.
pub struct ImportedTok {
    hf: tokenizers::Tokenizer,
    /// File name it was uploaded under.
    pub name: String,
    /// The file itself, kept so an exported model can ship it unchanged.
    pub json: Vec<u8>,
    /// "BPE", "Unigram", "WordPiece" or "WordLevel".
    pub model: String,
    /// Ids it defines (including added tokens).
    pub full_vocab: usize,
    max_id: u32,
}

impl std::fmt::Debug for ImportedTok {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImportedTok({}, {} ids)", self.name, self.full_vocab)
    }
}

impl ImportedTok {
    pub fn parse(name: &str, bytes: Vec<u8>) -> Result<ImportedTok, String> {
        let hf = tokenizers::Tokenizer::from_bytes(&bytes).map_err(|e| format!("That isn't a Hugging Face tokenizer.json ({e})."))?;
        let model = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| v["model"]["type"].as_str().map(str::to_string))
            .unwrap_or_else(|| "unknown".into());
        let vocab = hf.get_vocab(true);
        let max_id = vocab.values().copied().max().ok_or("The tokenizer has an empty vocabulary.")?;
        // ids index arrays, so a file with a wildly sparse id space is refused rather than allocated
        if max_id as usize > 4_000_000 {
            return Err(format!("The tokenizer has ids up to {max_id}; at most 4,000,000 are supported."));
        }
        Ok(ImportedTok { full_vocab: vocab.len(), hf, name: name.to_string(), json: bytes, model, max_id })
    }

    /// Text -> the tokenizer's own ids (no BOS/EOS or other special tokens are added).
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        self.hf.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
    }
}

/// An imported tokenizer with its ids compacted to the ones the data actually uses. Tokenizers
/// for real models have 30k-260k ids; a model trained from scratch on a small corpus only ever
/// sees a few thousand of them, and giving every one an embedding row (and a logit) would be
/// mostly dead weight. Compact ids are assigned in ascending original-id order.
#[derive(Debug)]
pub struct Compact {
    pub tok: Arc<ImportedTok>,
    to_compact: Vec<u32>,
    raw: Vec<u32>,
    special: Vec<bool>,
}

const ABSENT: u32 = u32::MAX;

impl Compact {
    /// Original tokenizer id of each compact id.
    pub fn raw_ids(&self) -> &[u32] {
        &self.raw
    }
}

/// Most distinct ids the data may use (ids are stored as `u16`).
const MAX_COMPACT: usize = 65_535;

/// Compacts `raw` ids (as produced by `ImportedTok::encode_raw`) into a tokenizer over just the ids
/// that occur, returning it with the remapped ids and their counts.
pub fn compact_ids(tok: &Arc<ImportedTok>, raw: &[u32]) -> Result<(Bpe, Vec<u16>, Vec<u64>), String> {
    let mut used = vec![false; tok.max_id as usize + 1];
    for &r in raw {
        if let Some(u) = used.get_mut(r as usize) {
            *u = true;
        }
    }
    let ids: Vec<u32> = used.iter().enumerate().filter(|(_, &u)| u).map(|(i, _)| i as u32).collect();
    if ids.len() > MAX_COMPACT {
        return Err(format!("The data uses {} different tokens of this tokenizer; at most {MAX_COMPACT} are supported. Use less text or a smaller tokenizer.", ids.len()));
    }
    let bpe = Bpe::compact(tok.clone(), ids);
    let ids: Vec<u16> = raw.iter().filter_map(|&r| bpe.compact_of(r)).map(|c| c as u16).collect();
    let mut counts = vec![0u64; bpe.vocab_size()];
    for &i in &ids {
        counts[i as usize] += 1;
    }
    Ok((bpe, ids, counts))
}

#[derive(Clone, Copy, PartialEq)]
enum Class {
    Letter,
    Digit,
    Space,
    Other,
}

fn class(c: char) -> Class {
    if c.is_alphabetic() {
        Class::Letter
    } else if c.is_numeric() {
        Class::Digit
    } else if c.is_whitespace() {
        Class::Space
    } else {
        Class::Other
    }
}

/// Splits text into GPT-2 pieces, exactly as the regex
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` does (the pattern
/// Hugging Face's byte-level BPE uses, so an exported `tokenizer.json` encodes identically):
/// `" word"`, `" 123"`, `" ,!"`, `"'s"`, and whitespace runs, where a run followed by text leaves
/// its last character for that text (only a space can attach to a word). Concatenating the pieces
/// gives back the input. Scans by byte offset (no per-character allocation), so it is cheap on
/// multi-megabyte chunks.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let n = text.len();
    let at = |i: usize| -> Option<(char, usize)> { text.get(i..).and_then(|t| t.chars().next()).map(|c| (c, c.len_utf8())) };
    let mut out = Vec::with_capacity(n / 4 + 1);
    let mut i = 0;
    while i < n {
        let start = i;
        let (c, l) = at(i).unwrap();
        // contractions, only lowercase and only where a piece starts
        if c == '\'' {
            let rest = &text[i + 1..];
            if let Some(k) = ["s", "t", "re", "ve", "m", "ll", "d"].iter().find(|s| rest.starts_with(**s)).map(|s| s.len()) {
                i += 1 + k;
                out.push(&text[start..i]);
                continue;
            }
        }
        // ` ?` then a run of letters, of digits, or of anything else that is not whitespace
        let body = if c == ' ' { i + l } else { i };
        if let Some((bc, _)) = at(body) {
            let cls = class(bc);
            if cls != Class::Space {
                let mut j = body;
                while let Some((ch, cl)) = at(j) {
                    if class(ch) != cls {
                        break;
                    }
                    j += cl;
                }
                out.push(&text[start..j]);
                i = j;
                continue;
            }
        }
        // whitespace: the whole run at the end of the text, otherwise all but its last character
        // (which then starts the next piece); a lone character is taken as it is
        let mut j = i;
        let mut last = i;
        while let Some((ch, cl)) = at(j) {
            if class(ch) != Class::Space {
                break;
            }
            last = j;
            j += cl;
        }
        let end = if j < n && last > i { last } else { j };
        out.push(&text[start..end]);
        i = end;
    }
    out
}

impl Bpe {
    fn from_merges(merges: Vec<(u32, u32)>) -> Self {
        let rank = merges.iter().enumerate().map(|(i, &p)| (p, (i as u32, FIRST_MERGE_ID + i as u32))).collect();
        Bpe { kind: Kind::Builtin { merges, rank } }
    }

    /// An imported tokenizer restricted to the sorted original `ids`.
    fn compact(tok: Arc<ImportedTok>, ids: Vec<u32>) -> Bpe {
        let mut to_compact = vec![ABSENT; tok.max_id as usize + 1];
        for (c, &r) in ids.iter().enumerate() {
            to_compact[r as usize] = c as u32;
        }
        let special_raw: std::collections::HashSet<u32> = tok.hf.get_added_tokens_decoder().into_iter().filter(|(_, t)| t.special).map(|(id, _)| id).collect();
        let special = ids.iter().map(|r| special_raw.contains(r)).collect();
        Bpe { kind: Kind::Imported(Arc::new(Compact { tok, to_compact, raw: ids, special })) }
    }

    fn compact_of(&self, raw: u32) -> Option<u32> {
        match &self.kind {
            Kind::Imported(c) => c.to_compact.get(raw as usize).copied().filter(|&x| x != ABSENT),
            Kind::Builtin { .. } => Some(raw),
        }
    }

    /// The imported tokenizer behind this one, if any.
    pub fn imported(&self) -> Option<&Arc<Compact>> {
        match &self.kind {
            Kind::Imported(c) => Some(c),
            Kind::Builtin { .. } => None,
        }
    }

    /// Special tokens (padding, unknown, BOS, EOS, …) are never sampled.
    pub fn is_special(&self, id: u32) -> bool {
        match &self.kind {
            Kind::Builtin { .. } => matches!(id, PAD_ID | UNK_ID | BOS_ID | EOS_ID),
            Kind::Imported(c) => c.special.get(id as usize).copied().unwrap_or(false),
        }
    }

    /// Ids -> text in the tokenizer's own way (an imported tokenizer's decoder, e.g. "▁" -> space).
    pub fn decode_ids(&self, ids: &[u32]) -> String {
        match &self.kind {
            Kind::Builtin { .. } => {
                let mut bytes = Vec::new();
                for &id in ids {
                    self.push_bytes(id, &mut bytes);
                }
                String::from_utf8_lossy(&bytes).into_owned()
            }
            Kind::Imported(c) => {
                let raw: Vec<u32> = ids.iter().filter_map(|&i| c.raw.get(i as usize).copied()).collect();
                c.tok.hf.decode(&raw, false).unwrap_or_default()
            }
        }
    }

    /// Learns up to `num_merges` merges from the distinct pieces of `text` (weighted by how often
    /// each occurs). A pair must occur at least twice to be worth a token.
    pub fn train(text: &str, num_merges: usize) -> Self {
        let mut freq: HashMap<&str, i64> = HashMap::new();
        for piece in pretokenize(text) {
            *freq.entry(piece).or_insert(0) += 1;
        }
        // deterministic order so the same text always gives the same tokenizer
        let mut pieces: Vec<(&str, i64)> = freq.into_iter().collect();
        pieces.sort();
        let mut words: Vec<(Vec<u32>, i64)> = pieces.into_iter().map(|(p, n)| (p.bytes().map(u32::from).collect(), n)).collect();

        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::new();
        for (w, n) in &words {
            for pair in w.windows(2) {
                *pair_counts.entry((pair[0], pair[1])).or_insert(0) += n;
            }
        }
        let mut merges = Vec::with_capacity(num_merges);
        for i in 0..num_merges {
            let best = pair_counts
                .iter()
                .filter(|&(_, &c)| c > 1)
                .max_by_key(|&(&pair, &c)| (c, std::cmp::Reverse(pair)))
                .map(|(&p, _)| p);
            let Some(pair) = best else { break };
            merges.push(pair);
            for (w, n) in words.iter_mut() {
                merge_in_place(w, pair, FIRST_MERGE_ID + i as u32, *n, &mut pair_counts);
            }
            pair_counts.remove(&pair);
        }
        Bpe::from_merges(merges)
    }

    pub fn vocab_size(&self) -> usize {
        match &self.kind {
            Kind::Builtin { merges, .. } => FIRST_MERGE_ID as usize + merges.len(),
            Kind::Imported(c) => c.raw.len(),
        }
    }

    /// The learned merges in rank order; token id `FIRST_MERGE_ID + i` is `merges()[i]` fused.
    pub fn merges(&self) -> &[(u32, u32)] {
        match &self.kind {
            Kind::Builtin { merges, .. } => merges,
            Kind::Imported(_) => &[],
        }
    }

    pub fn n_merges(&self) -> usize {
        self.merges().len()
    }

    /// Encodes one pre-token: repeatedly fuse the lowest-rank (earliest learned) adjacent pair.
    fn encode_piece(&self, piece: &str) -> Vec<u32> {
        let Kind::Builtin { rank, .. } = &self.kind else { return Vec::new() };
        let mut ids: Vec<u32> = piece.bytes().map(u32::from).collect();
        while ids.len() > 1 {
            let best = ids.windows(2).enumerate().filter_map(|(i, w)| rank.get(&(w[0], w[1])).map(|&(r, id)| (r, i, id))).min();
            let Some((_, at, id)) = best else { break };
            ids[at] = id;
            ids.remove(at + 1);
        }
        ids
    }

    /// Text -> ids (no BOS/EOS).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Kind::Imported(c) = &self.kind {
            // ids the data never used have no embedding row here and are dropped
            return c.tok.encode_raw(text).into_iter().filter_map(|r| self.compact_of(r)).collect();
        }
        let mut cache: HashMap<&str, Vec<u32>> = HashMap::new();
        let mut out = Vec::with_capacity(text.len() / 2);
        for piece in pretokenize(text) {
            let ids = cache.entry(piece).or_insert_with(|| self.encode_piece(piece));
            out.extend_from_slice(ids);
        }
        out
    }

    #[cfg(test)]
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_ids(ids)
    }

    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let mut b = Vec::new();
        self.push_bytes(id, &mut b);
        b
    }

    fn push_bytes(&self, id: u32, out: &mut Vec<u8>) {
        if let Kind::Imported(_) = &self.kind {
            out.extend_from_slice(self.token_text(id).as_bytes());
            return;
        }
        if id < NUM_BYTES {
            out.push(id as u8);
        } else if id >= FIRST_MERGE_ID {
            if let Some(&(a, b)) = self.merges().get((id - FIRST_MERGE_ID) as usize) {
                self.push_bytes(a, out);
                self.push_bytes(b, out);
            }
        }
        // specials (256..260) contribute no bytes
    }

    /// Human-readable form of one token for the UI. Invalid UTF-8 fragments
    /// (a multi-byte character split across tokens) show as `<0xE2>` etc.
    pub fn token_text(&self, id: u32) -> String {
        if let Kind::Imported(c) = &self.kind {
            let Some(&raw) = c.raw.get(id as usize) else { return String::new() };
            let piece = c.tok.hf.id_to_token(raw).unwrap_or_default();
            // SentencePiece-style "▁word" shows as " word"; other decoders (byte-level "Ġword") decode
            if let Some(rest) = piece.strip_prefix('▁') {
                return format!(" {}", rest.replace('▁', " "));
            }
            let dec = c.tok.hf.decode(&[raw], false).unwrap_or_default();
            return if dec.is_empty() || dec.contains('\u{FFFD}') { piece } else { dec };
        }
        match id {
            PAD_ID => return "<pad>".into(),
            UNK_ID => return "<unk>".into(),
            BOS_ID => return "<bos>".into(),
            EOS_ID => return "<eos>".into(),
            _ => {}
        }
        let mut bytes = Vec::new();
        self.push_bytes(id, &mut bytes);
        match std::str::from_utf8(&bytes) {
            Ok(s) => s.to_string(),
            Err(_) => bytes.iter().map(|b| format!("<0x{b:02X}>")).collect(),
        }
    }
}

fn merge_in_place(seq: &mut Vec<u32>, pair: (u32, u32), new_id: u32, weight: i64, counts: &mut HashMap<(u32, u32), i64>) {
    if seq.len() < 2 {
        return;
    }
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == pair.0 && seq[i + 1] == pair.1 {
            if let Some(&prev) = out.last() {
                *counts.entry((prev, pair.0)).or_insert(0) -= weight;
                *counts.entry((prev, new_id)).or_insert(0) += weight;
            }
            if i + 2 < seq.len() {
                let next = seq[i + 2];
                *counts.entry((pair.1, next)).or_insert(0) -= weight;
                *counts.entry((new_id, next)).or_insert(0) += weight;
            }
            *counts.entry(pair).or_insert(0) -= weight;
            out.push(new_id);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    *seq = out;
}

/// A tokenizer plus the corpus encoded with it (shared by inspect and training).
///
/// Token ids are stored as `u16` (vocabulary is at most 260 + 1000 merges), so a 500-million-token
/// corpus takes 1 GB rather than 2.
pub struct Tokenized {
    pub bpe: Bpe,
    pub ids: Vec<u16>,
    /// Characters of source text that produced `ids`.
    pub chars: usize,
    /// Bytes of source text.
    pub bytes: u64,
    /// Occurrences of each token id in `ids` (length = vocab size).
    pub counts: Vec<u64>,
    /// Token ids that occur in the corpus, most frequent first.
    pub frequent: Vec<u32>,
    /// True if the corpus was cut at the token limit.
    pub truncated: bool,
}

impl Tokenized {
    pub fn from_parts(bpe: Bpe, ids: Vec<u16>, chars: usize, bytes: u64, counts: Vec<u64>, truncated: bool) -> Self {
        let mut freq: Vec<(u32, u64)> = counts.iter().enumerate().filter(|(_, &c)| c > 0).map(|(i, &c)| (i as u32, c)).collect();
        freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Tokenized { bpe, ids, chars, bytes, counts, frequent: freq.into_iter().map(|x| x.0).collect(), truncated }
    }

    /// Small in-memory corpus (tests).
    #[cfg(test)]
    pub fn from_text(text: &str, merges: usize) -> Self {
        let bpe = Bpe::train(text, merges);
        let ids: Vec<u16> = bpe.encode(text).into_iter().map(|i| i as u16).collect();
        let mut counts = vec![0u64; bpe.vocab_size()];
        for &i in &ids {
            counts[i as usize] += 1;
        }
        Tokenized::from_parts(bpe, ids, text.chars().count(), text.len() as u64, counts, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_unseen_bytes() {
        let bpe = Bpe::train("hello world hello there world peace", 20);
        for t in ["hello world", "日本語 unseen", "{}[]\":,"] {
            assert_eq!(bpe.decode(&bpe.encode(t)), t);
        }
    }

    #[test]
    fn merges_frequent_pairs() {
        let bpe = Bpe::train(&"ab".repeat(100), 1);
        assert_eq!(bpe.encode("ab").len(), 1);
    }

    #[test]
    fn pretokenize_splits_like_gpt2_and_is_lossless() {
        let t = "Hello, world! It's 2024 — the  cat\nsat.";
        assert_eq!(pretokenize(t).concat(), t);
        let p = pretokenize("the cat, don't");
        assert_eq!(p, vec!["the", " cat", ",", " don", "'t"]);
        // whitespace runs leave their last character for what follows (GPT-2's `\s+(?!\S)`)
        assert_eq!(pretokenize("a\n\nb"), vec!["a", "\n", "\n", "b"]);
        assert_eq!(pretokenize("a\n\n\nb"), vec!["a", "\n\n", "\n", "b"]);
        assert_eq!(pretokenize("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pretokenize("end.\n\n"), vec!["end", ".", "\n\n"]);
        // contractions are only split off where a piece starts; after a space the quote is punctuation
        assert_eq!(pretokenize("'sun"), vec!["'s", "un"]);
        assert_eq!(pretokenize("x 'll"), vec!["x", " '", "ll"]);
        let p = pretokenize("a\n\nb  c");
        assert_eq!(p.concat(), "a\n\nb  c");
        assert!(pretokenize("").is_empty());
        assert_eq!(pretokenize("日本語 テスト").concat(), "日本語 テスト");
    }

    #[test]
    fn pretokenize_is_lossless_on_awkward_input() {
        let parts = [" ", "  ", "\n", "\r\n", "\t", "'", "'s", "ll", "a", "Zé", "日本", "٣", "12", "!", "—", "\u{a0}", "\u{2003}", "e\u{301}"];
        let mut seed = 7u64;
        for _ in 0..500 {
            let mut t = String::new();
            for _ in 0..(seed % 12) {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                t.push_str(parts[(seed >> 33) as usize % parts.len()]);
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let p = pretokenize(&t);
            assert_eq!(p.concat(), t, "{t:?}");
            assert!(p.iter().all(|x| !x.is_empty()), "{t:?} -> {p:?}");
        }
    }

    #[test]
    fn tokens_never_straddle_words() {
        let corpus = "the cat and the dog and the bird sat on the mat and the cat sat. ".repeat(20);
        let bpe = Bpe::train(&corpus, 60);
        for id in bpe.encode(&corpus) {
            let t = bpe.token_text(id);
            // a token is part of ONE word: at most one space, and only as its first character
            assert!(t.matches(' ').count() <= 1 && (!t.contains(' ') || t.starts_with(' ')), "token {t:?} spans words");
        }
        // frequent words become single tokens, with their leading space
        assert_eq!(bpe.encode(" the").len(), 1);
        assert_eq!(bpe.token_text(bpe.encode(" the")[0]), " the");
        assert_eq!(bpe.encode(" and").len(), 1);
    }

    #[test]
    fn zero_merges_is_byte_level() {
        let bpe = Bpe::train("abc", 0);
        assert_eq!(bpe.encode("abc"), vec![97, 98, 99]);
        assert_eq!(bpe.vocab_size(), 260);
    }

    /// A WordLevel tokenizer.json with `n` words, for testing very large vocabularies.
    fn word_level_json(n: usize) -> Vec<u8> {
        let vocab: Vec<String> = std::iter::once("\"[UNK]\":0".to_string()).chain((1..n).map(|i| format!("\"w{i}\":{i}"))).collect();
        format!(
            r#"{{"version":"1.0","added_tokens":[{{"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}],
            "normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":null,
            "model":{{"type":"WordLevel","vocab":{{{}}},"unk_token":"[UNK]"}}}}"#,
            vocab.join(",")
        )
        .into_bytes()
    }

    #[test]
    fn a_huge_vocabulary_is_compacted_to_the_ids_the_data_uses() {
        let tok = Arc::new(ImportedTok::parse("words.json", word_level_json(200_000)).unwrap());
        assert_eq!((tok.model.as_str(), tok.full_vocab), ("WordLevel", 200_000));
        let raw = tok.encode_raw("w5 w199999 w5 w7 nonsense w5");
        assert_eq!(raw, vec![5, 199_999, 5, 7, 0, 5], "unknown words become [UNK]");
        let (bpe, ids, counts) = compact_ids(&tok, &raw).unwrap();
        // five distinct ids were used; they keep their relative order
        assert_eq!(bpe.vocab_size(), 4);
        assert_eq!(bpe.imported().unwrap().raw_ids(), &[0, 5, 7, 199_999]);
        assert_eq!(ids, vec![1, 3, 1, 2, 0, 1]);
        assert_eq!(counts, vec![1, 3, 1, 1]);
        // new text is encoded into the same compact ids; ids the data never used are dropped
        assert_eq!(bpe.encode("w7 w5 w123"), vec![2, 1], "w123 is a real word the data never used: no embedding row, dropped");
        assert_eq!(bpe.encode("w7 zzz w5"), vec![2, 0, 1], "a word the tokenizer doesn't know is [UNK], which the data used");
        assert_eq!(bpe.decode_ids(&[1, 3]), "w5 w199999");
        assert!(bpe.is_special(0) && !bpe.is_special(1), "[UNK] is a special token, words are not");
        assert_eq!(bpe.token_text(3), "w199999");
    }

    #[test]
    fn sentencepiece_style_spacing_is_shown_and_decoded() {
        let json = r#"{"version":"1.0","added_tokens":[],"normalizer":null,
            "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},"post_processor":null,
            "decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},
            "model":{"type":"BPE","vocab":{"▁":0,"a":1,"b":2,"▁a":3,"▁b":4},"merges":["▁ a","▁ b"],"unk_token":null}}"#;
        let tok = Arc::new(ImportedTok::parse("sp.json", json.as_bytes().to_vec()).unwrap());
        let raw = tok.encode_raw("a b a");
        assert_eq!(raw, vec![3, 4, 3]);
        let (bpe, ids, _) = compact_ids(&tok, &raw).unwrap();
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        assert_eq!(bpe.token_text(ids[0]), " a", "▁a is shown as a word with its leading space");
        assert_eq!(bpe.decode_ids(&ids), "a b a");
    }

    #[test]
    fn rejects_files_that_are_not_tokenizers() {
        assert!(ImportedTok::parse("x.json", b"{\"hello\": 1}".to_vec()).is_err());
        assert!(ImportedTok::parse("x.json", b"not json at all".to_vec()).is_err());
    }

    #[test]
    fn too_many_distinct_ids_is_a_clear_error() {
        let tok = Arc::new(ImportedTok::parse("words.json", word_level_json(70_000)).unwrap());
        let raw: Vec<u32> = (0..70_000).collect();
        assert!(compact_ids(&tok, &raw).unwrap_err().contains("at most 65535"));
    }
}
