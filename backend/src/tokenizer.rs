//! Byte-level BPE, ported from transformer_rs `slm/bpe.rs`.
//!
//! The base vocabulary is the 256 byte values, so any text is representable
//! (zero OOV). Learned merges fuse frequent adjacent pairs into single tokens.
//! `merges = 0` therefore gives a plain byte/character-level tokenizer.
//!
//! Changes from the original:
//! - **Pre-tokenization** (GPT-2 style): text is first split into words, numbers, punctuation
//!   and whitespace runs, with a word's leading space attached to it (" the"). Merges are
//!   learned and applied *inside* these pieces only, so a token can never straddle two
//!   words ("t thy ", "e in") — the tokens are word parts, not arbitrary byte glue.
//! - Training counts each distinct piece once (weighted by frequency), so it is fast even on
//!   large corpora; `encode` memoises per distinct piece.
//! - Tokens can be rendered for display.

use std::collections::HashMap;

pub const PAD_ID: u32 = 256;
pub const UNK_ID: u32 = 257;
pub const BOS_ID: u32 = 258;
pub const EOS_ID: u32 = 259;
const NUM_BYTES: u32 = 256;
pub const FIRST_MERGE_ID: u32 = NUM_BYTES + 4;

#[derive(Debug, Clone)]
pub struct Bpe {
    /// Token id `FIRST_MERGE_ID + i` is the fusion of `merges[i]`.
    merges: Vec<(u32, u32)>,
    /// pair -> (rank, resulting id), for fast encoding.
    rank: HashMap<(u32, u32), (u32, u32)>,
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

/// Splits text into GPT-2-style pieces: `" word"`, `" 123"`, `" ,!"`, `"'s"`, and whitespace runs.
/// A single space directly before a word/number/symbol belongs to that piece; newlines and
/// extra spaces stay as their own pieces. Concatenating the pieces gives back the input.
/// Scans by byte offset (no per-character allocation), so it is cheap on multi-megabyte chunks.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let n = text.len();
    let at = |i: usize| -> Option<(char, usize)> { text.get(i..).and_then(|t| t.chars().next()).map(|c| (c, c.len_utf8())) };
    let mut out = Vec::with_capacity(n / 4 + 1);
    let mut i = 0;
    while i < n {
        let start = i;
        let (c, l) = at(i).unwrap();
        // a lone ' ' glues to the run that follows it
        let lead = c == ' ' && at(i + l).is_some_and(|(nc, _)| class(nc) != Class::Space);
        let body = if lead { i + l } else { i };
        let (bc, bl) = at(body).unwrap();
        let cls = class(bc);
        let mut j = body;
        match cls {
            Class::Space => {
                while let Some((ch, cl)) = at(j) {
                    if class(ch) != Class::Space {
                        break;
                    }
                    j += cl;
                }
                // a run followed by a word leaves its last ' ' for that word
                if j < n && text.as_bytes()[j - 1] == b' ' && j - 1 > start {
                    j -= 1;
                }
            }
            Class::Other if bc == '\'' => {
                // contractions: 's 't 'd 'm 'll 're 've
                j = body + bl;
                let rest = &text[j..];
                let take = ["ll", "re", "ve"]
                    .iter()
                    .find(|s| rest.starts_with(**s))
                    .map(|s| s.len())
                    .or_else(|| rest.chars().next().filter(|c| "stdm".contains(*c)).map(|_| 1));
                match take {
                    Some(k) if at(j + k).is_none_or(|(nc, _)| class(nc) != Class::Letter) => j += k,
                    _ => {
                        while let Some((ch, cl)) = at(j) {
                            if class(ch) != Class::Other || ch == '\'' {
                                break;
                            }
                            j += cl;
                        }
                    }
                }
            }
            _ => {
                while let Some((ch, cl)) = at(j) {
                    if class(ch) != cls || (cls == Class::Other && ch == '\'' && j > body) {
                        break;
                    }
                    j += cl;
                }
            }
        }
        out.push(&text[start..j]);
        i = j;
    }
    out
}

impl Bpe {
    fn from_merges(merges: Vec<(u32, u32)>) -> Self {
        let rank = merges.iter().enumerate().map(|(i, &p)| (p, (i as u32, FIRST_MERGE_ID + i as u32))).collect();
        Bpe { merges, rank }
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
        FIRST_MERGE_ID as usize + self.merges.len()
    }

    pub fn n_merges(&self) -> usize {
        self.merges.len()
    }

    /// Encodes one pre-token: repeatedly fuse the lowest-rank (earliest learned) adjacent pair.
    fn encode_piece(&self, piece: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = piece.bytes().map(u32::from).collect();
        while ids.len() > 1 {
            let best = ids.windows(2).enumerate().filter_map(|(i, w)| self.rank.get(&(w[0], w[1])).map(|&(r, id)| (r, i, id))).min();
            let Some((_, at, id)) = best else { break };
            ids[at] = id;
            ids.remove(at + 1);
        }
        ids
    }

    /// Text -> ids (no BOS/EOS).
    pub fn encode(&self, text: &str) -> Vec<u32> {
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
        let mut bytes = Vec::new();
        for &id in ids {
            self.push_bytes(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let mut b = Vec::new();
        self.push_bytes(id, &mut b);
        b
    }

    fn push_bytes(&self, id: u32, out: &mut Vec<u8>) {
        if id < NUM_BYTES {
            out.push(id as u8);
        } else if id >= FIRST_MERGE_ID {
            if let Some(&(a, b)) = self.merges.get((id - FIRST_MERGE_ID) as usize) {
                self.push_bytes(a, out);
                self.push_bytes(b, out);
            }
        }
        // specials (256..260) contribute no bytes
    }

    /// Human-readable form of one token for the UI. Invalid UTF-8 fragments
    /// (a multi-byte character split across tokens) show as `<0xE2>` etc.
    pub fn token_text(&self, id: u32) -> String {
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
        let p = pretokenize("a\n\nb  c");
        assert_eq!(p.concat(), "a\n\nb  c");
        assert!(pretokenize("").is_empty());
        assert_eq!(pretokenize("日本語 テスト").concat(), "日本語 テスト");
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
}
