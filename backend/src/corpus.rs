//! Turns a set of extracted text files (possibly gigabytes) into a trained tokenizer plus a
//! compact token array, without ever holding the text in memory as one string.
//!
//! 1. **Sample** a few tens of MB spread evenly across every file and train BPE on that
//!    (the standard way to build a tokenizer for a large corpus).
//! 2. **Encode** the whole corpus in ~1 MB chunks, many chunks in parallel (rayon), keeping
//!    results in order. Chunks are cut on newline/space so no word is split.
//! 3. Stop at a token limit (default 500 M tokens = 1 GB of `u16`) and say so.

use crate::tokenizer::{Bpe, Tokenized};
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

pub const SAMPLE_BYTES: u64 = 48 << 20;
/// Sample window size: small windows spread the sample evenly across a huge file.
const WINDOW: u64 = 64 << 10;
const CHUNK: usize = 1 << 20;

pub struct Cfg {
    pub merges: usize,
    pub max_tokens: usize,
    pub sample_bytes: u64,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg { merges: 200, max_tokens: 500_000_000, sample_bytes: SAMPLE_BYTES }
    }
}

/// `(stage, fraction 0..1, message)`
pub type Progress<'a> = &'a (dyn Fn(&'static str, f32, String) + Sync);

fn mb(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    }
}

/// Reads `len` bytes at `off`, trimmed to whole words and valid UTF-8 at both ends.
fn window(f: &mut File, off: u64, len: u64, first: bool, whole_file: bool) -> std::io::Result<String> {
    f.seek(SeekFrom::Start(off))?;
    let mut buf = Vec::with_capacity(len as usize);
    f.take(len).read_to_end(&mut buf)?;
    let mut start = 0;
    if !whole_file && !first {
        // skip the partial word we landed in
        start = buf.iter().position(|&b| b == b'\n' || b == b' ').map_or(buf.len(), |p| p + 1);
    }
    while start < buf.len() && (buf[start] & 0xC0) == 0x80 {
        start += 1;
    }
    let mut end = buf.len();
    if !whole_file {
        end = buf.iter().rposition(|&b| b == b'\n' || b == b' ').map_or(start, |p| p + 1).max(start);
    }
    Ok(String::from_utf8_lossy(&buf[start..end]).into_owned())
}

/// Evenly spread sample of at most `budget` bytes across all files.
pub fn sample(files: &[PathBuf], budget: u64) -> std::io::Result<String> {
    let sizes: Vec<u64> = files.iter().map(|p| std::fs::metadata(p).map(|m| m.len())).collect::<Result<_, _>>()?;
    let total: u64 = sizes.iter().sum();
    let mut out = String::new();
    for (p, &size) in files.iter().zip(&sizes) {
        if size == 0 {
            continue;
        }
        let mut f = File::open(p)?;
        if total <= budget || size <= WINDOW {
            out.push_str(&window(&mut f, 0, size, true, true)?);
            out.push('\n');
            continue;
        }
        // this file's share of the budget, as evenly spaced windows
        let share = (budget as u128 * size as u128 / total as u128) as u64;
        let n = (share / WINDOW).max(1);
        for k in 0..n {
            let off = if n == 1 { 0 } else { k * (size - WINDOW) / (n - 1) };
            out.push_str(&window(&mut f, off, WINDOW, k == 0, false)?);
            out.push('\n');
        }
    }
    Ok(out)
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r')
}

/// Where to split a buffer so that no tokenizer piece straddles the cut: just *before* the last
/// whitespace run (a word's leading space belongs to that word, so the next chunk must start with it).
/// If a single line is enormous and has no usable whitespace, fall back to the last UTF-8 boundary.
fn cut_point(buf: &[u8]) -> usize {
    if let Some(mut p) = buf.iter().rposition(|&b| is_ws(b)) {
        while p > 0 && is_ws(buf[p - 1]) {
            p -= 1;
        }
        if p > 0 {
            return p;
        }
    }
    if buf.len() >= 16 << 20 {
        return match std::str::from_utf8(buf) {
            Ok(_) => buf.len(),
            Err(e) => e.valid_up_to(),
        };
    }
    0
}

struct Encoded {
    ids: Vec<u16>,
    chars: usize,
}

/// Streams `files` in cut-on-boundary chunks, handing batches to `f` in order.
fn for_each_batch(files: &[PathBuf], batch: usize, cancel: &AtomicBool, progress: Progress, total: u64, mut f: impl FnMut(Vec<String>) -> bool) -> Result<(), String> {
    let mut done = 0u64;
    let mut pending: Vec<String> = Vec::with_capacity(batch);
    for path in files {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut carry: Vec<u8> = Vec::new();
        let mut buf = vec![0u8; CHUNK];
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let n = file.read(&mut buf).map_err(|e| e.to_string())?;
            done += n as u64;
            carry.extend_from_slice(&buf[..n]);
            let eof = n == 0;
            let cut = if eof { carry.len() } else { cut_point(&carry) };
            if cut > 0 {
                let rest = carry.split_off(cut);
                let chunk = String::from_utf8(std::mem::replace(&mut carry, rest)).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
                pending.push(chunk);
                if pending.len() >= batch {
                    if !f(std::mem::take(&mut pending)) {
                        return Ok(());
                    }
                    progress("encode", done as f32 / total.max(1) as f32, format!("Tokenizing… {} of {}", mb(done), mb(total)));
                }
            }
            if eof {
                break;
            }
        }
        // a separator between documents
        pending.push("\n\n".to_string());
    }
    if !pending.is_empty() {
        f(pending);
    }
    Ok(())
}

pub fn build(files: &[PathBuf], cfg: &Cfg, cancel: &AtomicBool, progress: Progress) -> Result<Tokenized, String> {
    let sizes: Vec<u64> = files.iter().map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).collect();
    let total: u64 = sizes.iter().sum();
    if total == 0 {
        return Err("There is no text to train on yet. Paste some text or upload a file.".into());
    }

    progress("tokenizer", 0.0, format!("Sampling {} to learn the vocabulary…", mb(total.min(cfg.sample_bytes))));
    let text = sample(files, cfg.sample_bytes).map_err(|e| e.to_string())?;
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let bpe = Bpe::train(&text, cfg.merges);
    drop(text);
    let vocab = bpe.vocab_size();
    assert!(vocab < 65_536, "token ids must fit in u16");

    progress("encode", 0.0, format!("Tokenizing {}…", mb(total)));
    let threads = rayon::current_num_threads().max(1);
    let mut ids: Vec<u16> = Vec::with_capacity((total as usize).min(cfg.max_tokens));
    let mut chars = 0usize;
    let mut counts = vec![0u64; vocab];
    let mut truncated = false;

    for_each_batch(files, threads * 2, cancel, progress, total, |chunks| {
        let encoded: Vec<Encoded> = chunks
            .par_iter()
            .map(|c| Encoded { ids: bpe.encode(c).into_iter().map(|i| i as u16).collect(), chars: c.chars().count() })
            .collect();
        for e in encoded {
            let room = cfg.max_tokens - ids.len();
            let take = e.ids.len().min(room);
            for &t in &e.ids[..take] {
                counts[t as usize] += 1;
            }
            ids.extend_from_slice(&e.ids[..take]);
            chars += e.chars;
            if take < e.ids.len() {
                truncated = true;
                return false;
            }
        }
        true
    })?;
    ids.shrink_to_fit();
    progress("encode", 1.0, format!("{} tokens", ids.len()));
    Ok(Tokenized::from_parts(bpe, ids, chars, total, counts, truncated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::AtomicBool;

    fn no_progress(_: &'static str, _: f32, _: String) {}

    fn words(n: usize, seed: u64) -> String {
        let vocab = ["the", "cat", "sat", "on", "a", "warm", "mat", "while", "rain", "fell", "over", "quiet", "hills", "and", "we", "walked"];
        let mut s = seed;
        (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); vocab[(s >> 33) as usize % vocab.len()] }).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn parallel_chunked_encoding_matches_encoding_the_whole_text() {
        let dir = tempfile::tempdir().unwrap();
        let text = (0..200).map(|i| format!("{}\n", words(6000, i))).collect::<String>();
        assert!(text.len() > 3 * CHUNK, "test must span several chunks ({} bytes)", text.len());
        let p = dir.path().join("a.txt");
        fs::write(&p, &text).unwrap();
        let cfg = Cfg { merges: 50, ..Default::default() };
        let t = build(&[p], &cfg, &AtomicBool::new(false), &no_progress).unwrap();
        // encoding the whole thing in one go must give the same tokens (chunk cuts are on word boundaries)
        let whole: Vec<u16> = t.bpe.encode(&text).into_iter().map(|i| i as u16).collect();
        // the corpus builder appends a "\n\n" document separator
        let sep = t.bpe.encode("\n\n").len();
        assert_eq!(t.ids.len(), whole.len() + sep, "token count differs");
        assert_eq!(&t.ids[..whole.len()], &whole[..], "chunked encoding must equal whole-text encoding");
        assert_eq!(t.counts.iter().sum::<u64>() as usize, t.ids.len());
        assert_eq!(t.chars, text.chars().count() + 2);
        assert!(!t.truncated);
    }

    #[test]
    fn several_files_and_token_limit() {
        let dir = tempfile::tempdir().unwrap();
        let files: Vec<PathBuf> = (0..3).map(|i| { let p = dir.path().join(format!("{i}.txt")); fs::write(&p, words(20_000, i)).unwrap(); p }).collect();
        let cfg = Cfg { merges: 30, max_tokens: 10_000, ..Default::default() };
        let t = build(&files, &cfg, &AtomicBool::new(false), &no_progress).unwrap();
        assert!(t.truncated && t.ids.len() == 10_000);
        let full = build(&files, &Cfg { merges: 30, ..Default::default() }, &AtomicBool::new(false), &no_progress).unwrap();
        assert!(!full.truncated && full.ids.len() > 30_000);
    }

    #[test]
    fn sampling_is_spread_across_a_large_file_and_respects_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big.txt");
        // beginning / middle / end carry different vocabulary
        let text = format!("{}\n{}\n{}\n", "alpha ".repeat(120_000), "middle ".repeat(120_000), "omega ".repeat(120_000));
        fs::write(&p, &text).unwrap();
        let s = sample(&[p], 600 << 10).unwrap();
        assert!(s.len() <= 700 << 10 && s.len() > 300 << 10, "{}", s.len());
        assert!(s.contains("alpha") && s.contains("middle") && s.contains("omega"), "sample must cover the whole file");
    }

    #[test]
    fn empty_input_and_cancellation() {
        let dir = tempfile::tempdir().unwrap();
        let e = dir.path().join("e.txt");
        fs::write(&e, "").unwrap();
        assert!(build(&[e], &Cfg::default(), &AtomicBool::new(false), &no_progress).err().unwrap().contains("no text"));
        let p = dir.path().join("t.txt");
        fs::write(&p, words(50_000, 1)).unwrap();
        assert_eq!(build(&[p], &Cfg::default(), &AtomicBool::new(true), &no_progress).err().unwrap(), "cancelled");
    }
}
