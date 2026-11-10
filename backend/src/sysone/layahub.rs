//! Downloading Laya from the Hugging Face Hub into the local cache.
//!
//! The weights are 842 MB and the Hub's CDN serves each connection slowly, so the file is fetched
//! as many parallel byte ranges (each retried, resuming where it broke off) into a `.part` file,
//! which is only renamed into place — and the `complete` marker only written — once every byte
//! has arrived. A half-finished download therefore never looks usable.

use crate::sysone::laya::{self, COMPLETE_MARKER, SMALL_FILES, WEIGHTS_FILE};
use futures_util::StreamExt;
use serde::Serialize;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const PARALLEL: usize = 32;
const CHUNK: u64 = 8 << 20;
const RETRIES: usize = 8;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayaStatus {
    /// missing | downloading | ready | error
    pub state: &'static str,
    pub done: u64,
    pub total: u64,
    pub message: String,
}

impl LayaStatus {
    pub fn current(dir: &Path) -> LayaStatus {
        if laya::is_downloaded(dir) {
            let total = std::fs::metadata(dir.join(WEIGHTS_FILE)).map(|m| m.len()).unwrap_or(0);
            LayaStatus { state: "ready", done: total, total, message: String::new() }
        } else {
            LayaStatus { state: "missing", done: 0, total: laya::WEIGHTS_BYTES_HINT, message: String::new() }
        }
    }
}

fn url(file: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{file}", laya::REPO)
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder().connect_timeout(Duration::from_secs(20)).read_timeout(Duration::from_secs(45)).build().map_err(|e| e.to_string())
}

/// Size of the weights, from a one-byte range request (`Content-Range: bytes 0-0/<total>`).
async fn weights_size(c: &reqwest::Client) -> Result<u64, String> {
    let r = c.get(url(WEIGHTS_FILE)).header("Range", "bytes=0-0").send().await.map_err(|e| format!("could not reach Hugging Face: {e}"))?;
    if !r.status().is_success() {
        return Err(format!("Hugging Face returned {} for the Laya weights", r.status()));
    }
    r.headers()
        .get("content-range")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.rsplit('/').next())
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| "Hugging Face didn't report the size of the Laya weights".to_string())
}

async fn fetch_range(c: &reqwest::Client, file: Arc<std::fs::File>, start: u64, end: u64, got_total: Arc<AtomicU64>, cancel: Arc<AtomicBool>) -> Result<(), String> {
    let mut pos = start;
    let mut last = String::new();
    for attempt in 0..RETRIES {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
        }
        let resp = match c.get(url(WEIGHTS_FILE)).header("Range", format!("bytes={pos}-{end}")).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                last = format!("HTTP {}", r.status());
                continue;
            }
            Err(e) => {
                last = e.to_string();
                continue;
            }
        };
        let mut stream = resp.bytes_stream();
        while let Some(piece) = stream.next().await {
            match piece {
                Ok(b) => {
                    let f = file.clone();
                    let at = pos;
                    let n = b.len() as u64;
                    tokio::task::spawn_blocking(move || f.write_all_at(&b, at)).await.map_err(|e| e.to_string())?.map_err(|e| format!("writing the file: {e}"))?;
                    pos += n;
                    got_total.fetch_add(n, Ordering::Relaxed);
                    if cancel.load(Ordering::Relaxed) {
                        return Err("cancelled".into());
                    }
                }
                Err(e) => {
                    last = e.to_string();
                    break;
                }
            }
        }
        if pos > end {
            return Ok(());
        }
    }
    Err(format!("the download kept failing ({last})"))
}

/// Downloads everything into `dir`, calling `progress(done, total)` as bytes arrive.
pub async fn download(dir: &Path, cancel: Arc<AtomicBool>, progress: impl Fn(u64, u64) + Send + Sync + 'static) -> Result<(), String> {
    let c = client()?;
    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let _ = std::fs::remove_file(dir.join(COMPLETE_MARKER));

    for f in SMALL_FILES {
        let bytes = c.get(url(f)).send().await.and_then(|r| r.error_for_status()).map_err(|e| format!("fetching {f}: {e}"))?.bytes().await.map_err(|e| format!("fetching {f}: {e}"))?;
        let target = dir.join(f);
        if let Some(p) = target.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        std::fs::write(&target, &bytes).map_err(|e| format!("writing {f}: {e}"))?;
    }

    let total = weights_size(&c).await?;
    let part = dir.join(format!("{WEIGHTS_FILE}.part"));
    let file = std::fs::OpenOptions::new().create(true).write(true).truncate(true).open(&part).map_err(|e| format!("creating {}: {e}", part.display()))?;
    let free = fs_free(dir);
    if free.map_or(false, |f| f < total + (256 << 20)) {
        return Err(format!("not enough disk space for the {} MB download", total >> 20));
    }
    file.set_len(total).map_err(|e| format!("reserving disk space: {e}"))?;
    let file = Arc::new(file);

    let got = Arc::new(AtomicU64::new(0));
    let ranges: Vec<(u64, u64)> = (0..total).step_by(CHUNK as usize).map(|s| (s, (s + CHUNK).min(total) - 1)).collect();
    let reporter = {
        let (got, cancel) = (got.clone(), cancel.clone());
        tokio::spawn(async move {
            while !cancel.load(Ordering::Relaxed) {
                progress(got.load(Ordering::Relaxed), total);
                if got.load(Ordering::Relaxed) >= total {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        })
    };
    let results: Vec<Result<(), String>> = futures_util::stream::iter(ranges)
        .map(|(s, e)| {
            let (c, file, got, cancel) = (c.clone(), file.clone(), got.clone(), cancel.clone());
            async move { fetch_range(&c, file, s, e, got, cancel).await }
        })
        .buffer_unordered(PARALLEL)
        .collect()
        .await;
    reporter.abort();
    if cancel.load(Ordering::Relaxed) {
        let _ = std::fs::remove_file(&part);
        return Err("cancelled".into());
    }
    if let Some(e) = results.into_iter().find_map(|r| r.err()) {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    file.sync_all().ok();
    let want = expected_sha256(&c).await?;
    let part2 = part.clone();
    let got_hash = tokio::task::spawn_blocking(move || sha256_of(&part2)).await.map_err(|e| e.to_string())??;
    if got_hash != want {
        let _ = std::fs::remove_file(&part);
        return Err("the downloaded weights are corrupt (checksum mismatch) — try again".into());
    }
    laya::Weights::open(&part).map_err(|e| format!("the downloaded weights are not valid: {e}"))?;
    std::fs::rename(&part, dir.join(WEIGHTS_FILE)).map_err(|e| format!("finishing the download: {e}"))?;
    std::fs::write(dir.join(COMPLETE_MARKER), total.to_string()).map_err(|e| e.to_string())?;
    Ok(())
}

/// The SHA-256 the Hub publishes for the weights (LFS metadata).
async fn expected_sha256(c: &reqwest::Client) -> Result<String, String> {
    let v: serde_json::Value = c
        .get(format!("https://huggingface.co/api/models/{}?blobs=true", laya::REPO))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("could not get the Laya checksum: {e}"))?
        .json()
        .await
        .map_err(|e| format!("could not read the Laya checksum: {e}"))?;
    v.get("siblings")
        .and_then(|s| s.as_array())
        .and_then(|a| a.iter().find(|f| f.get("rfilename").and_then(|n| n.as_str()) == Some(WEIGHTS_FILE)))
        .and_then(|f| f.pointer("/lfs/sha256"))
        .and_then(|h| h.as_str())
        .map(str::to_string)
        .ok_or_else(|| "the Hub didn't publish a checksum for the Laya weights".to_string())
}

fn sha256_of(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Free bytes on the filesystem holding `dir` (None if it can't be read).
fn fs_free(dir: &Path) -> Option<u64> {
    let out = std::process::Command::new("df").args(["-Pk"]).arg(dir).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let avail: u64 = text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
    Some(avail * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Downloads 842 MB: `cargo test --release hub_download -- --ignored --nocapture`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn hub_download_produces_a_loadable_cache() {
        let dir = std::env::temp_dir().join(format!("laya-dl-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let last = Arc::new(AtomicU64::new(0));
        let l2 = last.clone();
        let t0 = std::time::Instant::now();
        download(&dir, Arc::new(AtomicBool::new(false)), move |d, _| l2.store(d, Ordering::Relaxed)).await.expect("download");
        println!("downloaded in {:.0}s", t0.elapsed().as_secs_f32());
        assert!(laya::is_downloaded(&dir));
        assert_eq!(std::fs::metadata(dir.join(WEIGHTS_FILE)).unwrap().len(), laya::WEIGHTS_BYTES_HINT);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
