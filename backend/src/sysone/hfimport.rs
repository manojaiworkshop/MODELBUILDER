//! Import real classification datasets from the Hugging Face Hub's public
//! datasets-server API (no auth needed for public datasets) and convert
//! them into System One `Sample`s: each row becomes a Choice question over
//! a random subset of classes (gold + distractors) — mirroring how the
//! reference project's own open-dataset importer avoids showing every row
//! all N labels, so the model learns to decide *within whatever options
//! are offered*, matching how the app's own "Ask it" panel queries it.

use crate::sysone::schema::{Question, Sample, Target};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;
use serde_json::Value;

const API: &str = "https://datasets-server.huggingface.co";
/// HF's datasets-server caps `length` at 100 rows per request.
const ROWS_PER_PAGE: usize = 100;
/// Matches `generate::MAX_SAMPLES` — the real ceiling is how many samples a session can hold at
/// all (shared with the other two generation modes), not an arbitrary smaller import limit.
pub const MAX_ROWS: usize = crate::sysone::generate::MAX_SAMPLES;
pub const MAX_OPTIONS_PER_Q: usize = 10;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HfColumn {
    pub name: String,
    pub dtype: String,
    /// Present when this is a Hugging Face `ClassLabel` column — the full label vocabulary.
    pub class_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HfSplitInfo {
    pub config: String,
    pub split: String,
    pub num_rows: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HfDatasetInfo {
    pub dataset: String,
    pub splits: Vec<HfSplitInfo>,
    /// Columns (same shape across configs, in practice) — collected from the first config seen.
    pub columns: Vec<HfColumn>,
}

fn friendly_error(status: reqwest::StatusCode, body: &str) -> String {
    if status == 404 {
        return "Dataset not found (check the repo id, e.g. \"dair-ai/emotion\").".into();
    }
    let hint = serde_json::from_str::<Value>(body).ok().and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string));
    hint.unwrap_or_else(|| format!("Hugging Face returned {status}"))
}

pub async fn fetch_info(client: &reqwest::Client, dataset: &str) -> Result<HfDatasetInfo, String> {
    let resp = client.get(format!("{API}/info")).query(&[("dataset", dataset)]).send().await.map_err(|e| format!("could not reach Hugging Face: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(friendly_error(status, &body));
    }
    let v: Value = serde_json::from_str(&body).map_err(|e| format!("unexpected response from Hugging Face: {e}"))?;
    let configs = v.get("dataset_info").and_then(|d| d.as_object()).ok_or("no dataset_info in response")?;
    let mut splits = Vec::new();
    let mut columns = Vec::new();
    for (config_name, info) in configs {
        if columns.is_empty() {
            if let Some(features) = info.get("features").and_then(|f| f.as_object()) {
                for (col_name, feat) in features {
                    let ty = feat.get("_type").and_then(|t| t.as_str()).unwrap_or("");
                    let dtype = feat.get("dtype").and_then(|t| t.as_str()).unwrap_or(ty).to_string();
                    let class_names = if ty == "ClassLabel" {
                        feat.get("names").and_then(|n| n.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                    } else {
                        None
                    };
                    columns.push(HfColumn { name: col_name.clone(), dtype, class_names });
                }
            }
        }
        if let Some(sp) = info.get("splits").and_then(|s| s.as_object()) {
            for (split_name, sinfo) in sp {
                let num_rows = sinfo.get("num_examples").and_then(|n| n.as_u64()).unwrap_or(0) as usize;
                splits.push(HfSplitInfo { config: config_name.clone(), split: split_name.clone(), num_rows });
            }
        }
    }
    if splits.is_empty() {
        return Err("This dataset has no readable splits (it may need a loading script Hugging Face no longer supports).".into());
    }
    Ok(HfDatasetInfo { dataset: dataset.to_string(), splits, columns })
}

async fn fetch_rows_page(client: &reqwest::Client, dataset: &str, config: &str, split: &str, offset: usize, length: usize) -> Result<Vec<Value>, String> {
    let resp = client
        .get(format!("{API}/rows"))
        .query(&[("dataset", dataset), ("config", config), ("split", split), ("offset", &offset.to_string()), ("length", &length.to_string())])
        .send()
        .await
        .map_err(|e| format!("could not reach Hugging Face: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(friendly_error(status, &body));
    }
    let v: Value = serde_json::from_str(&body).map_err(|e| format!("unexpected response from Hugging Face: {e}"))?;
    let rows = v.get("rows").and_then(|r| r.as_array()).ok_or("no rows in response")?;
    Ok(rows.iter().filter_map(|r| r.get("row").cloned()).collect())
}

/// Pages through `total` rows (capped at `MAX_ROWS`), stopping early if the dataset has fewer.
/// `on_page(fetched_so_far, total)` is called after every page, so a caller can show live
/// progress — a multi-thousand-row import is dozens of sequential HTTP calls and would otherwise
/// look stalled for a while.
pub async fn fetch_rows(client: &reqwest::Client, dataset: &str, config: &str, split: &str, total: usize, mut on_page: impl FnMut(usize, usize)) -> Result<Vec<Value>, String> {
    let total = total.min(MAX_ROWS);
    let mut out = Vec::with_capacity(total);
    let mut offset = 0;
    while out.len() < total {
        let length = ROWS_PER_PAGE.min(total - out.len());
        let page = fetch_rows_page(client, dataset, config, split, offset, length).await?;
        if page.is_empty() {
            break;
        }
        let got = page.len();
        out.extend(page);
        offset += got;
        on_page(out.len(), total);
        if got < length {
            break;
        }
    }
    Ok(out)
}

fn cell_as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// One label cell (an index into `class_names`, or a bare string) -> its display name.
fn label_name(v: &Value, class_names: &[String]) -> Option<String> {
    if let Some(i) = v.as_u64() {
        return class_names.get(i as usize).cloned();
    }
    cell_as_string(v)
}

/// Converts rows into Choice-question Samples: each gets the gold class plus up to
/// `options_per_q - 1` random distractor classes, so the model learns to decide within
/// whatever subset of options it's offered rather than always seeing the full label set.
pub fn rows_to_samples(rows: &[Value], text_col: &str, label_col: &str, class_names: &[String], options_per_q: usize, question_text: &str, source: &str, seed: u64) -> Vec<Sample> {
    let mut rng = StdRng::seed_from_u64(seed);
    let k = options_per_q.clamp(2, MAX_OPTIONS_PER_Q).min(class_names.len().max(2));
    rows.iter()
        .filter_map(|row| {
            let text = row.get(text_col).and_then(cell_as_string)?;
            let text = text.trim();
            if text.is_empty() {
                return None;
            }
            let gold = label_name(row.get(label_col)?, class_names)?;
            if !class_names.contains(&gold) {
                return None;
            }
            let mut options: Vec<String> = class_names.iter().filter(|c| **c != gold).cloned().collect();
            options.shuffle(&mut rng);
            options.truncate(k.saturating_sub(1));
            options.push(gold.clone());
            options.shuffle(&mut rng);
            let gold_pos = options.iter().position(|o| *o == gold)?;
            let mut probs = vec![0.0f32; options.len()];
            probs[gold_pos] = 1.0;
            Some(Sample {
                id: 0,
                state: text.chars().take(2000).collect(),
                questions: vec![Question::Choice { text: question_text.to_string(), options: options.clone() }],
                targets: vec![Target::Choice { probs }],
                source: source.to_string(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(text: &str, label: i64) -> Value {
        serde_json::json!({"text": text, "label": label})
    }

    #[test]
    fn converts_rows_with_gold_plus_distractors_and_one_hot_target() {
        let classes: Vec<String> = ["sadness", "joy", "love", "anger", "fear", "surprise"].iter().map(|s| s.to_string()).collect();
        let rows = vec![row("i feel great today", 1), row("i am so scared", 4)];
        let samples = rows_to_samples(&rows, "text", "label", &classes, 3, "which emotion is this?", "hf: dair-ai/emotion", 42);
        assert_eq!(samples.len(), 2);
        for (s, gold) in samples.iter().zip(["joy", "fear"]) {
            let Question::Choice { options, .. } = &s.questions[0] else { panic!("expected a choice question") };
            assert_eq!(options.len(), 3, "should offer gold + 2 distractors, not all 6 classes");
            assert!(options.contains(&gold.to_string()));
            let Target::Choice { probs } = &s.targets[0] else { panic!("expected a choice target") };
            assert_eq!(probs.iter().filter(|&&p| p == 1.0).count(), 1);
            let gold_idx = options.iter().position(|o| o == gold).unwrap();
            assert_eq!(probs[gold_idx], 1.0);
            assert_eq!(s.source, "hf: dair-ai/emotion");
            s.validate().unwrap();
        }
    }

    #[test]
    fn skips_rows_with_empty_text_or_unknown_label() {
        let classes: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let rows = vec![row("", 0), row("fine", 0)];
        let samples = rows_to_samples(&rows, "text", "label", &classes, 2, "q?", "src", 1);
        assert_eq!(samples.len(), 1, "the empty-text row must be dropped, not crash the batch");
    }

    #[test]
    fn caps_options_at_the_number_of_available_classes() {
        let classes: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let samples = rows_to_samples(&[row("x", 0)], "text", "label", &classes, 10, "q?", "src", 1);
        let Question::Choice { options, .. } = &samples[0].questions[0] else { panic!() };
        assert_eq!(options.len(), 2, "cannot offer more options than classes that exist");
    }
}
