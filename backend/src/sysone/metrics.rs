//! Evaluation on `(Answer, Target)` pairs.
//!
//! Beyond accuracy, the metrics that matter for a decision model are
//! calibration: Brier score (mean squared error of probabilities) and
//! Expected Calibration Error — how far "stated probability" is from
//! "observed frequency", binned by top-class probability.
//!
//! Ported from `transformer_rs/src/sysone/metrics.rs`.

use crate::sysone::schema::{Answer, Target};
use serde::Serialize;

const BINS: usize = 10;

#[derive(Default, Clone)]
struct Bins {
    n: [usize; BINS],
    conf_sum: [f64; BINS],
    correct_sum: [f64; BINS],
}

impl Bins {
    fn add(&mut self, confidence: f32, correct: bool) {
        let b = ((confidence * BINS as f32) as usize).min(BINS - 1);
        self.n[b] += 1;
        self.conf_sum[b] += confidence as f64;
        self.correct_sum[b] += correct as u8 as f64;
    }

    fn ece(&self) -> f32 {
        let total: usize = self.n.iter().sum();
        if total == 0 {
            return 0.0;
        }
        (0..BINS)
            .filter(|&b| self.n[b] > 0)
            .map(|b| {
                let n = self.n[b] as f64;
                (n / total as f64) * ((self.correct_sum[b] - self.conf_sum[b]).abs() / n)
            })
            .sum::<f64>() as f32
    }
}

#[derive(Default, Clone)]
pub struct Accumulator {
    noul_n: usize,
    noul_correct: usize,
    noul_brier: f64,
    choice_n: usize,
    choice_correct: usize,
    choice_brier: f64,
    score_n: usize,
    score_correct: usize,
    score_brier: f64,
    score_mae: f64,
    prob_bins: Bins,
    conf_bins: Bins,
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0
}

impl Accumulator {
    pub fn add(&mut self, answer: &Answer, target: &Target) {
        let (correct, top_prob) = match (answer, target) {
            (Answer::Noul { p, .. }, Target::Noul { p: gold }) => {
                let correct = (*p > 0.5) == (*gold > 0.5);
                self.noul_n += 1;
                self.noul_correct += correct as usize;
                self.noul_brier += ((p - gold) as f64).powi(2);
                (correct, p.max(1.0 - p))
            }
            (Answer::Choice { probs, .. }, Target::Choice { probs: gold }) => {
                let correct = argmax(probs) == argmax(gold);
                self.choice_n += 1;
                self.choice_correct += correct as usize;
                self.choice_brier += probs.iter().zip(gold).map(|(p, g)| ((p - g) as f64).powi(2)).sum::<f64>();
                (correct, probs.iter().cloned().fold(0.0, f32::max))
            }
            (Answer::Score { probs, score, .. }, Target::Score { probs: gold }) => {
                let correct = argmax(probs) == argmax(gold);
                let levels = gold.len().max(2) as f32 - 1.0;
                let gold_score: f32 = gold.iter().enumerate().map(|(l, g)| g * l as f32 / levels).sum();
                self.score_n += 1;
                self.score_correct += correct as usize;
                self.score_brier += probs.iter().zip(gold).map(|(p, g)| ((p - g) as f64).powi(2)).sum::<f64>();
                self.score_mae += (score - gold_score).abs() as f64;
                (correct, probs.iter().cloned().fold(0.0, f32::max))
            }
            _ => return,
        };
        self.prob_bins.add(top_prob, correct);
        self.conf_bins.add(answer.confidence(), correct);
    }

    pub fn report(&self) -> Report {
        let ratio = |a: usize, n: usize| if n == 0 { f32::NAN } else { a as f32 / n as f32 };
        let mean = |x: f64, n: usize| if n == 0 { f32::NAN } else { (x / n as f64) as f32 };
        Report {
            n_noul: self.noul_n,
            n_choice: self.choice_n,
            n_score: self.score_n,
            noul_acc: ratio(self.noul_correct, self.noul_n),
            choice_acc: ratio(self.choice_correct, self.choice_n),
            score_acc: ratio(self.score_correct, self.score_n),
            noul_brier: mean(self.noul_brier, self.noul_n),
            choice_brier: mean(self.choice_brier, self.choice_n),
            score_brier: mean(self.score_brier, self.score_n),
            score_mae: mean(self.score_mae, self.score_n),
            prob_ece: self.prob_bins.ece(),
            conf_ece: self.conf_bins.ece(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    pub n_noul: usize,
    pub n_choice: usize,
    pub n_score: usize,
    pub noul_acc: f32,
    pub choice_acc: f32,
    pub score_acc: f32,
    pub noul_brier: f32,
    pub choice_brier: f32,
    pub score_brier: f32,
    /// Mean absolute error of the expected score, in [0, 1] units.
    pub score_mae: f32,
    /// ECE of the answer distribution's own top probability.
    pub prob_ece: f32,
    /// ECE of the separately-learned confidence head.
    pub conf_ece: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_and_overconfident_predictions_score_differently() {
        let mut perfect = Accumulator::default();
        let mut wrong = Accumulator::default();
        for _ in 0..100 {
            perfect.add(&Answer::Noul { p: 1.0, confidence: 1.0 }, &Target::Noul { p: 1.0 });
            wrong.add(&Answer::Noul { p: 0.95, confidence: 0.95 }, &Target::Noul { p: 0.0 });
        }
        let (p, w) = (perfect.report(), wrong.report());
        assert_eq!(p.noul_acc, 1.0);
        assert!(p.prob_ece < 1e-6 && p.noul_brier < 1e-6);
        assert_eq!(w.noul_acc, 0.0);
        assert!(w.prob_ece > 0.9, "confidently wrong should have huge ECE, got {}", w.prob_ece);
    }

    #[test]
    fn score_mae_uses_expected_level() {
        let mut acc = Accumulator::default();
        acc.add(&Answer::Score { probs: vec![0.0, 0.0, 1.0], score: 1.0, confidence: 1.0 }, &Target::Score { probs: vec![0.0, 1.0, 0.0] });
        let r = acc.report();
        assert!((r.score_mae - 0.5).abs() < 1e-6);
        assert_eq!(r.score_acc, 0.0);
    }
}
