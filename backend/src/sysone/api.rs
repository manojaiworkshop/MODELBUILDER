//! Turns raw model outputs into typed `Answer`s, and the live "ask it a
//! question" inference entry point used by the Observe tab.
//!
//! Ported from `transformer_rs/src/sysone/api.rs`.

use crate::sysone::encode::{make_batch, Batch, EncodeConfig, Example, Kind};
use crate::sysone::model::{Outputs, SysOneModel};
use crate::sysone::schema::{Answer, Question, MAX_OPTIONS};
use crate::sysone::tokens::Tokenizer;
use burn::prelude::*;

fn to_f32_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Converts model outputs for `examples` (which produced `batch`) into one
/// `Vec<Answer>` per example, in question order.
pub fn answers_from_outputs<B: Backend>(out: &Outputs<B>, batch: &Batch, examples: &[Example]) -> Vec<Vec<Answer>> {
    let (q, o) = (batch.q, batch.o);
    let noul = to_f32_vec(out.noul_logit.clone());
    let opt = to_f32_vec(out.opt_logits.clone());
    let conf = to_f32_vec(out.conf_logit.clone());

    examples
        .iter()
        .enumerate()
        .map(|(bi, ex)| {
            ex.questions
                .iter()
                .enumerate()
                .map(|(qi, eq)| {
                    let bq = bi * q + qi;
                    let confidence = sigmoid(conf[bq]);
                    match eq.kind {
                        Kind::Noul => Answer::Noul { p: sigmoid(noul[bq]), confidence },
                        Kind::Choice | Kind::Score => {
                            let n = eq.opt_pos.len();
                            let logits = &opt[bq * o..bq * o + n];
                            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
                            let sum: f32 = exps.iter().sum();
                            let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
                            if eq.kind == Kind::Choice {
                                let best = probs.iter().enumerate().fold((0, -1.0), |a, (i, &p)| if p > a.1 { (i, p) } else { a }).0;
                                Answer::Choice { probs, best, confidence }
                            } else {
                                let levels = (n.max(2) - 1) as f32;
                                let score = probs.iter().enumerate().map(|(l, p)| p * l as f32 / levels).sum();
                                Answer::Score { probs, score, confidence }
                            }
                        }
                    }
                })
                .collect()
        })
        .collect()
}

/// One forward pass answers every question in `questions`.
pub fn decide<B: Backend>(model: &SysOneModel<B>, tok: &Tokenizer, enc: &EncodeConfig, device: &B::Device, state: &str, questions: &[Question]) -> Result<Vec<Answer>, String> {
    if questions.is_empty() {
        return Err("at least one question is required".into());
    }
    for (i, q) in questions.iter().enumerate() {
        let n = q.options().len();
        if !matches!(q, Question::Noul { .. }) && !(2..=MAX_OPTIONS).contains(&n) {
            return Err(format!("question {i}: needs 2..={MAX_OPTIONS} options, got {n}"));
        }
    }
    let ex = Example::encode(tok, enc, state, questions, None).ok_or("questions are too long to fit the model's context window")?;
    let examples = [ex];
    let batch = make_batch(&examples);
    let out = model.forward(&batch.to_tensors::<B>(device));
    Ok(answers_from_outputs(&out, &batch, &examples).remove(0))
}
