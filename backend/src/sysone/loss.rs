//! Losses. All are proper scoring rules on probabilities, so minimizing them
//! pushes toward calibrated outputs — soft targets work directly, so
//! teacher-LLM probability labels need no special handling.
//!
//! - Noul:   binary cross-entropy vs. soft target p.
//! - Choice/Score: cross-entropy vs. soft target distribution.
//! - Confidence: BCE vs. "was the (detached) top decision correct?".
//!
//! Ported from `transformer_rs/src/sysone/loss.rs`.

use crate::sysone::encode::BatchTensors;
use crate::sysone::model::Outputs;
use burn::prelude::*;
use burn::tensor::activation::{log_sigmoid, log_softmax};

pub struct LossParts<B: Backend> {
    pub total: Tensor<B, 1>,
    pub noul: Tensor<B, 1>,
    pub opt: Tensor<B, 1>,
    pub conf: Tensor<B, 1>,
}

fn weighted_mean<B: Backend>(per_item: Tensor<B, 2>, weight: Tensor<B, 2>) -> Tensor<B, 1> {
    let denom = weight.clone().sum().clamp_min(1.0);
    (per_item * weight).sum() / denom
}

/// -(t * log sigmoid(x) + (1 - t) * log sigmoid(-x)), elementwise.
fn bce_with_logits<B: Backend>(x: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
    let pos = t.clone() * log_sigmoid(x.clone());
    let neg = t.neg().add_scalar(1.0) * log_sigmoid(x.neg());
    (pos + neg).neg()
}

pub fn compute_loss<B: Backend>(out: &Outputs<B>, t: &BatchTensors<B>, conf_weight: f32) -> LossParts<B> {
    let [b, q, _] = out.opt_logits.dims();

    let noul = weighted_mean(bce_with_logits(out.noul_logit.clone(), t.noul_t.clone()), t.w_noul.clone());

    let logp = log_softmax(out.opt_logits.clone(), 2);
    let opt_ce = (t.opt_t.clone() * logp).sum_dim(2).neg().reshape([b, q]);
    let opt = weighted_mean(opt_ce, t.w_opt.clone());

    let conf = {
        let noul_pred = out.noul_logit.clone().detach().greater_elem(0.0).float();
        let noul_gold = t.noul_t.clone().greater_elem(0.5).float();
        let noul_correct = (noul_pred - noul_gold).abs().neg().add_scalar(1.0);

        let pred_arg = out.opt_logits.clone().detach().argmax(2);
        let gold_arg = t.opt_t.clone().argmax(2);
        let opt_correct = pred_arg.equal(gold_arg).float().reshape([b, q]);

        let correct = noul_correct * t.w_noul.clone() + opt_correct * t.w_opt.clone();
        let w = t.w_noul.clone() + t.w_opt.clone();
        weighted_mean(bce_with_logits(out.conf_logit.clone(), correct), w)
    };

    let total = noul.clone() + opt.clone() + conf.clone().mul_scalar(conf_weight);
    LossParts { total, noul, opt, conf }
}
