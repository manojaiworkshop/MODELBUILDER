//! The trained model the Observe tab talks to: either the from-scratch System One or a
//! fine-tuned Laya. Both answer typed questions about a state; tool calling is built on that.

use crate::sysone::api;
use crate::sysone::laya::TrainedLaya;
use crate::sysone::schema::{Answer, Question};
use crate::sysone::toolcall::{answer_all, call_tool_with, ToolCallOutput};
use crate::sysone::toolgen::ToolDef;
use crate::sysone::train::{self, Event, TrainHyper, TrainedSysOne};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, NdArray, Wgpu};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

pub enum Engine {
    Scratch(TrainedSysOne<NdArray>),
    Laya(TrainedLaya<Wgpu>),
}

impl Engine {
    /// One call, one model pass (the from-scratch model errors if the questions don't fit its window).
    pub fn decide(&self, state: &str, questions: &[Question]) -> Result<Vec<Answer>, String> {
        match self {
            Engine::Scratch(t) => api::decide::<NdArray>(&t.model, &t.tok, &t.enc, &Default::default(), state, questions),
            Engine::Laya(t) => t.decide(&WgpuDevice::default(), state, questions),
        }
    }

    /// Any number of questions (split over passes as the context needs).
    pub fn answer(&self, state: &str, questions: &[Question]) -> Result<Vec<Answer>, String> {
        match self {
            Engine::Scratch(t) => answer_all::<NdArray>(&t.model, &t.tok, &t.enc, &Default::default(), state, questions),
            Engine::Laya(t) => t.decide(&WgpuDevice::default(), state, questions),
        }
    }

    pub fn call_tool(&self, request: &str, tools: &[ToolDef]) -> Result<ToolCallOutput, String> {
        call_tool_with(&|s, q| self.answer(s, q), request, tools)
    }
}

/// Train on what the hyper-parameters ask for (scratch on CPU/GPU, or Laya on the GPU).
pub fn run_training(samples: Vec<crate::sysone::schema::Sample>, hyper: TrainHyper, cancel: Arc<AtomicBool>, tx: UnboundedSender<Event>) -> Option<Engine> {
    if hyper.base != "laya" {
        return train::run_selected(samples, hyper, cancel, tx).map(Engine::Scratch);
    }
    let tx2 = tx.clone();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || crate::sysone::layatrain::run::<Autodiff<Wgpu>>(samples, hyper, WgpuDevice::default(), cancel, tx)));
    match r {
        Ok(t) => t.map(Engine::Laya),
        Err(e) => {
            let why = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            let _ = tx2.send(Event::Status(crate::events::StatusPayload {
                state: "error",
                message: format!("Laya needs the GPU (Vulkan), which failed: {}", why.chars().take(300).collect::<String>()),
                step: 0,
                total: 0,
            }));
            None
        }
    }
}
