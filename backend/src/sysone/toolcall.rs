//! Tool calling with a trained System One model: request in, tool call out.
//!
//! 1. **Which tool.** Every tool in the catalogue is asked as a yes/no ("`<tool>` — is this the
//!    right tool?"); the best few then go into one "which tool fits?" choice, as in training.
//! 2. **Which arguments.** For the chosen tool, the questions its JSON Schema implies are asked in
//!    one pass: is each optional argument given, true/false values, fixed-value arguments, numbers
//!    among the standard candidates (`toolgen::arg_specs`).
//! 3. **Assembly.** Plain code builds the arguments object from those answers and wraps it in the
//!    OpenAI `tool_calls` shape. Free-text arguments (an ID or date copied from the request) can't
//!    be answered by a choice yet; they are reported as `missing` rather than guessed.
//!
//! Every decision keeps its probability, so the caller can see how sure the model was.

use crate::sysone::api::decide;
use crate::sysone::encode::{EncodeConfig, Example};
use crate::sysone::model::SysOneModel;
use crate::sysone::schema::{Answer, Question};
use crate::sysone::toolgen::{arg_specs, ArgKind, ArgSpec, ToolDef, SPECIFICITY};
use crate::sysone::tokens::Tokenizer;
use burn::prelude::*;
use serde::Serialize;
use serde_json::{json, Map, Value};

/// Tools offered together in the "which tool" choice (what training used: the right one + 3).
const CHOICE_OPTIONS: usize = 4;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub name: String,
    /// P("this is the right tool"), from the yes/no question.
    pub relevance: f32,
    /// Probability in the "which tool" choice, for the tools that made it in.
    pub choice: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Decision {
    /// Dotted argument path, e.g. `filters.SYSTEM_TRACK_TYPE`.
    pub path: String,
    /// What was asked: "is it specified?", "which value?", …
    pub question: String,
    /// What the model answered, as text.
    pub answer: String,
    /// Probability of that answer.
    pub p: f32,
    /// The model's own confidence that the answer is right.
    pub confidence: f32,
    /// Whether it ended up in the call (a value for an argument that wasn't included is unused).
    pub used: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Missing {
    pub path: String,
    pub required: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallOutput {
    pub tool: String,
    /// Probability of the chosen tool in the choice (1.0 when there was only one tool).
    pub tool_confidence: f32,
    pub candidates: Vec<Candidate>,
    /// How completely the request specifies the call, from the score question.
    pub specificity: Option<String>,
    pub arguments: Value,
    pub decisions: Vec<Decision>,
    pub missing: Vec<Missing>,
    /// `[{"function": {"arguments": "<json string>", "name": …}, "id": …, "type": "function"}]`
    pub tool_calls: Value,
}

/// Answers `questions` for `state`, splitting them over as many forward passes as the context
/// needs (questions can't see each other, so splitting doesn't change any answer).
fn answer_all<B: Backend>(model: &SysOneModel<B>, tok: &Tokenizer, enc: &EncodeConfig, device: &B::Device, state: &str, questions: &[Question]) -> Result<Vec<Answer>, String> {
    let budget = enc.max_seq_len.saturating_sub(1 + 64.min(enc.max_seq_len / 3));
    let mut chunks: Vec<Vec<Question>> = vec![Vec::new()];
    let mut used = 0usize;
    for q in questions {
        let cost = Example::encode(tok, enc, "", std::slice::from_ref(q), None).map(|e| e.ids.len() - 1).ok_or("a question is too long for the model's context window")?;
        if used + cost > budget && !chunks.last().unwrap().is_empty() {
            chunks.push(Vec::new());
            used = 0;
        }
        chunks.last_mut().unwrap().push(q.clone());
        used += cost;
    }
    let mut out = Vec::with_capacity(questions.len());
    for chunk in chunks.iter().filter(|c| !c.is_empty()) {
        out.extend(decide(model, tok, enc, device, state, chunk)?);
    }
    Ok(out)
}

fn noul(a: &Answer) -> (f32, f32) {
    match a {
        Answer::Noul { p, confidence } => (*p, *confidence),
        _ => (0.5, 0.0),
    }
}

fn argmax(p: &[f32]) -> usize {
    p.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0
}

/// Builds the arguments object from the answers to `specs`' questions. `answers[i]` belongs to
/// `specs[i]`. Returns the arguments, every decision, and the arguments that couldn't be filled.
pub fn assemble(schema: &Value, specs: &[ArgSpec], answers: &[Answer]) -> (Value, Vec<Decision>, Vec<Missing>) {
    let mut decisions = Vec::new();
    let mut missing = Vec::new();

    fn find<'a>(specs: &'a [ArgSpec], answers: &'a [Answer], path: &[String], pred: impl Fn(&ArgKind) -> bool) -> Option<(usize, &'a ArgSpec, &'a Answer)> {
        specs.iter().enumerate().find(|(_, s)| s.path == path && pred(&s.kind)).map(|(i, s)| (i, s, &answers[i]))
    }

    #[allow(clippy::too_many_arguments)]
    fn build(prefix: &[String], schema: &Value, specs: &[ArgSpec], answers: &[Answer], depth: usize, decisions: &mut Vec<Decision>, missing: &mut Vec<Missing>) -> Map<String, Value> {
        let mut out = Map::new();
        let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else { return out };
        let req: Vec<&str> = schema.get("required").and_then(|r| r.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
        for (k, ps) in props {
            let mut path = prefix.to_vec();
            path.push(k.clone());
            let name = path.join(".");
            let required = req.contains(&k.as_str());

            // is it given?
            let include = match find(specs, answers, &path, |k| matches!(k, ArgKind::Present)) {
                Some((_, spec, a)) => {
                    let (p, conf) = noul(a);
                    decisions.push(Decision { path: name.clone(), question: spec.text.clone(), answer: if p >= 0.5 { "yes" } else { "no" }.into(), p: if p >= 0.5 { p } else { 1.0 - p }, confidence: conf, used: true });
                    p >= 0.5
                }
                None => required,
            };

            // its value, when the kind of argument can be answered by choice
            let value = find(specs, answers, &path, |k| !matches!(k, ArgKind::Present)).map(|(_, spec, a)| {
                let (v, label, p, conf): (Value, String, f32, f32) = match (&spec.kind, a) {
                    (ArgKind::Enum(vals), Answer::Choice { probs, best, confidence }) => (vals[*best].clone(), crate::sysone::toolgen::label_of(&vals[*best]), probs[*best], *confidence),
                    (ArgKind::Number(c, integer), Answer::Choice { probs, best, confidence }) => {
                        let x = c[*best];
                        let v = if *integer { json!(x as i64) } else { json!(x) };
                        (v, if x.fract() == 0.0 { format!("{}", x as i64) } else { x.to_string() }, probs[*best], *confidence)
                    }
                    (ArgKind::Bool, a) => {
                        let (p, conf) = noul(a);
                        (json!(p >= 0.5), (p >= 0.5).to_string(), if p >= 0.5 { p } else { 1.0 - p }, conf)
                    }
                    _ => (Value::Null, String::new(), 0.0, 0.0),
                };
                decisions.push(Decision { path: name.clone(), question: spec.text.clone(), answer: label, p, confidence: conf, used: include });
                v
            });

            if !include {
                continue;
            }
            match value {
                Some(v) if !v.is_null() => {
                    out.insert(k.clone(), v);
                }
                _ => {
                    // not a choice-type argument: an object recurses, anything else can't be filled yet
                    let types: Vec<String> = match ps.get("type") {
                        Some(Value::String(s)) => vec![s.clone()],
                        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
                        _ => vec![],
                    };
                    if types.iter().any(|t| t == "object") && ps.get("properties").is_some() && depth < 1 {
                        let child = build(&path, ps, specs, answers, depth + 1, decisions, missing);
                        if !child.is_empty() || required {
                            out.insert(k.clone(), Value::Object(child));
                        }
                    } else {
                        missing.push(Missing { path: name, required, reason: "free-text value — the model can't fill this in yet".into() });
                    }
                }
            }
        }
        out
    }

    let args = build(&[], schema, specs, answers, 0, &mut decisions, &mut missing);
    (Value::Object(args), decisions, missing)
}

fn call_id(request: &str, tool: &str) -> String {
    let h = request.bytes().chain(tool.bytes()).fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    format!("call-{h:016x}")
}

/// The whole pipeline for one request over `tools`.
pub fn call_tool<B: Backend>(model: &SysOneModel<B>, tok: &Tokenizer, enc: &EncodeConfig, device: &B::Device, request: &str, tools: &[ToolDef]) -> Result<ToolCallOutput, String> {
    if tools.is_empty() {
        return Err("There are no tools to choose from.".into());
    }

    // ---- 1. which tool
    let mut candidates: Vec<Candidate> = tools.iter().map(|t| Candidate { name: t.name.clone(), relevance: 1.0, choice: None }).collect();
    let (chosen, tool_confidence) = if tools.len() == 1 {
        (0, 1.0)
    } else {
        let qs: Vec<Question> = tools.iter().map(|t| Question::Noul { text: format!("{} — is this the right tool?", t.name) }).collect();
        for (c, a) in candidates.iter_mut().zip(answer_all(model, tok, enc, device, request, &qs)?) {
            c.relevance = noul(&a).0;
        }
        // the most relevant few, kept in catalogue order, go into one choice
        let mut order: Vec<usize> = (0..tools.len()).collect();
        order.sort_by(|&a, &b| candidates[b].relevance.partial_cmp(&candidates[a].relevance).unwrap_or(std::cmp::Ordering::Equal));
        let mut top: Vec<usize> = order.into_iter().take(CHOICE_OPTIONS.min(tools.len())).collect();
        top.sort_unstable();
        let q = Question::Choice { text: "Which tool fits this request?".into(), options: top.iter().map(|&i| tools[i].name.clone()).collect() };
        match answer_all(model, tok, enc, device, request, &[q])?.remove(0) {
            Answer::Choice { probs, best, .. } => {
                for (&i, &p) in top.iter().zip(&probs) {
                    candidates[i].choice = Some(p);
                }
                (top[best], probs[argmax(&probs)])
            }
            _ => return Err("unexpected answer to the tool question".into()),
        }
    };
    let tool = &tools[chosen];

    // ---- 2. its arguments (and how completely the request says what the call needs)
    let specs = arg_specs(&tool.input_schema);
    let mut qs: Vec<Question> = specs.iter().map(|s| s.question()).collect();
    qs.push(Question::Score { text: "How completely does the request specify the call?".into(), levels: SPECIFICITY.iter().map(|s| s.to_string()).collect() });
    let mut answers = answer_all(model, tok, enc, device, request, &qs)?;
    let specificity = match answers.pop() {
        Some(Answer::Score { probs, .. }) => Some(SPECIFICITY[argmax(&probs).min(SPECIFICITY.len() - 1)].to_string()),
        _ => None,
    };

    // ---- 3. the call
    let (arguments, decisions, missing) = assemble(&tool.input_schema, &specs, &answers);
    let tool_calls = json!([{
        "function": { "arguments": serde_json::to_string_pretty(&arguments).unwrap_or_else(|_| "{}".into()), "name": tool.name },
        "id": call_id(request, &tool.name),
        "type": "function",
    }]);
    candidates.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ToolCallOutput { tool: tool.name.clone(), tool_confidence, candidates, specificity, arguments, decisions, missing, tool_calls })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sysone::schema::Sample;
    use crate::sysone::toolgen::{build_sample, validate_args, Example as TgExample};
    use crate::sysone::train::{run, Event, TrainHyper};
    use burn::backend::{Autodiff, NdArray};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn tool(name: &str, schema: Value) -> ToolDef {
        ToolDef { name: name.into(), description: "d".into(), input_schema: schema }
    }

    fn catalogue() -> Vec<ToolDef> {
        vec![
            tool("srv__recent_rows", json!({"type":"object","required":["limit"],"properties":{
                "limit":{"type":"integer","default":20,"minimum":1,"maximum":100},
                "sort_desc":{"type":"boolean"},
                "kind":{"type":"string","enum":["IFC","FTD"]},
                "note":{"type":"string"},
                "filters":{"type":"object","properties":{"TRACK":{"type":"string","enum":["IFC","FTD"]}}}}})),
            tool("srv__count_rows", json!({"type":"object","properties":{"kind":{"type":"string","enum":["IFC","FTD"]}}})),
            tool("srv__ping", json!({"type":"object","properties":{}})),
        ]
    }

    #[test]
    fn assembles_arguments_from_answers() {
        let t = &catalogue()[0];
        let specs = arg_specs(&t.input_schema);
        // answer every question with the first option / "yes"
        let answers: Vec<Answer> = specs
            .iter()
            .map(|s| match &s.kind {
                ArgKind::Present | ArgKind::Bool => Answer::Noul { p: 0.9, confidence: 0.8 },
                ArgKind::Enum(v) => Answer::Choice { probs: { let mut p = vec![0.1 / v.len() as f32; v.len()]; p[1] = 0.9; p }, best: 1, confidence: 0.7 },
                ArgKind::Number(c, _) => Answer::Choice { probs: { let mut p = vec![0.0; c.len()]; p[3] = 1.0; p }, best: 3, confidence: 0.9 },
            })
            .collect();
        let (args, decisions, missing) = assemble(&t.input_schema, &specs, &answers);
        assert_eq!(args["limit"], 20); // candidates are 1,5,10,20,50,100 -> index 3
        assert_eq!(args["sort_desc"], true);
        assert_eq!(args["kind"], "FTD");
        assert_eq!(args["filters"]["TRACK"], "FTD");
        assert!(args.get("note").is_none(), "free text isn't invented");
        assert!(missing.iter().any(|m| m.path == "note" && !m.required));
        assert!(decisions.iter().any(|d| d.path == "kind" && d.answer == "FTD" && d.used));
        validate_args(&t.input_schema, &args, 0).unwrap();
    }

    #[test]
    fn declined_optional_arguments_are_left_out() {
        let t = &catalogue()[0];
        let specs = arg_specs(&t.input_schema);
        let answers: Vec<Answer> = specs
            .iter()
            .map(|s| match &s.kind {
                ArgKind::Present => Answer::Noul { p: 0.05, confidence: 0.9 },
                ArgKind::Bool => Answer::Noul { p: 0.95, confidence: 0.9 },
                ArgKind::Enum(v) => Answer::Choice { probs: vec![1.0 / v.len() as f32; v.len()], best: 0, confidence: 0.5 },
                ArgKind::Number(c, _) => Answer::Choice { probs: { let mut p = vec![0.0; c.len()]; p[1] = 1.0; p }, best: 1, confidence: 0.9 },
            })
            .collect();
        let (args, decisions, _) = assemble(&t.input_schema, &specs, &answers);
        assert_eq!(args, json!({"limit": 5}), "only the required argument remains");
        assert!(decisions.iter().any(|d| d.path == "sort_desc" && !d.used), "a value for an argument that was left out is reported as unused");
    }

    #[test]
    fn a_trained_model_returns_a_well_formed_tool_call() {
        let cat = catalogue();
        // training data: two clearly different request styles per tool
        let mut samples: Vec<Sample> = Vec::new();
        for i in 0..24 {
            let (n, kind) = ([5, 20][i % 2], ["IFC", "FTD"][i % 2]);
            let ex = TgExample { request: format!("show me the latest {n} rows please, ref {i}"), arguments: json!({"limit": n, "kind": kind}), specificity: 2 };
            samples.push(build_sample(&cat[0], &ex, &cat, 2));
            let ex = TgExample { request: format!("how many rows are there in total, ref {i}"), arguments: json!({"kind": kind}), specificity: 1 };
            samples.push(build_sample(&cat[1], &ex, &cat, 2));
        }
        type TB = Autodiff<NdArray>;
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let hyper = TrainHyper { epochs: 25, batch_size: 8, lr: 3e-3, warmup_steps: 10, d_model: 48, n_heads: 2, n_kv_heads: 1, n_layers: 2, d_ff: 96, max_seq_len: 512, ..Default::default() };
        let trained = run::<TB>(samples, hyper, Default::default(), Arc::new(AtomicBool::new(false)), tx).expect("trains");

        let out = call_tool::<NdArray>(&trained.model, &trained.tok, &trained.enc, &Default::default(), "show me the latest 20 rows please", &cat).expect("inference");
        assert!(cat.iter().any(|t| t.name == out.tool));
        assert_eq!(out.candidates.len(), 3);
        assert!((0.0..=1.0).contains(&out.tool_confidence));
        let calls = out.tool_calls.as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], out.tool);
        // the arguments field is a JSON *string*, as in the OpenAI format
        let parsed: Value = serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(parsed, out.arguments);
        let t = cat.iter().find(|t| t.name == out.tool).unwrap();
        // whatever it chose, the call can't break the schema except for arguments reported missing
        let mut a = out.arguments.clone();
        for m in out.missing.iter().filter(|m| m.required) {
            a[&m.path] = json!("x");
        }
        validate_args(&t.input_schema, &a, 0).unwrap();
        // one tool alone needs no choice
        let one = call_tool::<NdArray>(&trained.model, &trained.tok, &trained.enc, &Default::default(), "ping", &cat[2..]).unwrap();
        assert_eq!(one.tool, "srv__ping");
        assert_eq!(one.arguments, json!({}));
    }
}
