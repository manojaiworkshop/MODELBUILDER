//! Tool-calling samples for System One.
//!
//! For one tool at a time, a language model writes realistic user requests together with the
//! exact arguments the call should have. The arguments are checked against the tool's JSON
//! Schema, and then **code** (not the model) turns each request into a System One `Sample`:
//! the request is the state, and the call becomes typed questions with exact targets:
//!
//! | question | type | target |
//! |---|---|---|
//! | which tool fits this request? (the right one + distractors from the catalogue) | Choice | one-hot |
//! | `<tool>` — is this the right tool? (once for the right tool, once for a wrong one) | Noul | 1 / 0 |
//! | how completely does the request specify the call? (vague / partial / complete) | Score | the model's label |
//! | `<arg>` — is it specified? (optional arguments) | Noul | present or not |
//! | `<arg>` — is it true? (booleans) | Noul | value |
//! | `<arg>`: which value? (fixed-value arguments, and numbers among a standard set) | Choice | one-hot |
//!
//! Free-text arguments (an ID, a name, a date copied from the request) can't be expressed as a
//! choice; they stay in `Sample::call` for when a span-extraction head exists. Reading the
//! answers back into a call is then deterministic: the chosen tool, plus the argument decisions.

use crate::learn::{self, LlmConfig};
use crate::sysone::schema::{Question, Sample, Target, ToolCall};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub const MAX_TOOLS_PER_RUN: usize = 200;
pub const MAX_CATALOGUE: usize = 500;
pub const MAX_PER_TOOL: usize = 50;
pub const MAX_DISTRACTORS: usize = 6;
/// Examples requested from the model in one call.
const BATCH: usize = 8;
/// Most arguments (flattened one level) turned into questions for one sample.
const MAX_ARG_QUESTIONS: usize = 10;
/// Most values offered for a fixed-value argument.
const MAX_ENUM: usize = 10;
const MAX_SCHEMA_CHARS: usize = 6_000;
const MAX_DESC_CHARS: usize = 800;

/// The standard numbers a numeric argument is chosen among (plus its bounds / default).
const NUMBER_CANDIDATES: [f64; 6] = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0];

pub const SPECIFICITY: [&str; 3] = ["vague", "partial", "complete"];

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDef {
    /// Qualified name, `<server>__<tool>`.
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
}

// ------------------------------------------------------------ schema helpers

fn obj(v: &Value) -> Option<&Map<String, Value>> {
    v.as_object()
}

/// The `type` names a schema allows (handles `type` arrays and anyOf / oneOf).
fn types_of(s: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(t) = s.get("type") {
        match t {
            Value::String(x) => out.push(x.clone()),
            Value::Array(a) => out.extend(a.iter().filter_map(|x| x.as_str().map(String::from))),
            _ => {}
        }
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(a) = s.get(key).and_then(|a| a.as_array()) {
            for alt in a {
                out.extend(types_of(alt));
            }
        }
    }
    out
}

/// Fixed values of a schema (`enum`, or `const`s inside anyOf / oneOf), as JSON values.
fn enum_of(s: &Value) -> Vec<Value> {
    let mut out: Vec<Value> = s.get("enum").and_then(|e| e.as_array()).cloned().unwrap_or_default();
    if let Some(c) = s.get("const") {
        out.push(c.clone());
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(a) = s.get(key).and_then(|a| a.as_array()) {
            for alt in a {
                out.extend(enum_of(alt));
            }
        }
    }
    out.retain(|v| !v.is_null());
    let mut seen = HashSet::new();
    out.retain(|v| seen.insert(v.to_string()));
    out
}

pub fn label_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn has(types: &[String], t: &str) -> bool {
    types.iter().any(|x| x == t)
}

/// The standard numbers a numeric argument may take: its default and bounds (when it has them)
/// plus `NUMBER_CANDIDATES` inside the bounds, at most 8, ascending. Training and inference must
/// offer the same set, so this is the one place it is computed.
pub fn numeric_candidates(schema: &Value) -> Vec<f64> {
    let (min, max) = (schema.get("minimum").and_then(|v| v.as_f64()), schema.get("maximum").and_then(|v| v.as_f64()));
    let mut c: Vec<f64> = NUMBER_CANDIDATES.iter().copied().filter(|v| min.is_none_or(|m| *v >= m) && max.is_none_or(|m| *v <= m)).collect();
    for extra in [schema.get("default").and_then(|v| v.as_f64()), min, max].into_iter().flatten() {
        c.push(extra);
    }
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    c.dedup();
    // keep the bounds and default, thin the rest evenly
    while c.len() > 8 {
        let drop = (0..c.len()).find(|&i| i % 2 == 1).unwrap_or(1);
        c.remove(drop);
    }
    c
}

fn num_label(v: f64) -> String {
    if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") }
}

/// Checks `args` against the object schema `schema` (one nesting level deeper per recursion):
/// required arguments present, no unknown ones, right types, fixed values respected.
pub fn validate_args(schema: &Value, args: &Value, depth: usize) -> Result<(), String> {
    let Some(a) = obj(args) else { return Err("arguments must be an object".into()) };
    let props = schema.get("properties").and_then(|p| p.as_object());
    if let Some(req) = schema.get("required").and_then(|r| r.as_array()) {
        for r in req.iter().filter_map(|r| r.as_str()) {
            if a.get(r).is_none_or(|v| v.is_null()) {
                return Err(format!("required argument `{r}` is missing"));
            }
        }
    }
    let Some(props) = props else { return Ok(()) }; // free-form object
    for (k, v) in a {
        let Some(ps) = props.get(k) else {
            return Err(format!("unknown argument `{k}`"));
        };
        if v.is_null() {
            continue;
        }
        let types = types_of(ps);
        let ok = types.is_empty()
            || types.iter().any(|t| match t.as_str() {
                "string" => v.is_string(),
                "integer" => v.as_i64().is_some() || v.as_f64().is_some_and(|f| f.fract() == 0.0),
                "number" => v.is_number(),
                "boolean" => v.is_boolean(),
                "array" => v.is_array(),
                "object" => v.is_object(),
                "null" => v.is_null(),
                _ => true,
            });
        if !ok {
            return Err(format!("argument `{k}` has the wrong type"));
        }
        let fixed = enum_of(ps);
        if !fixed.is_empty() && !fixed.iter().any(|f| f == v) {
            return Err(format!("argument `{k}` must be one of the allowed values"));
        }
        if has(&types, "object") && ps.get("properties").is_some() && depth < 2 {
            validate_args(ps, v, depth + 1)?;
        }
    }
    Ok(())
}

// --------------------------------------------------------- sample building

pub struct Example {
    pub request: String,
    pub arguments: Value,
    /// Index into `SPECIFICITY`.
    pub specificity: usize,
}

fn qt(q: Question, t: Target) -> (Question, Target) {
    (q, t)
}

fn noul(text: String, p: f32) -> (Question, Target) {
    qt(Question::Noul { text }, Target::Noul { p })
}

fn choice(text: String, options: Vec<String>, gold: usize) -> (Question, Target) {
    let n = options.len();
    qt(Question::Choice { text, options }, Target::choice_hard(n, gold))
}

/// Questions for one argument (and, one level down, the members of an object argument).
#[allow(clippy::too_many_arguments)]
fn arg_questions(path: &str, schema: &Value, required: bool, value: Option<&Value>, depth: usize, out: &mut Vec<(Question, Target)>) {
    let present = value.is_some_and(|v| !v.is_null());
    if !required {
        out.push(noul(format!("{path}: is it specified?"), if present { 1.0 } else { 0.0 }));
    }
    let Some(v) = value.filter(|v| !v.is_null()) else { return };
    let types = types_of(schema);
    let fixed = enum_of(schema);
    if fixed.len() >= 2 {
        // fixed-value argument: choose among the allowed values (the right one is always offered)
        let mut vals: Vec<&Value> = fixed.iter().collect();
        if vals.len() > MAX_ENUM {
            let gold = vals.iter().position(|f| *f == v);
            let mut keep: Vec<&Value> = vals.iter().copied().take(MAX_ENUM).collect();
            if let (Some(g), false) = (gold, keep.iter().any(|k| *k == v)) {
                keep[MAX_ENUM - 1] = vals[g];
            }
            vals = keep;
        }
        if let Some(g) = vals.iter().position(|f| *f == v) {
            out.push(choice(format!("{path}: which value?"), vals.iter().map(|x| label_of(x)).collect(), g));
        }
    } else if has(&types, "boolean") && v.is_boolean() {
        out.push(noul(format!("{path}: is it true?"), if v.as_bool() == Some(true) { 1.0 } else { 0.0 }));
    } else if (has(&types, "integer") || has(&types, "number")) && v.is_number() {
        let cands = numeric_candidates(schema);
        let x = v.as_f64().unwrap_or(f64::NAN);
        if let Some(g) = cands.iter().position(|c| (*c - x).abs() < 1e-9) {
            if cands.len() >= 2 {
                out.push(choice(format!("{path}: which number?"), cands.iter().map(|c| num_label(*c)).collect(), g));
            }
        }
    } else if has(&types, "object") && depth < 1 {
        if let (Some(props), Some(map)) = (schema.get("properties").and_then(|p| p.as_object()), v.as_object()) {
            let req: HashSet<&str> = schema.get("required").and_then(|r| r.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
            for (k, ps) in props {
                arg_questions(&format!("{path}.{k}"), ps, req.contains(k.as_str()), map.get(k), depth + 1, out);
            }
        }
    }
    // free-text strings, arrays: kept in `Sample::call`, not expressible as a choice yet
}

// ----------------------------------------------- the same questions, from the schema alone

#[derive(Debug, Clone)]
pub enum ArgKind {
    /// Optional argument: is it specified?
    Present,
    Bool,
    Enum(Vec<Value>),
    /// Candidate numbers, and whether the argument is an integer.
    Number(Vec<f64>, bool),
}

/// One question about one argument, as asked at inference time (training asks the same ones, but
/// only when the answer exists in the example). `path` is the key path, e.g. `["filters", "kind"]`.
#[derive(Debug, Clone)]
pub struct ArgSpec {
    pub path: Vec<String>,
    pub text: String,
    pub kind: ArgKind,
    pub required: bool,
}

impl ArgSpec {
    pub fn question(&self) -> Question {
        match &self.kind {
            ArgKind::Present | ArgKind::Bool => Question::Noul { text: self.text.clone() },
            ArgKind::Enum(vals) => Question::Choice { text: self.text.clone(), options: vals.iter().map(label_of).collect() },
            ArgKind::Number(c, _) => Question::Choice { text: self.text.clone(), options: c.iter().map(|v| num_label(*v)).collect() },
        }
    }
}

/// Every question the model can be asked about a tool's arguments, in schema order. Text and
/// options match what `build_sample` writes for training, so the model recognises them.
pub fn arg_specs(schema: &Value) -> Vec<ArgSpec> {
    fn walk(prefix: &[String], schema: &Value, depth: usize, out: &mut Vec<ArgSpec>) {
        let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else { return };
        let req: HashSet<&str> = schema.get("required").and_then(|r| r.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
        for (k, ps) in props {
            let mut path = prefix.to_vec();
            path.push(k.clone());
            let name = path.join(".");
            let required = req.contains(k.as_str());
            if !required {
                out.push(ArgSpec { path: path.clone(), text: format!("{name}: is it specified?"), kind: ArgKind::Present, required });
            }
            let (types, fixed) = (types_of(ps), enum_of(ps));
            if fixed.len() >= 2 {
                let vals: Vec<Value> = fixed.into_iter().take(MAX_ENUM).collect();
                out.push(ArgSpec { path, text: format!("{name}: which value?"), kind: ArgKind::Enum(vals), required });
            } else if has(&types, "boolean") {
                out.push(ArgSpec { path, text: format!("{name}: is it true?"), kind: ArgKind::Bool, required });
            } else if has(&types, "integer") || has(&types, "number") {
                let c = numeric_candidates(ps);
                if c.len() >= 2 {
                    out.push(ArgSpec { path, text: format!("{name}: which number?"), kind: ArgKind::Number(c, !has(&types, "number")), required });
                }
            } else if has(&types, "object") && depth < 1 {
                walk(&path, ps, depth + 1, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(&[], schema, 0, &mut out);
    out
}

/// Turns one validated example into a System One sample. `catalogue` is every tool the dataset
/// is meant to tell apart; distractors for the "which tool" question come from it.
pub fn build_sample(tool: &ToolDef, ex: &Example, catalogue: &[ToolDef], distractors: usize) -> Sample {
    let mut rng = StdRng::seed_from_u64(ex.request.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3)) ^ tool.name.len() as u64);
    let mut qs: Vec<(Question, Target)> = Vec::new();

    let mut others: Vec<&ToolDef> = catalogue.iter().filter(|t| t.name != tool.name).collect();
    others.shuffle(&mut rng);
    let picked: Vec<&ToolDef> = others.into_iter().take(distractors).collect();
    if !picked.is_empty() {
        let mut names: Vec<&str> = picked.iter().map(|t| t.name.as_str()).collect();
        names.push(&tool.name);
        names.shuffle(&mut rng);
        let gold = names.iter().position(|n| *n == tool.name).unwrap_or(0);
        qs.push(choice("Which tool fits this request?".into(), names.iter().map(|n| n.to_string()).collect(), gold));
        // one right and one wrong tool, each as a yes/no — teaches the model to say no
        qs.push(noul(format!("{} — is this the right tool?", tool.name), 1.0));
        qs.push(noul(format!("{} — is this the right tool?", picked[0].name), 0.0));
    }
    qs.push(choice_score("How completely does the request specify the call?", ex.specificity));

    let schema = &tool.input_schema;
    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        let req: HashSet<&str> = schema.get("required").and_then(|r| r.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
        let mut args: Vec<(Question, Target)> = Vec::new();
        for (k, ps) in props {
            arg_questions(k, ps, req.contains(k.as_str()), ex.arguments.get(k), 0, &mut args);
        }
        args.shuffle(&mut rng);
        args.truncate(MAX_ARG_QUESTIONS);
        qs.extend(args);
    }
    qs.shuffle(&mut rng); // no question always sits at the same position
    let (questions, targets): (Vec<_>, Vec<_>) = qs.into_iter().unzip();
    Sample { id: 0, state: ex.request.clone(), questions, targets, source: format!("tool:{}", tool.name), call: Some(ToolCall { tool: tool.name.clone(), arguments: ex.arguments.clone() }) }
}

fn choice_score(text: &str, level: usize) -> (Question, Target) {
    let mut probs = vec![0.0; SPECIFICITY.len()];
    probs[level.min(SPECIFICITY.len() - 1)] = 1.0;
    qt(Question::Score { text: text.into(), levels: SPECIFICITY.iter().map(|s| s.to_string()).collect() }, Target::Score { probs })
}

// ------------------------------------------------------------- the model

fn system_prompt(n: usize) -> String {
    format!(
        "You write training data for a tool-calling model. You are given ONE tool (name, description, \
JSON Schema of its arguments). Invent {n} DIFFERENT realistic user requests that should be answered by \
calling exactly that tool, each with the exact arguments the call should have.\n\
Respond with ONLY a single JSON object, no markdown fences and no text around it, shaped exactly like:\n\
{{\"examples\": [{{\"request\": string, \"arguments\": object, \"specificity\": \"vague\" | \"partial\" | \"complete\"}}]}}\n\n\
Rules:\n\
- \"request\" is what a person would actually type to an assistant: natural, varied in length, tone and \
detail. Never mention the tool, its name, or JSON.\n\
- \"arguments\" must follow the schema exactly: every required argument present, only argument names \
that exist in the schema, the right JSON types, and values from the allowed list wherever the schema \
gives one. Values must come from the request itself (or be the schema's default when the request \
doesn't say) — never invent identifiers the request doesn't contain.\n\
- Vary which optional arguments are used across examples; include some examples that use only the \
required ones.\n\
- \"specificity\" says how completely the request pins down the call: \"complete\" = every value the \
call needs is stated, \"partial\" = some values are implied or left to defaults, \"vague\" = the request \
barely says what is wanted.\n\
- Use English unless the tool description is in another language."
    )
}

fn user_prompt(tool: &ToolDef, n: usize, avoid: &[String]) -> String {
    let desc: String = tool.description.chars().take(MAX_DESC_CHARS).collect();
    let mut schema = serde_json::to_string_pretty(&tool.input_schema).unwrap_or_default();
    if schema.len() > MAX_SCHEMA_CHARS {
        schema = serde_json::to_string(&tool.input_schema).unwrap_or_default().chars().take(MAX_SCHEMA_CHARS).collect();
    }
    let mut p = format!("Tool name: {}\n\nDescription:\n{desc}\n\nArguments schema:\n{schema}\n\nWrite {n} examples.", tool.name);
    if !avoid.is_empty() {
        p.push_str("\n\nAlready written — do not repeat these requests:\n");
        for r in avoid.iter().rev().take(12) {
            p.push_str(&format!("- {}\n", r.chars().take(160).collect::<String>()));
        }
    }
    p
}

/// Lenient extraction of `{"examples": [...]}` (or a bare array); examples whose arguments don't
/// satisfy the tool's schema are dropped.
pub fn parse_examples(raw: &str, tool: &ToolDef) -> Vec<Example> {
    let array = || {
        let (a, b) = (raw.find('[')?, raw.rfind(']')?);
        serde_json::from_str::<Value>(raw.get(a..=b)?).ok()
    };
    let object = || learn::balanced_json_object(raw).and_then(|o| serde_json::from_str::<Value>(o).ok());
    let array_first = matches!((raw.find('['), raw.find('{')), (Some(a), Some(o)) if a < o);
    let Some(v) = (if array_first { array().or_else(object) } else { object().or_else(array) }) else { return Vec::new() };
    let items = match &v {
        Value::Array(a) => Some(a),
        other => other.get("examples").and_then(|e| e.as_array()),
    };
    items
        .into_iter()
        .flatten()
        .filter_map(|it| {
            let request = it.get("request").and_then(|r| r.as_str())?.trim().to_string();
            if request.len() < 3 || request.len() > 1500 {
                return None;
            }
            // the model sometimes returns the arguments as a JSON string
            let mut arguments = it.get("arguments").cloned().unwrap_or_else(|| json!({}));
            if let Some(s) = arguments.as_str() {
                arguments = serde_json::from_str(s).ok()?;
            }
            if let Some(m) = arguments.as_object_mut() {
                m.retain(|_, v| !v.is_null());
            }
            validate_args(&tool.input_schema, &arguments, 0).ok()?;
            let specificity = it.get("specificity").and_then(|s| s.as_str()).and_then(|s| SPECIFICITY.iter().position(|x| s.eq_ignore_ascii_case(x))).unwrap_or(2);
            Some(Example { request, arguments, specificity })
        })
        .collect()
}

/// Generates `n` samples for one tool (several model calls of up to `BATCH` examples each, with
/// retries), returning whatever it managed to produce; an error only if it got none.
pub async fn generate_for_tool(client: &reqwest::Client, llm: &LlmConfig, tool: &ToolDef, catalogue: &[ToolDef], n: usize, distractors: usize, cancel: &AtomicBool) -> Result<Vec<Sample>, String> {
    let mut out: Vec<Sample> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut failures = 0;
    let mut last_err = String::new();
    while out.len() < n && failures < 3 {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let want = (n - out.len()).min(BATCH);
        let messages = vec![json!({"role": "system", "content": system_prompt(want)}), json!({"role": "user", "content": user_prompt(tool, want, &seen)})];
        match learn::complete(client, llm, messages, Duration::from_secs(120)).await {
            Ok(content) => {
                let fresh: Vec<Example> = parse_examples(&content, tool).into_iter().filter(|e| !seen.iter().any(|s| s.eq_ignore_ascii_case(&e.request))).collect();
                if fresh.is_empty() {
                    failures += 1;
                    last_err = "The model didn't return any usable examples (arguments that match the tool's schema).".into();
                    continue;
                }
                for e in fresh.into_iter().take(n - out.len()) {
                    seen.push(e.request.clone());
                    let s = build_sample(tool, &e, catalogue, distractors);
                    if s.validate().is_ok() {
                        out.push(s);
                    }
                }
            }
            Err(e) if e.fatal => return Err(e.msg),
            Err(e) => {
                failures += 1;
                last_err = e.msg;
                tokio::time::sleep(Duration::from_millis(800)).await;
            }
        }
    }
    if out.is_empty() {
        Err(if last_err.is_empty() { "cancelled".into() } else { last_err })
    } else {
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, schema: Value) -> ToolDef {
        ToolDef { name: name.into(), description: "d".into(), input_schema: schema }
    }

    fn recent_rows() -> ToolDef {
        tool(
            "ft__recent_rows",
            json!({"type":"object","required":["limit"],"properties":{
                "limit":{"type":"integer","default":20,"minimum":1,"maximum":100},
                "sort_desc":{"type":"boolean"},
                "kind":{"type":"string","enum":["IFC","FTD","OTHER"]},
                "note":{"type":"string"},
                "filters":{"type":"object","properties":{"SYSTEM_TRACK_TYPE":{"type":"string","enum":["IFC","FTD"]}}}}}),
        )
    }

    fn ping() -> ToolDef {
        tool("ft__ping", json!({"type":"object","properties":{}}))
    }

    #[test]
    fn validates_arguments_against_the_schema() {
        let t = recent_rows();
        assert!(validate_args(&t.input_schema, &json!({"limit": 20, "kind": "IFC"}), 0).is_ok());
        assert!(validate_args(&t.input_schema, &json!({}), 0).unwrap_err().contains("limit"));
        assert!(validate_args(&t.input_schema, &json!({"limit": "x"}), 0).unwrap_err().contains("type"));
        assert!(validate_args(&t.input_schema, &json!({"limit": 1, "kind": "NOPE"}), 0).unwrap_err().contains("allowed"));
        assert!(validate_args(&t.input_schema, &json!({"limit": 1, "ghost": 1}), 0).unwrap_err().contains("unknown"));
        assert!(validate_args(&t.input_schema, &json!({"limit": 1, "filters": {"SYSTEM_TRACK_TYPE": "ZZZ"}}), 0).is_err());
    }

    #[test]
    fn numeric_candidates_respect_bounds_and_default_and_are_stable() {
        let c = numeric_candidates(&json!({"minimum": 1, "maximum": 100, "default": 20}));
        assert_eq!(c, vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0]);
        let c = numeric_candidates(&json!({"minimum": 7, "maximum": 30, "default": 12}));
        assert!(c.contains(&7.0) && c.contains(&30.0) && c.contains(&12.0) && c.iter().all(|v| (7.0..=30.0).contains(v)));
        assert_eq!(numeric_candidates(&json!({})), NUMBER_CANDIDATES.to_vec());
        assert!(numeric_candidates(&json!({"default": 37})).len() <= 8);
    }

    #[test]
    fn a_call_becomes_typed_questions_with_exact_targets() {
        let t = recent_rows();
        let cat = vec![t.clone(), ping(), tool("ft__other", json!({"type":"object","properties":{}}))];
        let ex = Example { request: "show me the latest 20 IFC tracks, newest first".into(), arguments: json!({"limit": 20, "sort_desc": true, "kind": "IFC", "filters": {"SYSTEM_TRACK_TYPE": "IFC"}, "note": "free text"}), specificity: 2 };
        let s = build_sample(&t, &ex, &cat, 2);
        s.validate().unwrap();
        assert_eq!(s.source, "tool:ft__recent_rows");
        assert_eq!(s.call.as_ref().unwrap().arguments["limit"], 20);

        let find = |needle: &str| s.questions.iter().zip(&s.targets).find(|(q, _)| q.text().contains(needle));
        // which tool: the right one among the options, as a one-hot
        let (q, tg) = find("Which tool").unwrap();
        let Target::Choice { probs } = tg else { panic!() };
        assert_eq!(q.options()[probs.iter().position(|p| *p == 1.0).unwrap()], "ft__recent_rows");
        assert_eq!(q.options().len(), 3);
        // right tool yes, a wrong tool no
        let relevance: Vec<f32> = s.questions.iter().zip(&s.targets).filter(|(q, _)| q.text().contains("right tool")).map(|(_, t)| if let Target::Noul { p } = t { *p } else { -1.0 }).collect();
        assert_eq!(relevance.iter().filter(|p| **p == 1.0).count(), 1);
        assert_eq!(relevance.iter().filter(|p| **p == 0.0).count(), 1);
        // required `limit` isn't asked about as present/absent; its value is a number choice
        assert!(find("limit: is it specified").is_none());
        let (q, tg) = find("limit: which number").unwrap();
        let Target::Choice { probs } = tg else { panic!() };
        assert_eq!(q.options()[probs.iter().position(|p| *p == 1.0).unwrap()], "20");
        // optional + boolean + enum + nested
        assert!(matches!(find("sort_desc: is it specified").unwrap().1, Target::Noul { p } if *p == 1.0));
        assert!(matches!(find("sort_desc: is it true").unwrap().1, Target::Noul { p } if *p == 1.0));
        assert!(find("kind: which value").is_some());
        assert!(find("filters.SYSTEM_TRACK_TYPE: which value").is_some());
        // free text isn't a question
        assert!(find("note: which").is_none());
        // specificity score: "complete"
        let (_, tg) = find("How completely").unwrap();
        assert!(matches!(tg, Target::Score { probs } if probs[2] == 1.0));
    }

    #[test]
    fn inference_questions_match_the_training_ones() {
        let t = recent_rows();
        let specs: HashSet<(String, Vec<String>)> = arg_specs(&t.input_schema).iter().map(|s| (s.text.clone(), s.question().options().to_vec())).collect();
        let ex = Example { request: "x".into(), arguments: json!({"limit": 20, "sort_desc": true, "kind": "IFC", "filters": {"SYSTEM_TRACK_TYPE": "IFC"}}), specificity: 2 };
        let s = build_sample(&t, &ex, &[t.clone()], 0);
        for q in &s.questions {
            if q.text().contains("Which tool") || q.text().contains("How completely") {
                continue;
            }
            assert!(specs.contains(&(q.text().to_string(), q.options().to_vec())), "training asks {:?} but inference never would", q.text());
        }
        // required limit: no presence question; optional ones have one
        assert!(!specs.iter().any(|(t, _)| t == "limit: is it specified?"));
        assert!(specs.iter().any(|(t, _)| t == "filters.SYSTEM_TRACK_TYPE: is it specified?"));
    }

    #[test]
    fn a_missing_optional_argument_is_a_no_and_asks_nothing_more() {
        let t = recent_rows();
        let ex = Example { request: "latest rows".into(), arguments: json!({"limit": 10}), specificity: 1 };
        let s = build_sample(&t, &ex, &[t.clone()], 3);
        s.validate().unwrap();
        let no_kind = s.questions.iter().zip(&s.targets).find(|(q, _)| q.text().contains("kind: is it specified")).unwrap();
        assert!(matches!(no_kind.1, Target::Noul { p } if *p == 0.0));
        assert!(!s.questions.iter().any(|q| q.text().contains("kind: which value")));
        // only one tool in the catalogue: no distractors, so no tool question
        assert!(!s.questions.iter().any(|q| q.text().contains("Which tool")));
    }

    #[test]
    fn parses_examples_and_drops_the_ones_that_break_the_schema() {
        let t = recent_rows();
        let raw = r#"```json
        {"examples":[
          {"request":"latest 5 rows please","arguments":{"limit":5},"specificity":"complete"},
          {"request":"missing the required limit","arguments":{"kind":"IFC"},"specificity":"vague"},
          {"request":"bad enum","arguments":"{\"limit\": 3, \"kind\": \"WAT\"}"},
          {"request":"args as a string","arguments":"{\"limit\": 7}","specificity":"Partial"}
        ]}
        ```"#;
        let got = parse_examples(raw, &t);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].specificity, 2);
        assert_eq!(got[1].specificity, 1);
        assert_eq!(got[1].arguments["limit"], 7);
    }

    #[tokio::test]
    async fn generates_samples_for_a_tool_against_a_model_server() {
        use axum::{routing::post, Json, Router};
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|Json(body): Json<Value>| async move {
                assert!(body["messages"][1]["content"].as_str().unwrap().contains("ft__recent_rows"));
                let ex = json!({"examples": [
                    {"request": "latest 5 rows", "arguments": {"limit": 5}, "specificity": "complete"},
                    {"request": "give me 20 newest IFC rows", "arguments": {"limit": 20, "kind": "IFC", "sort_desc": true}, "specificity": "complete"},
                    {"request": "broken", "arguments": {"limit": "lots"}}]});
                Json(json!({"choices": [{"message": {"content": ex.to_string()}}]}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let llm = LlmConfig { base_url: format!("http://{addr}/v1"), api_key: "k".into(), model: "m".into() };
        let t = recent_rows();
        let cat = vec![t.clone(), ping()];
        let cancel = AtomicBool::new(false);
        let got = generate_for_tool(&reqwest::Client::new(), &llm, &t, &cat, 2, 3, &cancel).await.unwrap();
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|s| s.validate().is_ok() && s.call.is_some()));
    }
}
