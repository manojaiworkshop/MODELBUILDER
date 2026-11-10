"use client";
import { useEffect, useRef } from "react";
import { backendUrl, call, getSocket } from "./backend";
import { DEFAULT_TOKENIZER, isRunning, useStudio } from "./store";
import { blockLabel, CATALOG } from "./blocks";
import { qualifiedToolName } from "./mcp";
import type {
  McpServer, McpTool, ToolCallResult,
  BuildProgress, DataStyle, DataInfo, DocInfo, EvalPoint, HfDatasetInfo, InspectPayload, LearnAnswer, LlmConfig, QaPageList, QaPair, QaProgress, SamplePayload,
  LayaStatus, StatusPayload, StepPoint, SummaryPair, SummaryProgress, SysOneAnswer, SysOneEval, SysOneGenProgress, SysOneMix, SysOneQuestion, SysOneSample, SysOneStep, TokenizerInfo,
} from "./types";

/** Keeps the backend in step with the UI. Mount once. */
export function useBackendSync() {
  const sentText = useRef<string | null>(null);

  // ---- socket lifecycle + training events -------------------------------
  useEffect(() => {
    const s = getSocket();
    const st = useStudio.getState;
    const onConnect = () => {
      sentText.current = null; // new server session has no dataset
      st().setConn("connected");
      st().setDataInfo(null);
      st().setTokenizer(DEFAULT_TOKENIZER); // the new server session starts with the built-in one
      st().setDocs([]);
      st().setProgress(null);
      loadMissingMcpTools(); // servers added while the backend was down / outdated get their tools now
      st().setQa([]);
      st().setQaProgress(null);
      st().setSummaries([]);
      st().setSummaryProgress(null);
      if (st().trainSource === "qa" || st().trainSource === "summary") st().setTrainSource("text");
      if (st().trainMode === "continue") st().setTrainMode("new"); // the new session has no earlier weights
      st().setSysoneSamples([]);
      st().setSysoneGenProgress(null);
      st().setSysoneTrained(false); // the new server session has no earlier System One weights either
      st().setSysoneStatus({ state: "idle", message: "", step: 0, total: 0 });
    };
    const onDisconnect = () => st().setConn("disconnected");
    const onConnectError = () => st().setConn("disconnected");
    const onStatus = (p: StatusPayload) => st().setStatus(p);
    const onSteps = (p: StepPoint[]) => st().pushSteps(p);
    const onEval = (p: EvalPoint) => st().pushEval(p);
    const onSample = (p: SamplePayload) => st().pushSample(p);
    const onInspect = (p: InspectPayload) => {
      st().setInspect(p);
      st().setGradSnap(p);
    };
    const onToken = (t: { text: string }) => st().appendGen(t.text);
    const onQaList = (p: QaPair[]) => st().setQa(p);
    const onQaAdded = (p: QaPair[]) => st().addQa(p);
    const onQaProgress = (p: QaProgress) => st().setQaProgress(p);
    const onSummaryList = (p: SummaryPair[]) => st().setSummaries(p);
    const onSummaryAdded = (p: SummaryPair[]) => st().addSummaries(p);
    const onSummaryProgress = (p: SummaryProgress) => st().setSummaryProgress(p);
    const onSysoneList = (p: SysOneSample[]) => st().setSysoneSamples(p);
    const onSysoneAdded = (p: SysOneSample[]) => st().addSysoneSamples(p);
    const onSysoneGenProgress = (p: SysOneGenProgress) => st().setSysoneGenProgress(p);
    const onLayaStatus = (p: LayaStatus) => st().setLayaStatus(p);
    const onSysoneTrainStatus = (p: StatusPayload) => {
      st().setSysoneStatus(p);
      if (p.state === "done") st().setSysoneTrained(true);
    };
    const onSysoneTrainStep = (p: SysOneStep) => st().pushSysoneStep(p);
    const onSysoneTrainEval = (p: SysOneEval) => st().pushSysoneEval(p);
    const onDocs = (p: { docs: DocInfo[] }) => st().setDocs(p.docs);
    const onProgress = (p: BuildProgress) => st().setProgress(p.stage === "done" ? null : p);
    const onTokenizer = (p: TokenizerInfo) => st().setTokenizer(p);
    const onInfo = (p: DataInfo) => {
      st().setProgress(null);
      st().setDataInfo(p.ok ? p : null);
      if (!p.ok) {
        st().setInspect(null);
        st().setDataError(p.error);
      } else st().setDataError(null);
    };

    s.on("data:tokenizer", onTokenizer);
    s.on("connect", onConnect);
    s.on("disconnect", onDisconnect);
    s.on("connect_error", onConnectError);
    s.on("train:status", onStatus);
    s.on("train:steps", onSteps);
    s.on("train:eval", onEval);
    s.on("train:sample", onSample);
    s.on("train:inspect", onInspect);
    s.on("generate:token", onToken);
    s.on("data:docs", onDocs);
    s.on("qa:list", onQaList);
    s.on("qa:added", onQaAdded);
    s.on("qa:progress", onQaProgress);
    s.on("summarize:list", onSummaryList);
    s.on("summarize:added", onSummaryAdded);
    s.on("summarize:progress", onSummaryProgress);
    s.on("sysone:list", onSysoneList);
    s.on("sysone:added", onSysoneAdded);
    s.on("sysone:progress", onSysoneGenProgress);
    s.on("sysone:train:status", onSysoneTrainStatus);
    s.on("sysone:laya:status", onLayaStatus);
    s.on("sysone:train:step", onSysoneTrainStep);
    s.on("sysone:train:eval", onSysoneTrainEval);
    s.on("data:progress", onProgress);
    s.on("data:info", onInfo);
    if (s.connected) onConnect();
    return () => {
      s.off("connect", onConnect);
      s.off("disconnect", onDisconnect);
      s.off("connect_error", onConnectError);
      s.off("train:status", onStatus);
      s.off("train:steps", onSteps);
      s.off("train:eval", onEval);
      s.off("train:sample", onSample);
      s.off("train:inspect", onInspect);
      s.off("generate:token", onToken);
      s.off("data:docs", onDocs);
      s.off("data:tokenizer", onTokenizer);
      s.off("qa:list", onQaList);
      s.off("qa:added", onQaAdded);
      s.off("qa:progress", onQaProgress);
      s.off("summarize:list", onSummaryList);
      s.off("summarize:added", onSummaryAdded);
      s.off("summarize:progress", onSummaryProgress);
      s.off("sysone:list", onSysoneList);
      s.off("sysone:added", onSysoneAdded);
      s.off("sysone:progress", onSysoneGenProgress);
      s.off("sysone:train:status", onSysoneTrainStatus);
      s.off("sysone:laya:status", onLayaStatus);
      s.off("sysone:train:step", onSysoneTrainStep);
      s.off("sysone:train:eval", onSysoneTrainEval);
      s.off("data:progress", onProgress);
      s.off("data:info", onInfo);
    };
  }, []);

  const conn = useStudio((s) => s.conn);
  const text = useStudio((s) => s.text);
  const merges = useStudio((s) => s.merges);
  const usePasted = useStudio((s) => s.usePasted);

  // ---- dataset + tokenizer -----------------------------------------------
  // The backend rebuilds in the background and pushes `data:progress` / `data:info`.
  useEffect(() => {
    if (conn !== "connected") return;
    const t = setTimeout(async () => {
      try {
        const payload: { text?: string; merges: number; usePasted: boolean } = { merges, usePasted };
        if (sentText.current !== text) payload.text = text;
        const res = await call<{ ok: boolean; error?: string }>("data:set", payload);
        if (res.ok) sentText.current = text;
        else useStudio.getState().setDataError(res.error ?? "The backend rejected the text.");
      } catch (e) {
        useStudio.getState().setDataError((e as Error).message);
      }
    }, 450);
    return () => clearTimeout(t);
  }, [conn, text, merges, usePasted]);

  // ---- live inspection while building ------------------------------------
  const blocks = useStudio((s) => s.blocks);
  const probe = useStudio((s) => s.probe);
  const seqLen = useStudio((s) => s.hyper.seqLen);
  const dataRev = useStudio((s) => s.dataRev);
  const dataReady = useStudio((s) => !!s.dataInfo);
  const runState = useStudio((s) => s.status.state);
  const running = isRunning({ state: runState, message: "", step: 0, total: 0 });
  const seq = useRef(0);

  useEffect(() => {
    if (conn !== "connected" || !dataReady || running) return;
    const my = ++seq.current;
    const t = setTimeout(async () => {
      const st = useStudio.getState();
      st.setInspectBusy(true);
      try {
        const res = await call<InspectPayload | { ok: false; error: string }>("model:inspect", { spec: { blocks }, probe, seqLen }, 30_000);
        if (my !== seq.current) return; // a newer request superseded this one
        if ("layers" in res) st.setInspect(res);
        else st.setInspect(null);
      } catch {
        /* keep the previous picture */
      } finally {
        if (my === seq.current) st.setInspectBusy(false);
      }
    }, 220);
    return () => clearTimeout(t);
    // runState changes (e.g. finished) re-inspect so the trained weights are shown
  }, [conn, blocks, probe, seqLen, dataRev, dataReady, running, runState]);

  // ---- probe sentence follows edits during a run --------------------------
  useEffect(() => {
    if (!running) return;
    const t = setTimeout(() => call("probe:set", { text: probe }).catch(() => {}), 200);
    return () => clearTimeout(t);
  }, [probe, running]);
}

// ---- commands ----------------------------------------------------------------
export async function startTraining() {
  const st = useStudio.getState();
  st.resetRun();
  st.setGen({ text: "", error: null });
  try {
    // "continue" only makes sense for Q&A pairs / summaries and when there are weights to build on (the bar shows the same rule)
    const canContinue = st.trainSource === "qa" || st.trainSource === "summary";
    const mode = canContinue && st.trainMode === "continue" && st.inspect?.trained ? "continue" : "new";
    let probe = st.probe;
    // sample from a prompt shaped like the training text (in the first pair's style), so the samples show what it learned
    const shaped = PROBE_PREFIXES.some((p) => probe.startsWith(p));
    if (st.trainSource === "qa" && st.qa[0] && !shaped) {
      probe = qaProbe(st.qa[0]);
      st.setProbe(probe);
    } else if (st.trainSource === "summary" && st.summaries[0] && !shaped) {
      probe = summaryProbe(st.summaries[0]);
      st.setProbe(probe);
    }
    const res = await call<{ ok: boolean; error?: string }>("train:start", { spec: { blocks: st.blocks }, hyper: st.hyper, probe, source: st.trainSource, mode });
    if (!res.ok) st.setStatus({ state: "error", message: res.error ?? "Could not start training", step: 0, total: 0 });
  } catch (e) {
    st.setStatus({ state: "error", message: (e as Error).message, step: 0, total: 0 });
  }
}

const PROBE_PREFIXES = ["Question:", "Context:", "Text:", "Article:", "### Instruction:", "<|system|>", "<|user|>"];

const chatHead = (system: string) => (system ? `<|system|>\n${system}\n` : "");

/** The start of a training block for this pair, in its style, ending where the answer begins. */
function qaProbe(p: QaPair): string {
  if (p.style === "hf" && p.contextHead) return `Context: ${p.contextHead}\nQuestion: ${p.question}\nAnswer:`;
  if (p.style === "instruction") return `### Instruction:\n${p.question}\n\n${p.contextHead ? `### Input:\n${p.contextHead}\n\n` : ""}### Response:\n`;
  if (p.style === "chat") return `${chatHead(p.system)}<|user|>\n${p.question}\n<|assistant|>\n`;
  return `Question: ${p.question}\nAnswer:`;
}

function summaryProbe(p: SummaryPair): string {
  const t = p.text.slice(0, 200);
  if (p.style === "hf") return `Article: ${t}\nHighlights:`;
  if (p.style === "instruction") return `### Instruction:\nSummarize the following text.\n\n### Input:\n${t}\n\n### Response:\n`;
  if (p.style === "chat") return `${chatHead(p.system)}<|user|>\nSummarize the following text:\n\n${t}\n<|assistant|>\n`;
  return `Text: ${t}\nSummary:`;
}

// ---- Q&A dataset ------------------------------------------------------------

export async function loadQaPages(docId: string): Promise<{ ok: true; list: QaPageList } | { ok: false; error: string }> {
  try {
    const res = await call<({ ok: true } & QaPageList) | { ok: false; error: string }>("qa:pages", { id: docId }, 60_000);
    if (!res.ok) return { ok: false, error: res.error };
    return { ok: true, list: { pages: res.pages, synthetic: res.synthetic, truncated: res.truncated } };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

export async function generateQa(docId: string, pages: number[], perPage: number, concurrency: number, style: DataStyle, system: string): Promise<string | null> {
  const st = useStudio.getState();
  st.setQaProgress({ state: "running", done: 0, total: pages.length, pairs: 0, failed: 0, message: "" });
  try {
    const res = await call<{ ok: boolean; error?: string }>("qa:generate", { llm: st.llm, id: docId, pages, perPage, style, system, concurrency });
    if (!res.ok) {
      st.setQaProgress(null);
      return res.error ?? "Could not start generation.";
    }
    return null;
  } catch (e) {
    st.setQaProgress(null);
    return (e as Error).message;
  }
}

export const stopQa = () => call("qa:stop").catch(() => {});
export const removeQa = (ids: number[]) => call("qa:remove", { ids }).catch(() => {});
export const clearQa = () => call("qa:remove", { ids: [] }).catch(() => {});

// ---- Summary dataset --------------------------------------------------------
// Page listing is shared with Q&A (`loadQaPages` — it's just the document's page boundaries).

export async function generateSummaries(docId: string, pages: number[], sentences: number, concurrency: number, style: DataStyle, system: string): Promise<string | null> {
  const st = useStudio.getState();
  st.setSummaryProgress({ state: "running", done: 0, total: pages.length, summaries: 0, failed: 0, message: "" });
  try {
    const res = await call<{ ok: boolean; error?: string }>("summarize:generate", { llm: st.llm, id: docId, pages, sentences, style, system, concurrency });
    if (!res.ok) {
      st.setSummaryProgress(null);
      return res.error ?? "Could not start generation.";
    }
    return null;
  } catch (e) {
    st.setSummaryProgress(null);
    return (e as Error).message;
  }
}

export const stopSummarize = () => call("summarize:stop").catch(() => {});
export const removeSummaries = (ids: number[]) => call("summarize:remove", { ids }).catch(() => {});
export const clearSummaries = () => call("summarize:remove", { ids: [] }).catch(() => {});

// ---- System One dataset ------------------------------------------------------
// Page listing is shared with Q&A (`loadQaPages` — it's just the document's page boundaries).

export async function generateSysOne(docId: string, pages: number[], perPage: number, concurrency: number): Promise<string | null> {
  const st = useStudio.getState();
  st.setSysoneGenProgress({ state: "running", done: 0, total: pages.length, samples: 0, failed: 0, message: "" });
  try {
    const res = await call<{ ok: boolean; error?: string }>("sysone:generate", { llm: st.llm, id: docId, pages, perPage, concurrency });
    if (!res.ok) {
      st.setSysoneGenProgress(null);
      return res.error ?? "Could not start generation.";
    }
    return null;
  } catch (e) {
    st.setSysoneGenProgress(null);
    return (e as Error).message;
  }
}

/** Writes tool-calling samples for the picked tools (qualified names), `concurrency` tools at a time. */
export async function generateToolSamples(picked: string[], perTool: number, distractors: number, concurrency: number): Promise<string | null> {
  const st = useStudio.getState();
  const want = new Set(picked);
  const tools: { name: string; description: string; inputSchema: Record<string, unknown> }[] = [];
  const catalogue: typeof tools = [];
  for (const srv of st.mcpServers) {
    if (!srv.enabled) continue;
    for (const t of st.mcpTools[srv.id] ?? []) {
      const def = { name: qualifiedToolName(srv, t), description: t.description ?? "", inputSchema: t.inputSchema };
      catalogue.push(def);
      if (want.has(def.name)) tools.push(def);
    }
  }
  if (tools.length === 0) return "Pick at least one tool.";
  st.setSysoneGenProgress({ state: "running", done: 0, total: tools.length, samples: 0, failed: 0, message: "" });
  try {
    const res = await call<{ ok: boolean; error?: string }>("sysone:generate:tools", { llm: st.llm, tools, catalogue, perTool, distractors, concurrency });
    if (!res.ok) {
      st.setSysoneGenProgress(null);
      return res.error ?? "Could not start generation.";
    }
    return null;
  } catch (e) {
    st.setSysoneGenProgress(null);
    return (e as Error).message;
  }
}

export const stopSysOneGenerate = () => call("sysone:stop").catch(() => {});
export const removeSysOne = (ids: number[]) => call("sysone:remove", { ids }).catch(() => {});
export const clearSysOne = () => call("sysone:remove", { ids: [] }).catch(() => {});

/** No source document: the model invents samples for a domain + free-form seed context. */
export async function generateSysOneSynthetic(domain: string, context: string, mix: SysOneMix, total: number, perBatch: number, concurrency: number): Promise<string | null> {
  const st = useStudio.getState();
  const batches = Math.ceil(total / Math.max(1, perBatch));
  st.setSysoneGenProgress({ state: "running", done: 0, total: batches, samples: 0, failed: 0, message: "" });
  try {
    const res = await call<{ ok: boolean; error?: string }>("sysone:generate:synthetic", { llm: st.llm, domain, context, mix, total, perBatch, concurrency });
    if (!res.ok) {
      st.setSysoneGenProgress(null);
      return res.error ?? "Could not start generation.";
    }
    return null;
  } catch (e) {
    st.setSysoneGenProgress(null);
    return (e as Error).message;
  }
}

// ---- System One training + inference -----------------------------------------

export async function startSysOneTrain() {
  const st = useStudio.getState();
  st.resetSysoneRun();
  st.setSysoneInfer({ answers: null, error: null });
  try {
    const res = await call<{ ok: boolean; error?: string }>("sysone:train:start", { hyper: st.sysoneHyper });
    if (!res.ok) st.setSysoneStatus({ state: "error", message: res.error ?? "Could not start training", step: 0, total: 0 });
  } catch (e) {
    st.setSysoneStatus({ state: "error", message: (e as Error).message, step: 0, total: 0 });
  }
}

/** Asks the server whether Laya is already on disk (call when the Training tab opens). */
export async function checkLaya() {
  try {
    const r = await call<{ ok: boolean; status: LayaStatus }>("sysone:laya:status");
    if (r.ok) useStudio.getState().setLayaStatus(r.status);
  } catch {}
}

export async function downloadLaya() {
  const st = useStudio.getState();
  st.setLayaStatus({ state: "downloading", done: 0, total: 0, message: "" });
  try {
    const r = await call<{ ok: boolean; error?: string }>("sysone:laya:download");
    if (!r.ok) st.setLayaStatus({ state: "error", done: 0, total: 0, message: r.error ?? "Could not start the download" });
  } catch (e) {
    st.setLayaStatus({ state: "error", done: 0, total: 0, message: (e as Error).message });
  }
}

export const cancelLayaDownload = () => call("sysone:laya:cancel").catch(() => {});

export const stopSysOneTrain = () => call("sysone:train:stop").catch(() => {});

export async function inferSysOne(state: string, questions: SysOneQuestion[]): Promise<void> {
  const st = useStudio.getState();
  st.setSysoneInfer({ running: true, error: null });
  try {
    const res = await call<{ ok: boolean; answers?: SysOneAnswer[]; error?: string }>("sysone:infer", { state, questions });
    if (res.ok) st.setSysoneInfer({ running: false, answers: res.answers ?? [], error: null });
    else st.setSysoneInfer({ running: false, error: res.error ?? "Inference failed." });
  } catch (e) {
    st.setSysoneInfer({ running: false, error: (e as Error).message });
  }
}

/** The tools of every enabled MCP server (qualified names), as the backend wants them. */
export function enabledToolDefs() {
  const st = useStudio.getState();
  const defs: { name: string; description: string; inputSchema: Record<string, unknown> }[] = [];
  for (const srv of st.mcpServers) {
    if (!srv.enabled) continue;
    for (const t of st.mcpTools[srv.id] ?? []) defs.push({ name: qualifiedToolName(srv, t), description: t.description ?? "", inputSchema: t.inputSchema });
  }
  return defs;
}

/** Runs the trained System One model on a request: picks a tool and fills in its arguments. */
export async function callToolWithModel(request: string): Promise<{ ok: true; result: ToolCallResult } | { ok: false; error: string }> {
  const tools = enabledToolDefs();
  if (tools.length === 0) return { ok: false, error: "No tools to choose from — connect an MCP server and enable it first." };
  try {
    const res = await call<{ ok: true; result: ToolCallResult } | { ok: false; error: string }>("sysone:toolcall", { state: request, tools }, 60_000);
    return res;
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

// ---- System One: import a real classification dataset from Hugging Face ------

export async function fetchSysOneHfInfo(dataset: string): Promise<{ ok: true; info: HfDatasetInfo } | { ok: false; error: string }> {
  try {
    const res = await call<{ ok: boolean; info?: HfDatasetInfo; error?: string }>("sysone:hf:info", { dataset }, 30_000);
    if (!res.ok || !res.info) return { ok: false, error: res.error ?? "Could not look up that dataset." };
    return { ok: true, info: res.info };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

export interface SysOneHfImportParams {
  dataset: string;
  config: string;
  split: string;
  textCol: string;
  labelCol: string;
  optionsPerQ: number;
  total: number;
  questionText?: string;
}

export async function importSysOneHf(p: SysOneHfImportParams): Promise<string | null> {
  const st = useStudio.getState();
  st.setSysoneGenProgress({ state: "running", done: 0, total: 2, samples: 0, failed: 0, message: `Looking up ${p.dataset}…` });
  try {
    const res = await call<{ ok: boolean; error?: string }>("sysone:hf:import", p);
    if (!res.ok) {
      st.setSysoneGenProgress(null);
      return res.error ?? "Could not start the import.";
    }
    return null;
  } catch (e) {
    st.setSysoneGenProgress(null);
    return (e as Error).message;
  }
}

// ---- Tokenizer: built-in BPE or an imported tokenizer.json ----------------------

/** Uploads a Hugging Face tokenizer.json; the backend rebuilds the dataset with it. Returns an error message, or null. */
export async function importTokenizer(file: File): Promise<string | null> {
  const sid = getSocket().id;
  if (!sid) return "Not connected to the backend.";
  try {
    const res = await fetch(`${backendUrl()}/api/tokenizer?sid=${encodeURIComponent(sid)}&name=${encodeURIComponent(file.name)}`, { method: "PUT", body: file });
    const j = await res.json().catch(() => ({}));
    return res.ok && j.ok ? null : j.error || `Import failed (HTTP ${res.status}).`;
  } catch {
    return "Could not reach the backend.";
  }
}

/** Goes back to the built-in BPE (learned from the data). */
export const resetTokenizer = () => call("tokenizer:default").catch(() => {});

// ---- Hugging Face-style dataset export / import ------------------------------
export type DatasetKind = "qa" | "summary" | "sysone";
export type ExportStyle = "hf" | "simple" | "instruction" | "chat";
export type ExportFormat = "jsonl" | "parquet";
export type ImportStyle = ExportStyle | "auto";

/** Downloads the dataset as JSONL or Parquet in the chosen style (built by the backend). */
export async function exportDataset(kind: DatasetKind, style: ExportStyle, format: ExportFormat, system: string): Promise<string | null> {
  const sid = getSocket().id;
  if (!sid) return "Not connected to the backend.";
  try {
    const q = new URLSearchParams({ sid, kind, style, format, system });
    const res = await fetch(`${backendUrl()}/api/export?${q}`);
    if (!res.ok) return (await res.json().catch(() => ({}))).error || `Export failed (HTTP ${res.status}).`;
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = `${kind}-${style}.${format}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    return null;
  } catch {
    return "Could not reach the backend.";
  }
}

/** Uploads a JSON / JSONL / Parquet file; the backend adds its records and pushes the refreshed list. */
export async function importDataset(kind: DatasetKind, style: ImportStyle, file: File): Promise<{ error: string } | { added: number; skipped: number; capped: boolean }> {
  const sid = getSocket().id;
  if (!sid) return { error: "Not connected to the backend." };
  try {
    const res = await fetch(`${backendUrl()}/api/import?sid=${encodeURIComponent(sid)}&kind=${kind}&style=${style}&name=${encodeURIComponent(file.name)}`, { method: "PUT", body: file });
    const j = await res.json().catch(() => ({}));
    if (!res.ok || !j.ok) return { error: j.error || `Import failed (HTTP ${res.status}).` };
    return { added: j.added, skipped: j.skipped, capped: !!j.capped };
  } catch {
    return { error: "Could not reach the backend." };
  }
}

/** Downloads the trained model as a Hugging Face-style folder (zip): config, safetensors weights, tokenizer, README. */
export async function downloadModel(): Promise<string | null> {
  const sid = getSocket().id;
  if (!sid) return "Not connected to the backend.";
  try {
    const res = await fetch(`${backendUrl()}/api/model/export?sid=${encodeURIComponent(sid)}`);
    if (!res.ok) return (await res.json().catch(() => ({}))).error || `Download failed (HTTP ${res.status}).`;
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = "modelbuilder-model.zip";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    return null;
  } catch {
    return "Could not reach the backend.";
  }
}

export const stopTraining = () => call("train:stop").catch(() => {});

export async function generateText(prompt: string, opts: { maxNew: number; temperature: number; topK: number }) {
  const st = useStudio.getState();
  st.setGen({ running: true, text: "", error: null });
  try {
    const res = await call<{ ok: boolean; error?: string }>("generate", { prompt, maxNew: opts.maxNew, temperature: opts.temperature, topK: opts.topK, seed: Math.floor(Math.random() * 1e6) });
    if (!res.ok) st.setGen({ error: res.error ?? "Generation failed" });
  } catch (e) {
    st.setGen({ error: (e as Error).message });
  } finally {
    st.setGen({ running: false });
  }
}

export const stopGenerate = () => call("generate:stop").catch(() => {});

// ---- Learn chat --------------------------------------------------------------

let chatIdSeq = 0;
const chatId = () => `c${Date.now().toString(36)}${chatIdSeq++}`;

export async function askLearn(question: string) {
  const q = question.trim();
  if (!q) return;
  const st = useStudio.getState();
  if (st.learnChatBusy) return;

  const idx = st.blocks.findIndex((b) => b.id === st.learnBlockId);
  const block = idx >= 0 ? { title: blockLabel(st.blocks, idx), summary: CATALOG[st.blocks[idx].type].summary(st.blocks[idx]), params: st.blocks[idx] } : undefined;
  // last few turns only — enough continuity without an ever-growing prompt
  const history = st.learnChat
    .filter((m) => (m.role === "user" && m.question) || (m.role === "assistant" && m.answer))
    .slice(-8)
    .map((m) => (m.role === "user" ? { role: "user" as const, content: m.question! } : { role: "assistant" as const, content: m.answer!.text }));

  const userMsg = { id: chatId(), role: "user" as const, question: q };
  const pendingId = chatId();
  st.pushLearnMsg(userMsg);
  st.pushLearnMsg({ id: pendingId, role: "assistant", pending: true });
  st.setLearnChatBusy(true);
  try {
    const res = await call<{ ok: boolean; answer?: LearnAnswer; error?: string }>("learn:ask", { llm: st.llm, block, history, question: q }, 130_000);
    if (res.ok && res.answer) st.updateLearnMsg(pendingId, { pending: false, answer: res.answer });
    else st.updateLearnMsg(pendingId, { pending: false, error: res.error ?? "The model didn't answer." });
  } catch (e) {
    st.updateLearnMsg(pendingId, { pending: false, error: (e as Error).message });
  } finally {
    st.setLearnChatBusy(false);
  }
}

/** Retries the question of an errored assistant turn in place, replacing it. */
export async function retryLearn(msgId: string) {
  const st = useStudio.getState();
  const i = st.learnChat.findIndex((m) => m.id === msgId);
  const question = i > 0 ? st.learnChat[i - 1]?.question : undefined;
  if (!question) return;
  st.updateLearnMsg(msgId, { pending: true, error: undefined });
  st.setLearnChatBusy(true);
  const idx = st.blocks.findIndex((b) => b.id === st.learnBlockId);
  const block = idx >= 0 ? { title: blockLabel(st.blocks, idx), summary: CATALOG[st.blocks[idx].type].summary(st.blocks[idx]), params: st.blocks[idx] } : undefined;
  const history = st.learnChat
    .slice(0, i - 1)
    .filter((m) => (m.role === "user" && m.question) || (m.role === "assistant" && m.answer))
    .slice(-8)
    .map((m) => (m.role === "user" ? { role: "user" as const, content: m.question! } : { role: "assistant" as const, content: m.answer!.text }));
  try {
    const res = await call<{ ok: boolean; answer?: LearnAnswer; error?: string }>("learn:ask", { llm: st.llm, block, history, question }, 130_000);
    if (res.ok && res.answer) st.updateLearnMsg(msgId, { pending: false, answer: res.answer });
    else st.updateLearnMsg(msgId, { pending: false, error: res.error ?? "The model didn't answer." });
  } catch (e) {
    st.updateLearnMsg(msgId, { pending: false, error: (e as Error).message });
  } finally {
    st.setLearnChatBusy(false);
  }
}

export async function testLlmConnection(cfg: LlmConfig): Promise<{ ok: boolean; error?: string; models?: string[] }> {
  try {
    return await call("learn:test", cfg, 20_000);
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

// ---- Tool-calling dataset: MCP servers ------------------------------------------

const toWire = (s: McpServer) => ({
  name: s.name,
  transport: s.transport,
  command: s.command,
  args: s.args,
  env: s.env,
  cwd: s.cwd || undefined,
  url: s.url,
  timeoutSecs: s.timeoutSecs ?? undefined,
  connectTimeoutSecs: s.connectTimeoutSecs ?? undefined,
});

/** Connects to an MCP server, lists its tools and stores them. `test` only changes how the result is worded. */
export async function fetchMcpTools(id: string, mode: "refresh" | "test" = "refresh") {
  const st = useStudio.getState();
  const server = st.mcpServers.find((x) => x.id === id);
  if (!server || st.mcpStatus[id]?.busy) return;
  st.setMcpStatus(id, { busy: mode, ok: null, message: mode === "test" ? "Connecting…" : "Refreshing tools…" });
  // the backend enforces the same limits; give its own timeouts room to report first
  const budgetMs = ((server.connectTimeoutSecs ?? 20) + (server.timeoutSecs ?? 30) + 15) * 1000;
  try {
    const res = await call<{ ok: true; tools: McpTool[]; serverName: string | null; serverVersion: string | null; elapsedMs: number } | { ok: false; error: string }>("mcp:tools", toWire(server), budgetMs);
    if (res.ok) {
      useStudio.getState().setMcpTools(id, res.tools);
      const n = res.tools.length;
      useStudio.getState().setMcpStatus(id, {
        busy: null,
        ok: true,
        message: `${n} tool${n === 1 ? "" : "s"} · ${res.elapsedMs} ms`,
        serverName: res.serverName,
        serverVersion: res.serverVersion,
        elapsedMs: res.elapsedMs,
      });
    } else {
      useStudio.getState().setMcpStatus(id, { busy: null, ok: false, message: res.error });
    }
  } catch {
    // no ack at all: the backend is gone, or is an older build without MCP support
    useStudio.getState().setMcpStatus(id, { busy: null, ok: false, message: "The backend didn't answer. If you just updated the code, restart the backend (./dev.sh) and press Refresh tools." });
  }
}

/** On (re)connect: read the tools of every server that has none yet, one after another. */
export async function loadMissingMcpTools() {
  const st = useStudio.getState();
  for (const srv of st.mcpServers) {
    if (!useStudio.getState().mcpTools[srv.id]) await fetchMcpTools(srv.id, "refresh");
  }
}
