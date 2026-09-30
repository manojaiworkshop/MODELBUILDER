"use client";
import { useEffect, useRef } from "react";
import { call, getSocket } from "./backend";
import { isRunning, useStudio } from "./store";
import { blockLabel, CATALOG } from "./blocks";
import type { BuildProgress, DataInfo, DocInfo, EvalPoint, InspectPayload, LearnAnswer, LlmConfig, QaPageList, QaPair, QaProgress, SamplePayload, StatusPayload, StepPoint } from "./types";

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
      st().setDocs([]);
      st().setProgress(null);
      st().setQa([]);
      st().setQaProgress(null);
      if (st().trainSource === "qa") st().setTrainSource("text");
      if (st().trainMode === "continue") st().setTrainMode("new"); // the new session has no earlier weights
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
    const onDocs = (p: { docs: DocInfo[] }) => st().setDocs(p.docs);
    const onProgress = (p: BuildProgress) => st().setProgress(p.stage === "done" ? null : p);
    const onInfo = (p: DataInfo) => {
      st().setProgress(null);
      st().setDataInfo(p.ok ? p : null);
      if (!p.ok) {
        st().setInspect(null);
        st().setDataError(p.error);
      } else st().setDataError(null);
    };

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
      s.off("qa:list", onQaList);
      s.off("qa:added", onQaAdded);
      s.off("qa:progress", onQaProgress);
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
    // "continue" only makes sense for Q&A pairs and when there are weights to build on (the bar shows the same rule)
    const mode = st.trainSource === "qa" && st.trainMode === "continue" && st.inspect?.trained ? "continue" : "new";
    let probe = st.probe;
    if (st.trainSource === "qa") {
      // sample from a prompt shaped like the training text, so the samples show what it learned
      const q = st.qa[0]?.question;
      if (q && !probe.startsWith("Question:")) {
        probe = `Question: ${q}\nAnswer:`;
        st.setProbe(probe);
      }
    }
    const res = await call<{ ok: boolean; error?: string }>("train:start", { spec: { blocks: st.blocks }, hyper: st.hyper, probe, source: st.trainSource, mode });
    if (!res.ok) st.setStatus({ state: "error", message: res.error ?? "Could not start training", step: 0, total: 0 });
  } catch (e) {
    st.setStatus({ state: "error", message: (e as Error).message, step: 0, total: 0 });
  }
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

export async function generateQa(docId: string, pages: number[], perPage: number, concurrency: number): Promise<string | null> {
  const st = useStudio.getState();
  st.setQaProgress({ state: "running", done: 0, total: pages.length, pairs: 0, failed: 0, message: "" });
  try {
    const res = await call<{ ok: boolean; error?: string }>("qa:generate", { llm: st.llm, id: docId, pages, perPage, concurrency });
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
