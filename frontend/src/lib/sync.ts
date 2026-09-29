"use client";
import { useEffect, useRef } from "react";
import { call, getSocket } from "./backend";
import { isRunning, useStudio } from "./store";
import type { BuildProgress, DataInfo, DocInfo, EvalPoint, InspectPayload, SamplePayload, StatusPayload, StepPoint } from "./types";

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
    const res = await call<{ ok: boolean; error?: string }>("train:start", { spec: { blocks: st.blocks }, hyper: st.hyper, probe: st.probe });
    if (!res.ok) st.setStatus({ state: "error", message: res.error ?? "Could not start training", step: 0, total: 0 });
  } catch (e) {
    st.setStatus({ state: "error", message: (e as Error).message, step: 0, total: 0 });
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
