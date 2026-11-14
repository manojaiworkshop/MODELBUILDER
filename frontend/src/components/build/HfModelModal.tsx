"use client";
import { AlertTriangle, FolderUp, Info, Loader2, Search } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { buildBlocks, defaultFit, fetchHfConfig, MAX_D_MODEL, MAX_LAYERS, parseHfConfig, readConfigFromFiles, type Arch, type FitOptions } from "@/lib/hfconfig";
import { useStudio } from "@/lib/store";
import { Button, Field, Modal, NumberInput } from "../ui/kit";

const EXAMPLES = ["google/gemma-3-270m", "Qwen/Qwen3-0.6B", "openai-community/gpt2", "mistralai/Mixtral-8x7B-v0.1"];

/** Builds the block stack from a Hugging Face model: a pasted URL / repo id, or an uploaded model
 * folder (only its config.json is read — no weights are downloaded or uploaded). */
export function HfModelModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const setBlocks = useStudio((s) => s.setBlocks);
  const setModelKind = useStudio((s) => s.setModelKind);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [arch, setArch] = useState<Arch | null>(null);
  const [fit, setFit] = useState<FitOptions>({ dModel: 128, layers: 4, maxExperts: 4, maxWindow: 32 });
  const folderRef = useRef<HTMLInputElement>(null);

  function accept(cfg: unknown, id: string) {
    try {
      const a = parseHfConfig(cfg, id);
      setArch(a);
      setFit(defaultFit(a));
      setError("");
    } catch (e) {
      setArch(null);
      setError((e as Error).message);
    }
  }
  async function run(p: Promise<{ cfg: unknown; id: string }>) {
    setBusy(true);
    setError("");
    try {
      const { cfg, id } = await p;
      accept(cfg, id);
    } catch (e) {
      setArch(null);
      setError((e as Error).message);
    }
    setBusy(false);
  }

  const built = useMemo(() => (arch ? buildBlocks(arch, fit) : null), [arch, fit]);

  function apply() {
    if (!built) return;
    setModelKind("causal");
    setBlocks(built.blocks);
    onClose();
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Build from a Hugging Face model"
      sub="Reads the model's config.json and assembles the matching block stack — from a URL, or from a model folder on your computer."
      icon={<FolderUp size={16} className="text-accent" />}
      width="max-w-xl"
    >
      <div className="space-y-4">
        <Field label="Model URL or repo id" help='e.g. "google/gemma-3-270m" or https://huggingface.co/Qwen/Qwen3-0.6B'>
          <div className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && input.trim() && run(fetchHfConfig(input))}
              placeholder="org/model-name"
              spellCheck={false}
              className="h-9 min-w-0 flex-1 rounded-lg border border-border bg-surface px-3 font-mono text-sm outline-none focus:border-accent"
            />
            <Button size="sm" disabled={!input.trim() || busy} onClick={() => run(fetchHfConfig(input))}>
              {busy ? <Loader2 size={13} className="animate-spin" /> : <Search size={13} />} Fetch
            </Button>
          </div>
          <p className="mt-1 flex flex-wrap gap-x-1.5 text-[11px] text-muted">
            Try: {EXAMPLES.map((d, i) => (
              <span key={d}>
                <button type="button" onClick={() => setInput(d)} className="underline decoration-dotted hover:text-accent">{d}</button>
                {i < EXAMPLES.length - 1 ? "," : ""}
              </span>
            ))}
          </p>
        </Field>

        <div className="flex items-center gap-3 text-xs text-muted">
          <span className="h-px flex-1 bg-border" /> or <span className="h-px flex-1 bg-border" />
        </div>

        <div>
          <input
            ref={folderRef}
            type="file"
            multiple
            className="hidden"
            {...({ webkitdirectory: "" } as Record<string, string>)}
            onChange={(e) => {
              if (e.target.files?.length) run(readConfigFromFiles(e.target.files));
              e.target.value = "";
            }}
          />
          <input
            id="hf-config-file"
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) run(readConfigFromFiles(e.target.files));
              e.target.value = "";
            }}
          />
          <div className="flex flex-wrap gap-2">
            <Button size="sm" disabled={busy} onClick={() => folderRef.current?.click()}><FolderUp size={13} /> Upload model folder</Button>
            <Button size="sm" disabled={busy} onClick={() => document.getElementById("hf-config-file")?.click()}>Choose config.json</Button>
          </div>
          <p className="mt-1 text-[11px] text-muted">Only config.json is read (in your browser); the weights in the folder are not uploaded.</p>
        </div>

        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-danger/40 bg-danger/5 px-3 py-2 text-xs text-danger">
            <AlertTriangle size={14} className="mt-px shrink-0" /> {error}
          </div>
        )}

        {arch && built && (
          <div className="space-y-3 rounded-lg border border-border bg-surface-2/40 p-3">
            {arch.nativeBlocks ? (
              <p className="text-xs">ModelBuilder export detected: <b>{arch.nativeBlocks.length}</b> blocks.</p>
            ) : (
              <>
                <div className="text-xs">
                  <div className="font-medium">{arch.name} <span className="font-normal text-muted">· {arch.modelType}</span></div>
                  <div className="mt-0.5 text-muted">
                    {arch.layers} layers · width {arch.hidden} · {arch.heads} heads / {arch.kvHeads} KV · {arch.ffn === "moe" ? `MoE ${arch.topK}/${arch.experts}` : arch.ffn.toUpperCase()} ff {arch.dff} · {arch.norm}Norm · {arch.rope ? `RoPE θ=${arch.ropeTheta}` : "learned positions"}
                    {arch.windows.some((w) => w) && ` · sliding window ${Math.max(...arch.windows)}`}{arch.qkNorm && " · QK-norm"}{arch.sink && " · attention sink"}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Field label="d_model" help={`Width of the rebuilt model (≤ ${MAX_D_MODEL}).`}>
                    <NumberInput value={fit.dModel} min={16} max={Math.min(MAX_D_MODEL, arch.hidden)} step={16} onChange={(v) => setFit({ ...fit, dModel: v })} />
                  </Field>
                  <Field label="Layers" help={`Attention + feed-forward pairs (≤ ${MAX_LAYERS}).`}>
                    <NumberInput value={fit.layers} min={1} max={Math.min(MAX_LAYERS, arch.layers)} onChange={(v) => setFit({ ...fit, layers: v })} />
                  </Field>
                  {arch.ffn === "moe" && (
                    <Field label="Max experts">
                      <NumberInput value={fit.maxExperts} min={2} max={16} onChange={(v) => setFit({ ...fit, maxExperts: v })} />
                    </Field>
                  )}
                  {arch.windows.some((w) => w) && (
                    <Field label="Max window">
                      <NumberInput value={fit.maxWindow} min={4} max={256} step={4} onChange={(v) => setFit({ ...fit, maxWindow: v })} />
                    </Field>
                  )}
                </div>
              </>
            )}
            <ul className="space-y-1 text-[11px] text-muted">
              {built.notes.map((n, i) => (
                <li key={i} className="flex gap-1.5"><Info size={12} className="mt-px shrink-0" />{n}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={!built} onClick={apply}>Build model ({built ? built.blocks.length : 0} blocks)</Button>
        </div>
      </div>
    </Modal>
  );
}
