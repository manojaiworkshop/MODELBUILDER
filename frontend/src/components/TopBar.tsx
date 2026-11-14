"use client";
import { Download, Layers3, Moon, Play, Square, Sun } from "lucide-react";
import { useState } from "react";
import { HfModelModal } from "./build/HfModelModal";
import { PRESETS } from "@/lib/presets";
import { isRunning, useStudio } from "@/lib/store";
import { startSysOneTrain, startTraining, stopSysOneTrain, stopTraining } from "@/lib/sync";
import { backendUrl } from "@/lib/backend";
import { Button, cn, useTheme } from "./ui/kit";

const JEV_PRESET_ID = "jev-sysone";

function ConnPill() {
  const conn = useStudio((s) => s.conn);
  const map = {
    connected: ["bg-ok", "Connected"],
    connecting: ["bg-warn", "Connecting…"],
    disconnected: ["bg-danger", "Backend offline"],
  } as const;
  const [dot, label] = map[conn];
  return (
    <span className="hidden items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs text-muted sm:inline-flex" title={backendUrl()}>
      <span className={cn("h-2 w-2 rounded-full", dot, conn === "connecting" && "animate-pulse")} />
      {label}
    </span>
  );
}

export function TopBar() {
  const [theme, toggle] = useTheme();
  const setBlocks = useStudio((s) => s.setBlocks);
  const status = useStudio((s) => s.status);
  const conn = useStudio((s) => s.conn);
  const blocked = useStudio((s) => !!s.inspect && !s.inspect.ok);
  const modelKind = useStudio((s) => s.modelKind);
  const setModelKind = useStudio((s) => s.setModelKind);
  const sysoneStatus = useStudio((s) => s.sysoneStatus);
  const nSysoneSamples = useStudio((s) => s.sysoneSamples.length);
  const sysone = modelKind === "sysone";
  const [hfOpen, setHfOpen] = useState(false);
  const running = sysone ? isRunning(sysoneStatus) : isRunning(status);

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border bg-surface px-3 sm:px-4">
      <div className="flex items-center gap-2.5">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-accent-fg"><Layers3 size={18} /></span>
        <div className="leading-tight">
          <div className="text-sm font-semibold">ModelBuilder</div>
          <div className="hidden text-[11px] text-muted sm:block">build · train · observe a transformer</div>
        </div>
      </div>

      <div className="mx-1 hidden h-6 w-px bg-border md:block" />
      <label className="hidden items-center gap-2 text-xs text-muted md:flex">
        Preset
        <select
          aria-label="Load a preset architecture"
          disabled={running}
          value=""
          onChange={(e) => {
            if (e.target.value === JEV_PRESET_ID) {
              setModelKind("sysone");
              return;
            }
            const p = PRESETS.find((x) => x.id === e.target.value);
            if (p) {
              setModelKind("causal");
              setBlocks(p.build());
            }
          }}
          className="h-8 w-44 rounded-md border border-border bg-surface px-2 text-sm text-fg outline-none focus:border-accent disabled:opacity-50"
        >
          <option value="" disabled>Load…</option>
          {PRESETS.map((p) => <option key={p.id} value={p.id} title={p.blurb}>{p.name}</option>)}
          <option value={JEV_PRESET_ID} title="System One: a bidirectional decision model — state + typed questions in, calibrated probabilities out. Not a block stack.">
            JEV model (System One)
          </option>
        </select>
      </label>
      <Button size="sm" disabled={running || sysone} onClick={() => setHfOpen(true)} title="Build the block stack from a Hugging Face model URL or an uploaded model folder (reads config.json)">
        <Download size={13} /> <span className="hidden lg:inline">From Hugging Face</span>
      </Button>
      <HfModelModal open={hfOpen} onClose={() => setHfOpen(false)} />

      <div className="flex-1" />
      <ConnPill />
      {running ? (
        <Button variant="danger" onClick={() => (sysone ? stopSysOneTrain() : stopTraining())}><Square size={14} fill="currentColor" /> Stop</Button>
      ) : sysone ? (
        <Button variant="primary" disabled={conn !== "connected" || nSysoneSamples === 0} onClick={() => startSysOneTrain()} title={nSysoneSamples === 0 ? "Generate System One samples in the Data tab first" : "Train the JEV model on your generated samples"}>
          <Play size={14} fill="currentColor" /> Run
        </Button>
      ) : (
        <Button variant="primary" disabled={conn !== "connected" || blocked} onClick={() => startTraining()} title={blocked ? "Fix the model errors first" : "Train the model on your text"}>
          <Play size={14} fill="currentColor" /> Run
        </Button>
      )}
      <button onClick={toggle} aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} className="rounded-lg p-2 text-muted hover:bg-surface-2 hover:text-fg">
        {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
      </button>
    </header>
  );
}
