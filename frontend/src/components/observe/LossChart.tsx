"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useStudio } from "@/lib/store";

const H = 210;
const M = { l: 44, r: 12, t: 12, b: 26 };

function bucket(values: (number | undefined)[], maxPts: number): { x: number; y: number }[] {
  const n = values.length;
  const size = Math.max(1, Math.ceil(n / maxPts));
  const out: { x: number; y: number }[] = [];
  for (let i = 0; i < n; i += size) {
    let sum = 0, cnt = 0;
    for (let j = i; j < Math.min(n, i + size); j++) {
      const v = values[j];
      if (v !== undefined && Number.isFinite(v)) { sum += v; cnt++; }
    }
    if (cnt) out.push({ x: Math.min(n, i + size) , y: sum / cnt });
  }
  return out;
}

function niceMax(v: number) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

export function LossChart() {
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(520);
  const [hover, setHover] = useState<number | null>(null);
  const rev = useStudio((s) => s.series.rev);
  const series = useStudio((s) => s.series);
  const total = useStudio((s) => s.status.total || s.hyper.steps);
  const vocab = useStudio((s) => s.inspect?.vocabSize);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(260, el.clientWidth)));
    ro.observe(el);
    setW(Math.max(260, el.clientWidth));
    return () => ro.disconnect();
  }, []);

  const { raw, smooth, val, yMax, xMax } = useMemo(() => {
    void rev;
    const loss = series.loss;
    const raw = bucket(loss, 500);
    const smooth = raw.reduce<{ x: number; y: number }[]>((acc, p) => {
      const prev = acc.at(-1);
      acc.push({ x: p.x, y: prev ? prev.y * 0.85 + p.y * 0.15 : p.y });
      return acc;
    }, []);
    const val = series.evals.map((e) => ({ x: e.step, y: e.valLoss }));
    const peak = Math.max(0.1, ...raw.map((p) => p.y), ...val.map((p) => p.y));
    const yMax = niceMax(peak * 1.05);
    return { raw, smooth, val, yMax, xMax: Math.max(total, loss.length, 1) };
  }, [series, rev, total]);

  const iw = w - M.l - M.r;
  const ih = H - M.t - M.b;
  const X = (x: number) => M.l + (x / xMax) * iw;
  const Y = (y: number) => M.t + ih - (Math.min(y, yMax) / yMax) * ih;
  const path = (pts: { x: number; y: number }[]) => pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
  const ticksY = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);
  const ticksX = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * xMax));
  const guess = vocab ? Math.log(vocab) : null;

  const hoverStep = hover === null ? null : Math.max(1, Math.min(series.loss.length, Math.round(((hover - M.l) / iw) * xMax)));
  const hoverLoss = hoverStep ? series.loss[hoverStep - 1] : undefined;
  const hoverVal = hoverStep ? series.evals.findLast((e) => e.step <= hoverStep) : undefined;
  const empty = series.loss.length === 0;

  return (
    <div ref={wrap} className="relative w-full">
      <svg width={w} height={H} role="img" aria-label="Training and validation loss over steps" onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); setHover(e.clientX - r.left); }} onMouseLeave={() => setHover(null)}>
        {ticksY.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={w - M.r} y1={Y(t)} y2={Y(t)} stroke="var(--border)" strokeDasharray={t === 0 ? undefined : "3 4"} />
            <text x={M.l - 6} y={Y(t) + 3.5} textAnchor="end" fontSize="10" fill="var(--muted)">{t.toFixed(t < 1 ? 2 : 1)}</text>
          </g>
        ))}
        {ticksX.map((t) => (
          <text key={t} x={X(t)} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--muted)">{t}</text>
        ))}
        {guess !== null && guess < yMax && (
          <g>
            <line x1={M.l} x2={w - M.r} y1={Y(guess)} y2={Y(guess)} stroke="var(--warn)" strokeDasharray="2 4" opacity="0.7" />
            <text x={w - M.r - 4} y={Y(guess) - 4} textAnchor="end" fontSize="10" fill="var(--warn)">random guess ln({vocab}) = {guess.toFixed(2)}</text>
          </g>
        )}
        {!empty && (
          <>
            <path d={path(raw)} fill="none" stroke="var(--accent)" strokeOpacity="0.28" strokeWidth="1.2" />
            <path d={path(smooth)} fill="none" stroke="var(--accent)" strokeWidth="2.2" strokeLinejoin="round" />
            {val.length > 0 && <path d={path(val)} fill="none" stroke="var(--ok)" strokeWidth="2" strokeDasharray="5 3" />}
            {val.map((p, i) => <circle key={i} cx={X(p.x)} cy={Y(p.y)} r="3" fill="var(--ok)" />)}
            {smooth.length > 0 && <circle cx={X(smooth[smooth.length - 1].x)} cy={Y(smooth[smooth.length - 1].y)} r="4" fill="var(--accent)" className="live-dot" />}
          </>
        )}
        {hoverStep && hover !== null && !empty && <line x1={X(hoverStep)} x2={X(hoverStep)} y1={M.t} y2={M.t + ih} stroke="var(--fg)" strokeOpacity="0.25" />}
      </svg>
      {empty && <div className="absolute inset-0 flex items-center justify-center text-sm text-muted">Press Run to start training</div>}
      {hoverStep && hoverLoss !== undefined && (
        <div className="pointer-events-none absolute right-3 top-2 rounded-md border border-border bg-surface px-2.5 py-1.5 text-[11px] shadow-card">
          <div className="font-medium">step {hoverStep}</div>
          <div className="text-accent">train {hoverLoss.toFixed(3)}</div>
          {hoverVal && <div className="text-ok">val {hoverVal.valLoss.toFixed(3)} (step {hoverVal.step})</div>}
        </div>
      )}
      <div className="flex items-center gap-4 px-3 pb-2 text-[11px] text-muted">
        <span className="flex items-center gap-1.5"><i className="inline-block h-0.5 w-4 bg-accent" /> train (smoothed)</span>
        <span className="flex items-center gap-1.5"><i className="inline-block h-0.5 w-4 border-t-2 border-dashed border-ok" /> validation</span>
      </div>
    </div>
  );
}
