"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "./kit";

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.trim().replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return Number.isFinite(n) ? [(n >> 16) & 255, (n >> 8) & 255, n & 255] : [99, 102, 241];
}

interface Props {
  data: number[];
  rows: number;
  cols: number;
  /** diverging: negative=blue, positive=red. sequential: 0..max in the accent colour. */
  mode?: "diverging" | "sequential";
  max?: number;
  /** CSS px per row. */
  cellH?: number;
  /** CSS px per column; omit to stretch across the container. */
  cellW?: number;
  rowLabels?: string[];
  colLabels?: string[];
  className?: string;
  /** Keep cells square (attention maps) instead of using cellH. */
  square?: boolean;
  /** Shown in the hover tooltip after the value. */
  unit?: string;
}

/** One pixel per cell, stretched with `image-rendering: pixelated`: crisp, cheap to redraw every probe. */
export function Heatmap({ data, rows, cols, mode = "diverging", max, cellH = 14, cellW, rowLabels, colLabels, className, unit, square }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [theme] = useTheme();
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const scale = useMemo(() => {
    if (max && max > 0) return max;
    let m = 1e-6;
    for (const v of data) m = Math.max(m, Math.abs(v));
    return mode === "sequential" ? Math.max(m, 1e-6) : m;
  }, [data, max, mode]);

  useEffect(() => {
    const c = ref.current;
    if (!c || !rows || !cols) return;
    c.width = cols;
    c.height = rows;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const cs = getComputedStyle(c);
    const acc = hexToRgb(cs.getPropertyValue("--accent"));
    const pos: [number, number, number] = theme === "dark" ? [251, 113, 133] : [225, 29, 72];
    const neg: [number, number, number] = theme === "dark" ? [96, 165, 250] : [37, 99, 235];
    const img = ctx.createImageData(cols, rows);
    for (let i = 0; i < rows * cols; i++) {
      const v = data[i] ?? 0;
      let rgb = acc;
      let a: number;
      if (mode === "sequential") {
        a = Math.pow(Math.min(1, Math.max(0, v / scale)), 0.75);
      } else {
        rgb = v >= 0 ? pos : neg;
        a = Math.min(1, Math.abs(v) / scale);
      }
      img.data[i * 4] = rgb[0];
      img.data[i * 4 + 1] = rgb[1];
      img.data[i * 4 + 2] = rgb[2];
      img.data[i * 4 + 3] = Math.round(a * 255);
    }
    ctx.putImageData(img, 0, 0);
  }, [data, rows, cols, mode, scale, theme]);

  return (
    <div className={`relative ${className ?? ""}`} style={{ width: cellW ? Math.min(cols * cellW, 4000) : "100%", maxWidth: "100%" }}>
      <canvas
        ref={ref}
        role="img"
        aria-label={`${rows} by ${cols} heatmap`}
        className="block w-full rounded-[3px] bg-surface-2/60"
        style={{ imageRendering: "pixelated", ...(square ? { aspectRatio: `${cols} / ${rows}` } : { height: rows * cellH }) }}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const col = Math.min(cols - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * cols)));
          const row = Math.min(rows - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * rows)));
          const v = data[row * cols + col];
          const rl = rowLabels?.[row] ?? `row ${row}`;
          const cl = colLabels?.[col] ?? `col ${col}`;
          setTip({ x: e.clientX - r.left, y: e.clientY - r.top, text: `${rl} × ${cl}: ${v?.toFixed(3)}${unit ?? ""}` });
        }}
        onMouseLeave={() => setTip(null)}
      />
      {tip && (
        <div
          className="pointer-events-none absolute z-20 whitespace-nowrap rounded-md border border-border bg-surface px-2 py-1 text-[11px] shadow-card"
          style={{ left: Math.min(tip.x + 10, 9999), top: tip.y + 14 }}
        >
          {tip.text}
        </div>
      )}
    </div>
  );
}
