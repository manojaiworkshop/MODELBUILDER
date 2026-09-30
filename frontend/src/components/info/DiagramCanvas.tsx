"use client";
import { useEffect, useRef } from "react";
import type { DiagramEdge, DiagramNode, DiagramSpec } from "@/lib/types";
import { useTheme } from "../ui/kit";

const BOX_W = 148;
const BOX_H = 56;
const GAP_X = 72;
const GAP_Y = 22;
const PAD = 28;

interface Placed {
  node: DiagramNode;
  x: number;
  y: number;
}

/** Assigns each node a left-to-right column. Explicit `level`s are honored only when every node
 * has one; otherwise every node's column is its longest path from a root (in-degree-0 node),
 * which is cycle-safe and gives a sensible flow even for a graph the model didn't lay out itself. */
function computeLevels(nodes: DiagramNode[], edges: DiagramEdge[]): Map<string, number> {
  if (nodes.length > 0 && nodes.every((n) => n.level !== null && n.level !== undefined)) {
    return new Map(nodes.map((n) => [n.id, n.level as number]));
  }
  const ids = new Set(nodes.map((n) => n.id));
  const level = new Map<string, number>();
  const visiting = new Set<string>();
  const incoming = new Map<string, DiagramEdge[]>();
  for (const e of edges) {
    if (!ids.has(e.from) || !ids.has(e.to)) continue;
    (incoming.get(e.to) ?? incoming.set(e.to, []).get(e.to)!).push(e);
  }
  const dfs = (id: string): number => {
    const cached = level.get(id);
    if (cached !== undefined) return cached;
    if (visiting.has(id)) return 0; // cycle guard: break the loop here rather than recurse forever
    visiting.add(id);
    let maxPred = -1;
    for (const e of incoming.get(id) ?? []) maxPred = Math.max(maxPred, dfs(e.from));
    visiting.delete(id);
    const lv = maxPred + 1;
    level.set(id, lv);
    return lv;
  };
  for (const n of nodes) dfs(n.id);
  return level;
}

function layout(spec: DiagramSpec): { placed: Map<string, Placed>; width: number; height: number } {
  const levels = computeLevels(spec.nodes, spec.edges);
  const columns = new Map<number, DiagramNode[]>();
  for (const n of spec.nodes) {
    const lv = levels.get(n.id) ?? 0;
    (columns.get(lv) ?? columns.set(lv, []).get(lv)!).push(n);
  }
  const numLevels = Math.max(1, ...Array.from(columns.keys(), (k) => k + 1));
  const colHeight = (col: DiagramNode[]) => col.length * BOX_H + Math.max(0, col.length - 1) * GAP_Y;
  const maxColHeight = Math.max(BOX_H, ...Array.from(columns.values()).map(colHeight));

  const placed = new Map<string, Placed>();
  for (const [lv, col] of columns) {
    const h = colHeight(col);
    let y = PAD + (maxColHeight - h) / 2;
    const x = PAD + lv * (BOX_W + GAP_X);
    for (const node of col) {
      placed.set(node.id, { node, x, y });
      y += BOX_H + GAP_Y;
    }
  }
  return { placed, width: PAD * 2 + numLevels * BOX_W + (numLevels - 1) * GAP_X, height: PAD * 2 + maxColHeight };
}

/** Greedy word-wrap with no line cap, then — only if that overflows `maxLines` — collapses
 * everything from the last visible line onward into one line and truncates it to fit, character
 * by character, so the final line stays as full as possible instead of stopping at one word. */
function wrapLines(ctx: CanvasRenderingContext2D, text: string, maxWidth: number, maxLines: number): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const all: string[] = [];
  let cur = "";
  for (const w of words) {
    const next = cur ? `${cur} ${w}` : w;
    if (cur && ctx.measureText(next).width > maxWidth) {
      all.push(cur);
      cur = w;
    } else {
      cur = next;
    }
  }
  if (cur) all.push(cur);
  if (all.length <= maxLines) return all;

  const shown = all.slice(0, maxLines);
  let last = all.slice(maxLines - 1).join(" ");
  while (ctx.measureText(`${last}…`).width > maxWidth && last.length > 1) last = last.slice(0, -1);
  shown[maxLines - 1] = `${last}…`;
  return shown;
}

function colors(el: HTMLElement) {
  const cs = getComputedStyle(el);
  const g = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    boxFill: g("--surface-2", "#eff1f5"),
    boxBorder: g("--border-strong", "#cbd1db"),
    fg: g("--fg", "#0f172a"),
    muted: g("--muted", "#64748b"),
    accent: g("--accent", "#4f46e5"),
    surface: g("--surface", "#fff"),
  };
}

function drawArrow(ctx: CanvasRenderingContext2D, from: [number, number], to: [number, number], ctrl: [number, number][], stroke: string, label: string | null, fg: string, pillBg: string) {
  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.globalAlpha = 0.65;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(...from);
  if (ctrl.length === 2) ctx.bezierCurveTo(ctrl[0][0], ctrl[0][1], ctrl[1][0], ctrl[1][1], to[0], to[1]);
  else ctx.lineTo(to[0], to[1]);
  ctx.stroke();

  // arrowhead, oriented along the curve's final tangent
  const prev = ctrl.length === 2 ? ctrl[1] : from;
  const angle = Math.atan2(to[1] - prev[1], to[0] - prev[0]);
  const size = 7;
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = stroke;
  ctx.beginPath();
  ctx.moveTo(to[0], to[1]);
  ctx.lineTo(to[0] - size * Math.cos(angle - Math.PI / 6), to[1] - size * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(to[0] - size * Math.cos(angle + Math.PI / 6), to[1] - size * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  if (label) {
    const mid: [number, number] = ctrl.length === 2
      ? bezierPoint(from, ctrl[0], ctrl[1], to, 0.5)
      : [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2];
    ctx.save();
    ctx.font = "10px var(--font-mono, monospace)";
    const w = ctx.measureText(label).width + 10;
    ctx.fillStyle = pillBg;
    roundRect(ctx, mid[0] - w / 2, mid[1] - 9, w, 18, 9);
    ctx.fill();
    ctx.fillStyle = fg;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, mid[0], mid[1] + 0.5);
    ctx.restore();
  }
}

function bezierPoint(p0: [number, number], p1: [number, number], p2: [number, number], p3: [number, number], t: number): [number, number] {
  const mt = 1 - t;
  const x = mt * mt * mt * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t * t * t * p3[0];
  const y = mt * mt * mt * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t * t * t * p3[1];
  return [x, y];
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Small, labeled node/edge diagram — the model's own visual aid, drawn on canvas so it stays
 * crisp and cheap to redraw on theme/resize changes without a charting library. */
export function DiagramCanvas({ spec }: { spec: DiagramSpec }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [theme] = useTheme();
  const { placed, width, height } = layout(spec);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const c = colors(canvas);

    // edges first, so boxes sit on top of the lines that meet their borders
    for (const e of spec.edges) {
      const a = placed.get(e.from);
      const b = placed.get(e.to);
      if (!a || !b) continue;
      const forward = b.x > a.x;
      const from: [number, number] = forward ? [a.x + BOX_W, a.y + BOX_H / 2] : [a.x, a.y + BOX_H / 2];
      const to: [number, number] = forward ? [b.x, b.y + BOX_H / 2] : [b.x + BOX_W, b.y + BOX_H / 2];
      const dx = Math.max(40, Math.abs(to[0] - from[0]) / 2);
      const bend = forward ? 0 : (a.y <= b.y ? -46 : 46); // bow same-level/back edges clear of the boxes
      const ctrl: [number, number][] = forward
        ? [[from[0] + dx, from[1]], [to[0] - dx, to[1]]]
        : [[from[0] + dx * Math.sign(to[0] - from[0] || 1), from[1] + bend], [to[0] - dx * Math.sign(to[0] - from[0] || -1), to[1] + bend]];
      drawArrow(ctx, from, to, ctrl, c.accent, e.label, c.muted, c.surface);
    }

    for (const { node, x, y } of placed.values()) {
      ctx.fillStyle = c.boxFill;
      ctx.strokeStyle = c.boxBorder;
      ctx.lineWidth = 1;
      roundRect(ctx, x, y, BOX_W, BOX_H, 10);
      ctx.fill();
      ctx.stroke();

      ctx.textAlign = "center";
      ctx.fillStyle = c.fg;
      ctx.font = "600 12px var(--font-sans, sans-serif)";
      const lines = wrapLines(ctx, node.label || node.id, BOX_W - 20, node.sublabel ? 2 : 3);
      const startY = y + BOX_H / 2 - ((lines.length - 1) * 14) / 2 - (node.sublabel ? 7 : 0);
      lines.forEach((line, i) => ctx.fillText(line, x + BOX_W / 2, startY + i * 14));

      if (node.sublabel) {
        ctx.fillStyle = c.muted;
        ctx.font = "10px var(--font-mono, monospace)";
        const subY = startY + lines.length * 14 + 4;
        const [sub] = wrapLines(ctx, node.sublabel, BOX_W - 20, 1);
        ctx.fillText(sub, x + BOX_W / 2, subY);
      }
    }
  }, [placed, width, height, theme, spec]);

  return (
    <div ref={wrapRef} className="overflow-x-auto rounded-lg border border-border bg-surface-2/40 p-1">
      {spec.title && <div className="px-2 pt-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted">{spec.title}</div>}
      <canvas ref={canvasRef} role="img" aria-label={spec.title ?? "Concept diagram"} />
    </div>
  );
}
