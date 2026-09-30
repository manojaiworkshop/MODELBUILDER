"use client";
// Runs a user-edited "tiny piece" snippet for the Playground — entirely in the browser, never
// sent to the backend. Safety model:
//   - a fresh Web Worker per run (its own JS realm: no DOM, no cookies, no localStorage);
//   - network access is nulled out on that realm before any user code runs (fetch/XHR/importScripts/
//     WebSocket), so a snippet can't phone home even accidentally;
//   - a hard wall-clock timeout terminates a runaway/looping worker outright;
//   - user code runs through `new Function(...)`, which closes over the *global* scope of the
//     worker realm (not this module's), so nulling those globals actually restricts it — a local
//     `const fetch = undefined` here would not.
// The only capability exposed to the snippet is a small vector/matrix helper library (below) plus
// a `print` function, whose arguments are captured, formatted, and — when they look like a numeric
// vector/matrix — also kept as raw numbers so the UI can draw a small heatmap next to the text.

export interface LogEntry {
  text: string;
  /** Present when an argument to `print` was a flat numeric array or a rectangular 2D one. */
  data: { rows: number; cols: number; values: number[] } | null;
}

export interface SandboxResult {
  ok: boolean;
  entries: LogEntry[];
  error?: string;
  ms: number;
}

const TIMEOUT_MS = 2500;

// Kept inside the worker string, not this module, so it's the WORKER's `fetch` etc. that get
// nulled — see the file-level comment on why a closure here wouldn't help.
const WORKER_SRC = `
"use strict";
self.fetch = undefined;
self.XMLHttpRequest = undefined;
self.WebSocket = undefined;
self.importScripts = undefined;

function prettyNum(x) {
  if (x === Infinity) return "Infinity";
  if (x === -Infinity) return "-Infinity";
  if (Number.isNaN(x)) return "NaN";
  return String(Math.round(x * 1000) / 1000);
}
function prettyPrint(x) {
  if (typeof x === "number") return prettyNum(x);
  if (typeof x === "string") return x;
  if (Array.isArray(x)) return "[" + x.map(prettyPrint).join(", ") + "]";
  try { return JSON.stringify(x); } catch (e) { return String(x); }
}
function isNumArray(a) { return Array.isArray(a) && a.length > 0 && a.every(function (v) { return typeof v === "number"; }); }
function asData(args) {
  for (var i = args.length - 1; i >= 0; i--) {
    var a = args[i];
    if (isNumArray(a)) {
      if (!a.every(Number.isFinite)) return null;
      return { rows: 1, cols: a.length, values: a.slice() };
    }
    if (Array.isArray(a) && a.length > 0 && a.every(isNumArray)) {
      var cols = a[0].length;
      if (a.every(function (row) { return row.length === cols; }) && a.every(function (row) { return row.every(Number.isFinite); })) {
        return { rows: a.length, cols: cols, values: [].concat.apply([], a) };
      }
    }
  }
  return null;
}

// ---- tiny linear-algebra helpers made available to the snippet ----
function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function transpose(A) { return A[0].map(function (_, j) { return A.map(function (row) { return row[j]; }); }); }
function matmul(A, B) { var Bt = transpose(B); return A.map(function (row) { return Bt.map(function (col) { return dot(row, col); }); }); }
function matvec(A, v) { return A.map(function (row) { return dot(row, v); }); }
function softmax(v) {
  var finite = v.filter(Number.isFinite);
  var m = finite.length ? Math.max.apply(null, finite) : 0;
  var e = v.map(function (x) { return Number.isFinite(x) ? Math.exp(x - m) : 0; });
  var s = e.reduce(function (a, b) { return a + b; }, 0) || 1;
  return e.map(function (x) { return x / s; });
}
function rmsnorm(v, eps) {
  eps = eps === undefined ? 1e-6 : eps;
  var ms = v.reduce(function (s, x) { return s + x * x; }, 0) / v.length;
  var scale = 1 / Math.sqrt(ms + eps);
  return v.map(function (x) { return x * scale; });
}
function gelu(x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))); }
function silu(x) { return x / (1 + Math.exp(-x)); }
function relu(x) { return Math.max(0, x); }

self.onmessage = function (ev) {
  var entries = [];
  function print() {
    var args = Array.prototype.slice.call(arguments);
    entries.push({ text: args.map(prettyPrint).join(" "), data: asData(args) });
  }
  try {
    var fn = new Function("print", "dot", "matvec", "matmul", "transpose", "softmax", "rmsnorm", "gelu", "silu", "relu", ev.data.code);
    fn(print, dot, matvec, matmul, transpose, softmax, rmsnorm, gelu, silu, relu);
    self.postMessage({ ok: true, entries: entries });
  } catch (err) {
    self.postMessage({ ok: false, entries: entries, error: (err && err.message) || String(err) });
  }
};
`;

let workerUrl: string | null = null;
function getWorkerUrl(): string {
  if (!workerUrl) workerUrl = URL.createObjectURL(new Blob([WORKER_SRC], { type: "application/javascript" }));
  return workerUrl;
}

export function runInSandbox(code: string): Promise<SandboxResult> {
  return new Promise((resolve) => {
    const t0 = performance.now();
    let settled = false;
    const worker = new Worker(getWorkerUrl());
    const finish = (r: Omit<SandboxResult, "ms">) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      worker.terminate();
      resolve({ ...r, ms: Math.round(performance.now() - t0) });
    };
    const timer = setTimeout(() => finish({ ok: false, entries: [], error: `Timed out after ${TIMEOUT_MS / 1000}s — check for an infinite loop.` }), TIMEOUT_MS);
    worker.onmessage = (e) => finish(e.data as Omit<SandboxResult, "ms">);
    worker.onerror = (e) => finish({ ok: false, entries: [], error: e.message || "The sandbox crashed." });
    worker.postMessage({ code });
  });
}
