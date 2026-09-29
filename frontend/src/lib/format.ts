export function fmtBytes(n: number): string {
  if (n >= 2 ** 30) return `${(n / 2 ** 30).toFixed(2)} GB`;
  if (n >= 2 ** 20) return `${(n / 2 ** 20).toFixed(n >= 100 * 2 ** 20 ? 0 : 1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(n >= 100 * 1024 ? 0 : 1)} KB`;
  return `${n} B`;
}

export const fmtCount = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n.toLocaleString());

export function fmtSeconds(ms: number): string {
  return ms < 1000 ? `${ms} ms` : ms < 60_000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}
