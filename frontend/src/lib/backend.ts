import { io, type Socket } from "socket.io-client";

/** NEXT_PUBLIC_BACKEND_URL wins; otherwise the same host the page was served from, port 4000. */
export function backendUrl(): string {
  const env = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (env) return env.replace(/\/$/, "");
  if (typeof window === "undefined") return "http://localhost:4000";
  return `${window.location.protocol}//${window.location.hostname}:4000`;
}

let socket: Socket | null = null;

export function getSocket(): Socket {
  if (!socket) {
    socket = io(backendUrl(), { transports: ["websocket"], reconnectionDelayMax: 3000 });
  }
  return socket;
}

/** Emit with an ack; rejects on timeout so the UI never waits forever. */
export function call<T = unknown>(event: string, payload?: unknown, timeoutMs = 60_000): Promise<T> {
  return new Promise((resolve, reject) => {
    const s = getSocket();
    const cb = (err: Error | null, res: T) => (err ? reject(new Error(`${event}: no reply from the backend (${timeoutMs / 1000}s)`)) : resolve(res));
    if (payload === undefined) s.timeout(timeoutMs).emit(event, cb);
    else s.timeout(timeoutMs).emit(event, payload, cb);
  });
}
