"use client";
import { backendUrl, getSocket } from "./backend";
import { SAMPLES } from "./samples";
import { useStudio } from "./store";
import type { UploadItem } from "./types";

const MAX_PARALLEL = 2;
const xhrs = new Map<string, XMLHttpRequest>();
const files = new Map<string, File>();
let running = 0;

export const MAX_FILE_BYTES = 1024 ** 3; // mirrors the backend default; the server has the final say

let counter = 0;

/** Queue files for upload. Directories dropped/picked contribute their files (path kept in the name). */
export function uploadFiles(list: File[]) {
  const st = useStudio.getState();
  if (!list.length) return;
  // the built-in sample text would otherwise silently join the user's own data
  if (st.usePasted && SAMPLES.some((s) => s.text === st.text)) st.setUsePasted(false);

  const items: UploadItem[] = list.map((f) => {
    const id = `u${Date.now().toString(36)}${counter++}`;
    files.set(id, f);
    const name = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
    const tooBig = f.size > MAX_FILE_BYTES;
    return { id, name, size: f.size, loaded: 0, state: tooBig ? "error" : "queued", error: tooBig ? `Larger than the ${MAX_FILE_BYTES / 1024 ** 3} GB per-file limit.` : undefined };
  });
  st.addUploads(items);
  pump();
}

function pump() {
  const st = useStudio.getState();
  while (running < MAX_PARALLEL) {
    const next = useStudio.getState().uploads.find((u) => u.state === "queued");
    if (!next) return;
    start(next.id);
  }
  void st;
}

function start(id: string) {
  const st = useStudio.getState();
  const file = files.get(id);
  const item = st.uploads.find((u) => u.id === id);
  const sid = getSocket().id;
  if (!file || !item) return;
  if (!sid) {
    st.patchUpload(id, { state: "error", error: "Not connected to the backend." });
    return;
  }
  running++;
  st.patchUpload(id, { state: "uploading", loaded: 0 });

  const xhr = new XMLHttpRequest();
  xhrs.set(id, xhr);
  xhr.open("PUT", `${backendUrl()}/api/upload?sid=${encodeURIComponent(sid)}&name=${encodeURIComponent(item.name)}`);
  xhr.upload.onprogress = (e) => useStudio.getState().patchUpload(id, { loaded: e.loaded });
  const finish = (error?: string) => {
    running--;
    xhrs.delete(id);
    files.delete(id);
    // on success the server announces the document via `data:docs`; drop the local row
    if (error) useStudio.getState().patchUpload(id, { state: "error", error });
    else useStudio.getState().dropUpload(id);
    pump();
  };
  xhr.onload = () => {
    let msg = "";
    try {
      msg = JSON.parse(xhr.responseText)?.error ?? "";
    } catch {}
    finish(xhr.status >= 200 && xhr.status < 300 ? undefined : msg || `Upload failed (HTTP ${xhr.status}).`);
  };
  xhr.onerror = () => finish("Could not reach the backend (is it running?).");
  xhr.onabort = () => finish("Cancelled.");
  xhr.send(file); // a File body is streamed by the browser; a 1 GB file is never read into memory
}

export function cancelUpload(id: string) {
  const x = xhrs.get(id);
  if (x) x.abort();
  else {
    files.delete(id);
    useStudio.getState().dropUpload(id);
  }
}
