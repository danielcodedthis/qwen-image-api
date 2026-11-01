// Talking to the API. The UI (nginx) and API (FastAPI) are on different ports.
// The same page serves both containers: opened on :3001 it talks to the Turbo API
// (:8001), otherwise to the base API (:8000). Override with ?api=PORT.

const params = new URLSearchParams(location.search);
export const API_PORT = params.get("api") || (location.port === "3001" ? "8001" : "8000");
export const API_BASE = `http://${location.hostname || "localhost"}:${API_PORT}`;

export const apiUrl = path => (path.startsWith("http") ? path : API_BASE + path);

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function handle(res) {
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try { detail = JSON.parse(text).detail || text; } catch (e) { /* not JSON */ }
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), res.status);
  }
  return res.json();
}

export const getJSON = path => fetch(apiUrl(path)).then(handle);

export const postJSON = (path, body) => fetch(apiUrl(path), {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
}).then(handle);

export const postForm = (path, form) => fetch(apiUrl(path), { method: "POST", body: form }).then(handle);

/** Blob -> data URL (base64), for sending uploads inside a JSON body. */
export function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

export function b64ToBlob(b64, type = "image/png") {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

/** Download a (cross-origin) file: the `download` attribute alone is ignored across origins. */
export async function downloadUrl(url, filename) {
  const blob = await fetch(apiUrl(url)).then(r => { if (!r.ok) throw new Error(r.statusText); return r.blob(); });
  const href = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 10_000);
}
