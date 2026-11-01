// The stage: shows the latest (or a selected) result, its metadata, the history strip,
// and for edits a Compare button that flips between the result and its source.
import { apiUrl } from "./api.js";
import { setSeed } from "./controls.js";
import { $, el } from "./dom.js";
import { emit } from "./events.js";

const MAX_HISTORY = 12;
const history = [];
let shown = null, comparing = false;

export function metaSpan(text, cls) { return el("span", { className: cls || "" }, text); }
export function setMeta(parts) { $("meta").replaceChildren(...parts); }

function setCompare(on) {
  comparing = on && !!(shown && shown.sourceUrl);
  $("result").src = comparing ? shown.sourceUrl : shown.url;
  $("compare").setAttribute("aria-pressed", String(comparing));
  $("compare-tag").hidden = !comparing;
}

export function show(item) {
  shown = item;
  comparing = false;
  const img = $("result");
  img.src = item.url;
  img.hidden = false;
  $("empty").hidden = true;
  $("compare-tag").hidden = true;
  $("open-full").href = item.file ? apiUrl(`/v1/gallery/${encodeURIComponent(item.id)}/image`) : item.url;
  $("open-full").hidden = false;
  $("download").href = item.url;
  $("download").download = item.filename;
  $("download").hidden = false;
  $("compare").hidden = !item.sourceUrl;
  $("compare").setAttribute("aria-pressed", "false");
  $("edit-this").hidden = false;

  const edited = item.kind === "edit" ? `Edit${item.references > 1 ? ` (${item.references} images)` : ""} · ` : "";
  const parts = [metaSpan(edited + (item.model || "")),
                 metaSpan(`${String(item.size).replace("x", "×")} px`),
                 metaSpan(`${item.total}s total`, "ok")];
  if (item.server != null) parts.push(metaSpan(`${item.server}s on the server`));
  if (item.seed != null) {
    const b = el("button", { type: "button", className: "link-btn", title: "Use this seed again" }, `seed ${item.seed}`);
    b.addEventListener("click", () => setSeed(item.seed));
    parts.push(b);
  }
  parts.push(metaSpan(item.time));
  setMeta(parts);
  document.querySelectorAll("#history .thumb").forEach(t => t.setAttribute("aria-current", String(t.dataset.id === item.id)));
}

export function add(item) {
  history.unshift(item);
  while (history.length > MAX_HISTORY) {
    const old = history.pop();
    URL.revokeObjectURL(old.url);
    if (old.sourceUrl && old.sourceUrl.startsWith("blob:")) URL.revokeObjectURL(old.sourceUrl);
  }
  $("history").replaceChildren(...history.map(h => {
    const b = el("button", { type: "button", className: "thumb", title: `${h.kind === "edit" ? "Edit, " : ""}${h.size}, ${h.time}` },
      el("img", { src: h.url, alt: "" }));
    b.dataset.id = h.id;
    b.addEventListener("click", () => show(h));
    return b;
  }));
}

/** Text for the empty stage, per mode. */
export function setEmptyText(text) { $("empty-text").textContent = text; }

// Compare: click toggles; holding the button shows the source while pressed.
const compareBtn = $("compare");
let holdTimer = null, held = false;
compareBtn.addEventListener("pointerdown", () => { held = false; holdTimer = setTimeout(() => { held = true; setCompare(true); }, 250); });
compareBtn.addEventListener("pointerup", () => { clearTimeout(holdTimer); if (held) setCompare(false); else setCompare(!comparing); });
compareBtn.addEventListener("pointerleave", () => { clearTimeout(holdTimer); if (held) { held = false; setCompare(false); } });

$("edit-this").addEventListener("click", () => {
  if (!shown) return;
  emit("use-as-source", { url: shown.url, name: shown.file, label: shown.file || "previous result" });
});
