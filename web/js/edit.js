// Edit mode: one or more reference images + an instruction -> POST /v1/images/edits.
// Qwen-Image-2.1 edits natively: "Change the background to a sunset beach". With several
// images the prompt refers to them as Picture 1, Picture 2, ... in the order shown.
import { apiUrl, blobToDataUrl, postJSON } from "./api.js";
import { autoEditSize, commonParams, config, currentEditSize, currentModelId,
         setAutoSizeLabel } from "./controls.js";
import { $, el, persist } from "./dom.js";
import { on } from "./events.js";
import * as jobs from "./jobs.js";
import * as result from "./result.js";

const promptEl = $("edit-prompt"), zone = $("dropzone"), fileInput = $("source-file");
persist(promptEl, "editPrompt");

/** [{ blob, previewUrl, width, height, name (file in output/ or null), label }] */
let sources = [];
const maxImages = () => (config ? config.edit.max_images : 4);

function describeSize() {
  const last = sources[sources.length - 1];
  if (!last || !config) { setAutoSizeLabel("like the last image"); $("edit-size-hint").textContent = ""; return; }
  const [w, h] = autoEditSize(last.width, last.height);
  setAutoSizeLabel(`${w}×${h}`);
  $("edit-size-hint").textContent = currentEditSize() === "auto"
    ? (sources.length > 1 ? "Shaped like the last image" : "Same aspect ratio as the image")
    : "";
}
on("controls", describeSize);
on("config", () => { describeSize(); render(); });

function render() {
  const list = $("source-list");
  list.replaceChildren(...sources.map((s, i) => {
    const remove = el("button", { type: "button", className: "ref-remove", title: "Remove",
                                  ariaLabel: `Remove Picture ${i + 1}` }, "×");
    remove.addEventListener("click", e => { e.stopPropagation(); removeAt(i); });
    return el("figure", { className: "ref", title: `${s.label} · ${s.width}×${s.height}` },
      el("img", { src: s.previewUrl, alt: `Picture ${i + 1}` }),
      el("figcaption", {}, `Picture ${i + 1}`), remove);
  }));
  const full = sources.length >= maxImages();
  list.hidden = !sources.length;
  zone.hidden = full;
  $("drop-empty-title").textContent = sources.length ? "Add another image" : "Drop images here";
  $("source-clear").hidden = !sources.length;
  $("source-hint").textContent = sources.length
    ? `${sources.length} of ${maxImages()} image${maxImages() === 1 ? "" : "s"}` +
      (sources.length > 1 ? ". Refer to them as Picture 1, Picture 2, … in the prompt." : ".")
    : `Up to ${maxImages()} images (PNG, JPEG or WebP; transparency is kept). You can also use one from the Gallery.`;
  describeSize();
}

async function readSize(blob) {
  const bmp = await createImageBitmap(blob);
  const size = { width: bmp.width, height: bmp.height };
  bmp.close();
  return size;
}

async function addBlob(blob, name, label) {
  if (sources.length >= maxImages()) {
    $("source-hint").textContent = `At most ${maxImages()} images (edit.max_images in the config).`;
    return;
  }
  let size;
  try {
    size = await readSize(blob);
  } catch (e) {
    $("source-hint").textContent = "That file isn't an image the browser can read. Use PNG, JPEG or WebP.";
    return;
  }
  sources.push({ blob, previewUrl: URL.createObjectURL(blob), ...size, name: name || null, label: label || "upload" });
  render();
}

function removeAt(i) {
  const [gone] = sources.splice(i, 1);
  if (gone) URL.revokeObjectURL(gone.previewUrl);
  render();
}

export function clearSources() {
  sources.forEach(s => URL.revokeObjectURL(s.previewUrl));
  sources = [];
  fileInput.value = "";
  render();
}

/** Add an image already on the server (gallery) or a blob URL (a result on the stage). */
export async function addImage({ url, name, label }) {
  const blob = await fetch(url.startsWith("blob:") ? url : apiUrl(url)).then(r => r.blob());
  await addBlob(blob, name, label || name);
}

async function addFiles(files) {
  for (const f of [...files].filter(f => f.type.startsWith("image/"))) await addBlob(f, null, f.name);
}

// ---- picking files: click, keyboard, drop, paste ----
zone.addEventListener("click", () => fileInput.click());
zone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", async () => { await addFiles(fileInput.files); fileInput.value = ""; });
$("source-clear").addEventListener("click", clearSources);
for (const target of [zone, $("source-list")]) {
  target.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("over"); });
  target.addEventListener("dragleave", () => zone.classList.remove("over"));
  target.addEventListener("drop", e => { e.preventDefault(); zone.classList.remove("over"); addFiles(e.dataTransfer.files || []); });
}
document.addEventListener("paste", e => {
  if (promptEl.closest("[hidden]")) return;  // only in Edit mode
  const items = [...(e.clipboardData?.items || [])].filter(i => i.type.startsWith("image/"));
  if (!items.length) return;
  e.preventDefault();
  items.forEach(i => addBlob(i.getAsFile(), null, "pasted image"));
});

// ---- run ----
export async function runEdit() {
  if ($("go-edit").disabled) return;
  if (!sources.length) {
    result.setMeta([result.metaSpan("Add at least one image first.", "err")]);
    zone.focus();
    return;
  }
  const prompt = promptEl.value.trim();
  if (!prompt) {
    result.setMeta([result.metaSpan("Write what to change before editing.", "err")]);
    promptEl.focus();
    return;
  }
  const size = currentEditSize();
  const refs = [...sources];
  const send = async () => {
    // One JSON request keeps the order: images already in output/ go by name, the rest as data URLs.
    const images = await Promise.all(refs.map(async s => (s.name ? { source: s.name } : { data: await blobToDataUrl(s.blob) })));
    const body = { prompt, images, ...commonParams() };
    if (size !== "auto") body.size = size;
    return postJSON("/v1/images/edits", body);
  };
  const last = refs[refs.length - 1];
  await jobs.run({ jobKind: "edit", model: currentModelId(), send,
                   meta: { sourceUrl: URL.createObjectURL(last.blob) } });
}

$("go-edit").addEventListener("click", runEdit);
for (const node of [promptEl, $("neg")]) {
  node.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !$("go-edit").closest("[hidden]")) { e.preventDefault(); runEdit(); }
  });
}

export function setPrompt(text) {
  promptEl.value = text;
  promptEl.dispatchEvent(new Event("input"));
}

render();
