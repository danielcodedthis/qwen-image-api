// Gallery: every image in output/ as a grid of thumbnails; selecting one shows it
// full size with its output/logs/<id>.log below.
import { apiUrl, downloadUrl, getJSON } from "./api.js";
import { $, el } from "./dom.js";
import { emit, on } from "./events.js";
import { go } from "./router.js";

const PAGE = 60;
let items = [], total = 0, loading = false, detail = null;

// ---------------------------------------------------------------- list
function tile(it) {
  const caption = it.prompt ? it.prompt.replace(/\s+/g, " ").slice(0, 120) : it.file;
  const btn = el("button", { type: "button", className: "tile", title: caption },
    el("img", { src: apiUrl(it.thumb_url), alt: caption, loading: "lazy", decoding: "async" }),
    it.kind === "edit" ? el("span", { className: "badge" }, it.sources > 1 ? `edit · ${it.sources} images` : "edit") : null,
    el("span", { className: "cap" }, [it.size, caption].filter(Boolean).join(" · ")));
  btn.addEventListener("click", () => go("gallery", it.id));
  return btn;
}

function renderList() {
  $("grid").replaceChildren(...items.map(tile));
  $("gallery-empty").hidden = total > 0;
  $("gallery-more").hidden = items.length >= total;
  $("gallery-summary").textContent = total ? `${total} image${total === 1 ? "" : "s"} in output/` : "";
  setCount(total);
}

function setCount(n) { $("gallery-count").textContent = n ? `(${n})` : ""; }

async function loadPage(reset) {
  if (loading) return;
  loading = true;
  try {
    const page = await getJSON(`/v1/gallery?offset=${reset ? 0 : items.length}&limit=${PAGE}`);
    items = reset ? page.items : items.concat(page.items);
    total = page.total;
    renderList();
  } catch (e) {
    $("gallery-summary").textContent = "Couldn't load the gallery: " + e.message;
  } finally {
    loading = false;
  }
}

export async function refreshCount() {
  try { setCount((await getJSON("/v1/gallery?limit=1")).total); } catch (e) { /* API offline */ }
}

$("gallery-more").addEventListener("click", () => loadPage(false));
$("gallery-refresh").addEventListener("click", () => loadPage(true));
on("job", () => { refreshCount(); if (!$("gallery").hidden) loadPage(true); });

// ---------------------------------------------------------------- detail
const FACTS = [
  ["status", "Status"], ["mode", "Mode"], ["model", "Model"], ["size", "Size"], ["seed", "Seed"],
  ["steps_requested", "Steps"], ["steps_ran", "Ran"], ["true_cfg_scale", "CFG"], ["reference_images", "Images"],
  ["edit_resolution", "Edit resolution"], ["duration_seconds", "Seconds"], ["peak_vram_gb", "Peak VRAM GB"],
  ["peak_ram_gb", "Peak RAM GB"], ["memory_mode", "Memory"], ["time", "Time"],
];

async function openDetail(id) {
  $("gallery-list").hidden = true;
  $("gallery-detail").hidden = false;
  $("detail-img").removeAttribute("src");
  $("detail-facts").replaceChildren();
  $("detail-log").textContent = "Loading...";
  $("detail-source").hidden = true;
  try {
    detail = await getJSON(`/v1/gallery/${encodeURIComponent(id)}`);
  } catch (e) {
    detail = null;
    $("detail-log-label").textContent = "";
    $("detail-log").textContent = e.status === 404 ? `No image named ${id}.png in output/.` : e.message;
    return;
  }
  const img = $("detail-img");
  img.src = apiUrl(detail.image_url);
  img.alt = detail.prompt || detail.file;
  $("detail-open").href = apiUrl(detail.image_url);

  const f = detail.fields || {};
  $("detail-facts").replaceChildren(...FACTS.filter(([k]) => f[k] !== undefined && f[k] !== "")
    .map(([k, label]) => el("span", {}, `${label} `, el("b", {}, f[k]))));
  if (f.error) $("detail-facts").append(el("span", { className: "err" }, `Error: ${f.error}`));

  const refs = detail.source_urls || [];
  $("detail-source").hidden = !refs.length;
  $("detail-source-imgs").replaceChildren(...refs.map((url, i) =>
    el("a", { href: apiUrl(url), target: "_blank", rel: "noopener", title: `Picture ${i + 1}` },
       el("img", { src: apiUrl(url), alt: `Picture ${i + 1}` }))));

  $("detail-log-label").textContent = detail.log ? `output/logs/${detail.id}.log` : "No log file for this image";
  $("detail-log").textContent = detail.log || "This image has no output/logs/<name>.log (made outside the API, or before logs existed).";
  $("detail-reuse").disabled = !detail.prompt;
  window.scrollTo({ top: 0 });
}

function showList() {
  $("gallery-detail").hidden = true;
  $("gallery-list").hidden = false;
  detail = null;
}

$("detail-back").addEventListener("click", () => go("gallery"));
$("detail-download").addEventListener("click", e => {
  e.preventDefault();
  if (detail) downloadUrl(detail.image_url, detail.file);
});
$("detail-edit").addEventListener("click", () => {
  if (detail) emit("use-as-source", { url: detail.image_url, name: detail.file, label: detail.file });
});
$("detail-reuse").addEventListener("click", () => {
  if (detail) emit("reuse-settings", detail);
});

/** Called by the router whenever the gallery view is shown. */
export function enter(id) {
  if (id) openDetail(id);
  else {
    showList();
    loadPage(true);
  }
}
