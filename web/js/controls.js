// Settings shared by Create and Edit: model, negative prompt, advanced overrides,
// plus the size pickers (Create sizes, Edit output sizes).
import { getJSON } from "./api.js";
import { $, el, gcd, numOrNull, parseSize, persist, store } from "./dom.js";
import { emit } from "./events.js";

export let config = null;
let selectedModel = null, selectedSize = null, selectedEditSize = "auto", defaultNeg = "";

// ---------------------------------------------------------------- models
export function modelLabel(id) {
  const m = config && config.models.find(x => x.id === id);
  return m ? m.label : (id || "none");
}
export const currentModelId = () => selectedModel;
export function currentModel() {
  return config.models.find(m => m.id === selectedModel) || config.models[0];
}

function stepsText(m) {
  const steps = m.fixed_steps ? `${m.fixed_steps} steps (fixed)` : `${m.total_steps} steps`;
  return `${steps}, ${m.true_cfg_scale > 1 ? `CFG ${m.true_cfg_scale}` : "CFG off"}`;
}

function renderModels() {
  const stored = store.get("model");
  const ok = id => config.models.some(m => m.id === id && m.available);
  const firstOk = (config.models.find(m => m.available) || config.models[0]).id;
  const pick = ok(stored) ? stored : ok(config.default_model) ? config.default_model : firstOk;

  $("models").replaceChildren(...config.models.map(m => {
    const btn = el("button", { type: "button", className: "size", disabled: !m.available,
                               title: m.available ? "" : `Not downloaded yet: ${m.path}` },
      el("span", {}, el("span", { className: "dims" }, m.label),
         el("span", { className: "ratio" }, m.available ? stepsText(m) : "Not downloaded")));
    btn.dataset.model = m.id;
    btn.addEventListener("click", () => selectModel(m.id));
    return btn;
  }));
  // One model per container is the normal setup: then there is nothing to choose.
  $("models-section").hidden = config.models.length < 2;
  selectModel(pick);
}

function selectModel(id) {
  selectedModel = id;
  store.set("model", id);
  document.querySelectorAll("#models .size").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.model === id)));
  const m = currentModel();
  defaultNeg = m.negative_prompt || "";
  $("model-hint").textContent = stepsText(m);
  const stepsEl = $("steps");
  stepsEl.disabled = !!m.fixed_steps;
  stepsEl.placeholder = m.fixed_steps ? `${m.fixed_steps} (fixed)` : String(m.total_steps);
  if (m.fixed_steps) stepsEl.title = "This checkpoint always runs its saved schedule.";
  else stepsEl.removeAttribute("title");
  $("cfg").placeholder = m.true_cfg_scale > 1 ? String(m.true_cfg_scale) : "1 (off)";
  updateNegHint();
  emit("controls");
}

// ---------------------------------------------------------------- sizes
const fits = size => { const [w, h] = parseSize(size); return !config.max_pixels || w * h <= config.max_pixels; };

function sizeButton(size, label, sub, onPick) {
  const [w, h] = parseSize(size);
  const k = w && h ? 20 / Math.max(w, h) : 0;
  const box = el("span", { className: "box" });
  if (k) box.append(el("i", { style: `width:${Math.round(w * k)}px;height:${Math.round(h * k)}px` }));
  const btn = el("button", { type: "button", className: "size" }, box,
    el("span", {}, el("span", { className: "dims" }, label), el("span", { className: "ratio" }, sub)));
  btn.dataset.size = size;
  btn.addEventListener("click", () => onPick(size));
  return btn;
}

function ratioLabel(size) {
  const [w, h] = parseSize(size), g = gcd(w, h);
  return `${w / g}:${h / g}`;
}

function renderSizes() {
  const sizes = config.available_sizes;
  const stored = store.get("size");
  selectedSize = sizes.includes(stored) && fits(stored) ? stored : (sizes.find(fits) || sizes[0]);
  $("sizes").replaceChildren(...sizes.map(size => {
    const btn = sizeButton(size, size.replace("x", "×"), ratioLabel(size), selectSize);
    btn.disabled = !fits(size);
    if (btn.disabled) btn.title = "Over max_pixels in the config";
    return btn;
  }));
  selectSize(selectedSize);
}

export function selectSize(size) {
  if (!document.querySelector(`#sizes .size[data-size="${size}"]:not(:disabled)`)) return;
  selectedSize = size;
  store.set("size", size);
  document.querySelectorAll("#sizes .size").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.size === size)));
}
export const currentSize = () => selectedSize;

function renderEditSizes() {
  const auto = sizeButton("auto", "Auto", "like the last image", selectEditSize);
  auto.querySelector(".box").textContent = "↔";
  const rest = config.available_sizes.map(size => {
    const btn = sizeButton(size, size.replace("x", "×"), ratioLabel(size), selectEditSize);
    btn.disabled = !fits(size);
    return btn;
  });
  $("edit-sizes").replaceChildren(auto, ...rest);
  const stored = store.get("editSize");
  selectEditSize(stored && (stored === "auto" || (config.available_sizes.includes(stored) && fits(stored))) ? stored : "auto");
}

export function selectEditSize(size) {
  selectedEditSize = size;
  store.set("editSize", size);
  document.querySelectorAll("#edit-sizes .size").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.size === size)));
  emit("controls");
}
export const currentEditSize = () => selectedEditSize;

/** Same rule as server/imaging.py auto_edit_size (= the pipeline's own), so the UI can
 *  show the output size in advance: the image's aspect ratio at resolution², sides x32. */
export function autoEditSize(srcW, srcH) {
  const res = config.edit.resolution, ratio = srcW / srcH;
  const r32 = x => Math.max(32, Math.round(x / 32) * 32);
  const wf = Math.sqrt(res * res * ratio);
  let w = r32(wf), h = r32(wf / ratio);
  while (w * h > config.max_pixels && Math.max(w, h) > 32) { if (w >= h) w -= 32; else h -= 32; }
  return [w, h];
}

export function setAutoSizeLabel(text) {
  const auto = document.querySelector('#edit-sizes .size[data-size="auto"] .ratio');
  if (auto) auto.textContent = text;
}

// ---------------------------------------------------------------- negative + advanced
const negEl = $("neg"), stepsEl = $("steps"), cfgEl = $("cfg"), seedEl = $("seed");
$("neg-reset").addEventListener("click", () => { negEl.value = defaultNeg; store.set("neg", defaultNeg); });
negEl.addEventListener("input", () => store.set("neg", negEl.value));
persist(stepsEl, "steps");
persist(cfgEl, "cfg");
persist(seedEl, "seed");
cfgEl.addEventListener("input", updateNegHint);
$("adv-details").open = !!(stepsEl.value || cfgEl.value || seedEl.value);
$("adv-reset").addEventListener("click", () => {
  [stepsEl, cfgEl, seedEl].forEach(n => { n.value = ""; n.dispatchEvent(new Event("input")); });
});

/** The negative prompt only matters when CFG is on (> 1). */
function updateNegHint() {
  if (!config) return;
  const cfg = numOrNull(cfgEl) ?? currentModel().true_cfg_scale;
  $("neg-hint").textContent = cfg > 1 ? "Used (CFG on)" : "Ignored while CFG is off";
}

export function setSeed(seed) {
  seedEl.value = String(seed);
  seedEl.dispatchEvent(new Event("input"));
  $("adv-details").open = true;
}

/** Fields both endpoints take. Empty advanced fields mean "use the config value". */
export function commonParams() {
  const steps = numOrNull(stepsEl), seed = numOrNull(seedEl);
  return {
    model: selectedModel,
    negative_prompt: negEl.value,
    steps: steps === null || stepsEl.disabled ? null : Math.round(steps),
    true_cfg_scale: numOrNull(cfgEl),
    seed: seed === null ? null : Math.round(seed),
  };
}

/** Fill the shared controls from a gallery item's log fields ("Reuse settings"). */
export function applyFields(fields, negative) {
  const modelId = (fields.model || "").split(" ")[0];
  if (config.models.some(m => m.id === modelId && m.available)) selectModel(modelId);
  if (fields.seed) setSeed(fields.seed);
  if (negative) { negEl.value = negative; store.set("neg", negative); }
  if (fields.size && config.available_sizes.includes(fields.size)) selectSize(fields.size);
}

// ---------------------------------------------------------------- load
export async function loadConfig() {
  if (config) return config;
  try {
    config = await getJSON("/config");
  } catch (e) {
    setTimeout(loadConfig, 3000);  // API not up yet: retry
    return null;
  }
  renderModels();
  renderSizes();
  renderEditSizes();
  const storedNeg = store.get("neg");
  negEl.value = storedNeg !== null ? storedNeg : defaultNeg;
  updateNegHint();
  emit("config", config);
  return config;
}
