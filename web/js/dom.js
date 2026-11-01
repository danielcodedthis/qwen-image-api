// Small DOM and formatting helpers shared by every module.

export const $ = id => document.getElementById(id);

/** el("span", {className: "x", title: "y"}, "text", childNode, ...) */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null) continue;
    if (k === "dataset") Object.assign(node.dataset, v);
    else if (k in node) node[k] = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** Only touch the DOM when the text actually changes (no flicker, no relayout). */
export function setText(id, text) {
  const node = typeof id === "string" ? $(id) : id;
  if (node.textContent !== text) node.textContent = text;
}

export function fmtDuration(sec) {
  sec = Math.max(0, Math.round(sec));
  return sec >= 60 ? `${Math.floor(sec / 60)}m ${String(sec % 60).padStart(2, "0")}s` : `${sec}s`;
}

export function gcd(a, b) { return b ? gcd(b, a % b) : a; }

export function parseSize(size) {
  const [w, h] = String(size).split("x").map(Number);
  return [w, h];
}

/** localStorage that never throws (private windows, blocked storage). Keys start with "qwen."
 *  (qwen.prompt, qwen.neg, qwen.size, qwen.model are the same keys the old UI used). */
export const store = {
  get(k) { try { return localStorage.getItem("qwen." + k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem("qwen." + k, v); } catch (e) { /* ignore */ } },
};

/** Bind an input/textarea to localStorage under `key`. */
export function persist(node, key, fallback = "") {
  const saved = store.get(key);
  node.value = saved !== null ? saved : fallback;
  node.addEventListener("input", () => store.set(key, node.value));
}

export function numOrNull(node) {
  const v = node.value.trim();
  return v === "" || isNaN(Number(v)) ? null : Number(v);
}
