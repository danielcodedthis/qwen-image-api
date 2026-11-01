// Create mode: text prompt -> POST /v1/images/generations.
import { postJSON } from "./api.js";
import { commonParams, currentModelId, currentSize } from "./controls.js";
import { $, persist } from "./dom.js";
import * as jobs from "./jobs.js";
import * as result from "./result.js";

const promptEl = $("prompt");
persist(promptEl, "prompt");

// ---- transparent images ----
// Qwen-Image-2.1 makes RGBA images natively when the prompt says so (model card wording).
const RGBA_PREFIX = "This is an RGBA image with transparency. ";
const RGBA_SUFFIX = " The image has alpha channel and the background is transparent.";

$("rgba-example").addEventListener("click", () => {
  let text = promptEl.value.trim() || "A cute cartoon dragon sticker.";
  if (!text.startsWith(RGBA_PREFIX.trim())) text = RGBA_PREFIX + text;
  if (!text.includes("background is transparent")) text += RGBA_SUFFIX;
  promptEl.value = text;
  promptEl.dispatchEvent(new Event("input"));
});

export function setPrompt(text) {
  promptEl.value = text;
  promptEl.dispatchEvent(new Event("input"));
}

export async function generate() {
  if ($("go").disabled) return;
  const prompt = promptEl.value.trim();
  if (!prompt) {
    result.setMeta([result.metaSpan("Enter a prompt before generating.", "err")]);
    promptEl.focus();
    return;
  }
  const size = currentSize(), model = currentModelId();
  await jobs.run({
    jobKind: "generate",
    model,
    send: () => postJSON("/v1/images/generations", { prompt, size, ...commonParams() }),
  });
}

$("go").addEventListener("click", generate);
for (const node of [promptEl, $("neg")]) {
  node.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !$("go").closest("[hidden]")) { e.preventDefault(); generate(); }
  });
}
