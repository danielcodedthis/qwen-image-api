// Entry point: wires the modules together and switches between Create, Edit and Gallery.
import { API_PORT } from "./api.js";
import { applyFields, loadConfig } from "./controls.js";
import * as create from "./create.js";
import { $ } from "./dom.js";
import * as edit from "./edit.js";
import { on } from "./events.js";
import * as gallery from "./gallery.js";
import * as result from "./result.js";
import { current, go } from "./router.js";
import { startPolling } from "./status.js";

const EMPTY_TEXT = {
  create: "Write a prompt, pick a size and press Generate.",
  edit: "Add one or more images, write what to change and press Edit image.",
};
const TITLE = API_PORT === "8001" ? "Qwen Image Turbo" : "Qwen Image";
$("app-title").textContent = TITLE;

function showView({ view, id }) {
  document.querySelectorAll("[data-view]").forEach(b => {
    if (b.dataset.view === view) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  document.querySelectorAll("[data-modes]").forEach(s => {
    s.hidden = !s.dataset.modes.split(" ").includes(view);
  });
  $("stage").hidden = view === "gallery";
  $("gallery").hidden = view !== "gallery";
  if (view === "gallery") gallery.enter(id);
  else result.setEmptyText(EMPTY_TEXT[view]);
  document.title = `${TITLE} · ${view[0].toUpperCase()}${view.slice(1)}`;
}

document.querySelectorAll("[data-view]").forEach(b => b.addEventListener("click", () => go(b.dataset.view)));
on("route", showView);

// "Edit this" (stage or gallery): start a new edit with that image and switch to Edit.
on("use-as-source", async src => {
  go("edit");
  try {
    edit.clearSources();
    await edit.addImage(src);
  } catch (e) { result.setMeta([result.metaSpan("Couldn't load that image: " + e.message, "err")]); }
});

// "Reuse settings" in the gallery: back to the mode the image was made in, filled in.
on("reuse-settings", item => {
  const fields = item.fields || {};
  applyFields(fields, item.negative_prompt);
  if (fields.mode === "edit") {
    edit.setPrompt(item.prompt || "");
    go("edit");
    edit.clearSources();
    (async () => {
      for (const [i, url] of (item.source_urls || []).entries())
        await edit.addImage({ url, name: null, label: `Picture ${i + 1} of ${item.file}` });
    })();
  } else {
    create.setPrompt(item.prompt || "");
    go("create");
  }
});

showView(current());
startPolling();
loadConfig();
gallery.refreshCount();
