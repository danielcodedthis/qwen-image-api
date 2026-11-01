// Running a job (generate or edit) from the UI: the progress overlay, the ETA, the
// Generate / Edit buttons, and turning the API response into a result item.
import { b64ToBlob } from "./api.js";
import { config, currentModelId, modelLabel } from "./controls.js";
import { $, fmtDuration, setText } from "./dom.js";
import { emit, on } from "./events.js";
import * as result from "./result.js";
import { server } from "./status.js";

let running = false, kind = null, startTs = 0, requestedModel = null, endedAt = 0;
// A /status poll sent just before our job finished can still say busy: ignore it briefly.
const END_GRACE_MS = 2500;
let step = 0, totalSteps = 0, phase = "idle", firstStepTs = null, firstStepNo = 0, lastStepSeen = 0, etaDeadline = null;

export const isRunning = () => running;

/** Disabled-state and hint of both primary buttons. `ready` comes from each mode. */
export function renderButtons() {
  const s = server.status;
  const externalBusy = server.ready && s && s.busy && !running && Date.now() - endedAt > END_GRACE_MS;
  const base = !server.ready || !config || running || externalBusy;

  for (const [id, hintId, label, busyLabel] of [["go", "go-hint", "Generate", "Generating..."],
                                                 ["go-edit", "go-edit-hint", "Edit image", "Editing..."]]) {
    const btn = $(id);
    btn.disabled = base;
    btn.textContent = running && kind === (id === "go" ? "generate" : "edit") ? busyLabel : label;
    let hint = "";
    if (!server.ready) hint = "Waiting for the API to start...";
    else if (externalBusy) hint = "Another job is running (started elsewhere, e.g. by an agent).";
    else if (running) hint = "";
    else if (s && !s.model_loaded) hint = "The model loads on the first run, so it will take a few minutes.";
    else if (s && s.loaded_model && currentModelId() && currentModelId() !== s.loaded_model)
      hint = `${modelLabel(currentModelId())} will replace ${modelLabel(s.loaded_model)} in memory, which takes a few minutes.`;
    setText(hintId, hint);
  }
}

// ---------------------------------------------------------------- progress
on("status", s => {
  if (s && running) {
    totalSteps = s.total_steps || totalSteps;
    phase = s.phase || phase;
    // the server keeps the last run's step after it finishes; only trust it while busy
    step = s.busy ? s.step : 0;
    if (step > 0 && phase === "denoising") {
      if (firstStepTs === null) {
        firstStepTs = Date.now(); firstStepNo = step; lastStepSeen = step;
      } else if (step !== lastStepSeen) {
        lastStepSeen = step;
        // Re-estimate only when a new step lands; count down smoothly in between.
        if (step - firstStepNo >= 2 && totalSteps) {
          const perStep = (Date.now() - firstStepTs) / (step - firstStepNo);
          etaDeadline = Date.now() + (totalSteps - step) * perStep;
        }
      }
    }
  }
  renderButtons();
  renderProgress();
});

function progressTitle() {
  const s = server.status;
  const needsLoad = !s || !s.model_loaded || (requestedModel && s.loaded_model !== requestedModel);
  switch (phase) {
    case "loading": return needsLoad ? "Loading the model..." : "Starting...";
    case "encoding": return "Reading the reference images...";
    case "decoding": return "Finishing up...";
    case "saving": return "Saving...";
    case "denoising": return step === 0 ? (kind === "edit" ? "Reading the images and instruction..." : "Reading the prompt...")
                                        : (kind === "edit" ? "Editing image" : "Generating image");
    default: return "Starting...";
  }
}

function renderProgress() {
  $("overlay").hidden = !running;
  $("canvas").classList.toggle("busy", running);
  if ($("result").hidden) $("empty").hidden = running;
  if (!running) return;

  setText("progress-title", progressTitle());
  const bar = $("progress-bar");
  if (step > 0 && totalSteps) {
    bar.classList.remove("indeterminate");
    $("progress-fill").style.width = (step / totalSteps * 100) + "%";
    setText("progress-step", `Step ${step} of ${totalSteps}`);
  } else {
    bar.classList.add("indeterminate");
    $("progress-fill").style.width = "";
    setText("progress-step", "");
  }
  setText("progress-elapsed", `${fmtDuration((Date.now() - startTs) / 1000)} elapsed`);

  let eta = "";
  if (etaDeadline !== null && phase === "denoising" && !(totalSteps && step >= totalSteps)) {
    let left = Math.max(0, Math.round((etaDeadline - Date.now()) / 1000));
    if (left >= 60) left = Math.round(left / 5) * 5;  // coarser when far away, so it doesn't fidget
    eta = left === 0 ? "almost done" : `about ${fmtDuration(left)} left`;
  }
  setText("progress-eta", eta);
}
setInterval(renderProgress, 500);

// ---------------------------------------------------------------- run
/**
 * Run one job. `send` performs the request and returns the API's JSON.
 * `meta` is merged into the result item (size label, source preview, ...).
 */
export async function run({ jobKind, model, send, meta = {} }) {
  if (running) return null;
  running = true; kind = jobKind; requestedModel = model; startTs = Date.now();
  step = 0; totalSteps = 0; phase = "loading"; firstStepTs = null; firstStepNo = 0; lastStepSeen = 0; etaDeadline = null;
  result.setMeta([]);
  renderButtons();
  renderProgress();
  try {
    const data = await send();
    const now = new Date();
    const item = {
      id: data.id || String(now.getTime()),
      kind: jobKind,
      url: URL.createObjectURL(b64ToBlob(data.data[0].b64_json)),
      file: data.file,
      size: data.size,
      total: ((Date.now() - startTs) / 1000).toFixed(1),
      server: data.generation_seconds,
      time: now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      filename: data.file || `qwen_${data.size}_${now.getTime()}.png`,
      model: modelLabel(data.model || model),
      seed: data.seed,
      references: data.reference_images,
      ...meta,
    };
    result.add(item);
    result.show(item);
    emit("job", item);
    return item;
  } catch (err) {
    const secs = ((Date.now() - startTs) / 1000).toFixed(1);
    const msg = err instanceof TypeError
      ? "Lost the connection to the API. It may have restarted or run out of memory. Check the server log."
      : err.message;
    result.setMeta([result.metaSpan(`Failed after ${secs}s`, "err"), result.metaSpan(msg, "err")]);
    return null;
  } finally {
    running = false;
    endedAt = Date.now();
    renderButtons();
    renderProgress();
  }
}
