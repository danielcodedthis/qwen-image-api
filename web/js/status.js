// Polls GET /status once a second: header chips, VRAM/RAM meters, alerts, server log.
import { getJSON } from "./api.js";
import { modelLabel } from "./controls.js";
import { $ } from "./dom.js";
import { emit } from "./events.js";

export const server = { ready: false, status: null };

const MODE_LABELS = { balanced: "GPU + RAM (balanced)", cpu: "CPU only (RAM)" };

function setChip(id, state, text) {
  const node = $(id);
  node.className = "chip" + (state ? " " + state : "");
  node.querySelector(".t").textContent = text;
}

function renderChips() {
  const s = server.status;
  setChip("chip-api", server.ready ? "ok" : "err", server.ready ? "API online" : "API offline");
  if (!server.ready) {
    setChip("chip-model", "", "Model unknown");
    setChip("chip-mode", "", "Memory unknown");
    return;
  }
  setChip("chip-model", s.model_loaded ? "ok" : "warn",
          s.model_loaded ? `Loaded: ${modelLabel(s.loaded_model)}` : "No model loaded");
  setChip("chip-mode", "", "Memory: " + (MODE_LABELS[s.memory_mode] || s.memory_mode));
}

function setMeter(prefix, stats) {
  const text = $(prefix + "-text"), bar = $(prefix + "-bar");
  if (!stats || stats.error) { text.textContent = "n/a"; bar.style.width = "0%"; return; }
  const pct = stats.used_gb / stats.total_gb * 100;
  text.textContent = `${stats.used_gb} / ${stats.total_gb} GB`;
  bar.style.width = pct + "%";
  bar.className = "fill" + (pct >= 90 ? " err" : pct >= 75 ? " warn" : "");
}

function renderAlerts(list) {
  $("alerts").replaceChildren(...(list || []).map(msg => {
    const d = document.createElement("div");
    d.className = "alert";
    d.textContent = msg;
    return d;
  }));
}

function renderLogs(lines) {
  const box = $("logs");
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
  box.replaceChildren(...(lines || []).map(line => {
    const d = document.createElement("div");
    d.textContent = line;
    if (/ ERROR /.test(line)) d.className = "log-err";
    else if (/ WARNING /.test(line)) d.className = "log-warn";
    return d;
  }));
  if (atBottom) box.scrollTop = box.scrollHeight;
}

async function poll() {
  try {
    server.status = await getJSON("/status");
    server.ready = true;
    setMeter("vram", server.status.gpu);
    setMeter("ram", server.status.ram);
    renderAlerts(server.status.warnings);
    renderLogs(server.status.logs);
  } catch (e) {
    server.ready = false;
  }
  renderChips();
  emit("status", server.ready ? server.status : null);
}

export function startPolling() {
  poll();
  setInterval(poll, 1000);
}
