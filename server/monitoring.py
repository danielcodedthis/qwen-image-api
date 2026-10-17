import json
import logging
import subprocess
import time
from collections import deque
from threading import Lock

import torch

from config import config

# --- In-memory log capture ---
log_buffer = deque(maxlen=200)
log_lock = Lock()


class BufferHandler(logging.Handler):
    def emit(self, record):
        with log_lock:
            log_buffer.append(self.format(record))


logger = logging.getLogger("qwen-image-api")
logger.setLevel(logging.INFO)
_handler = BufferHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(_handler)

# Also echo to the container's stdout so `docker logs` shows warnings/errors.
_stream = logging.StreamHandler()
_stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_stream)


def get_recent_logs(n=50):
    with log_lock:
        return list(log_buffer)[-n:]


# --- Settings ---
RUN_ON_RAM = config["memory"]["run_on_ram"]
VRAM_WARN_PCT = config["monitoring"]["vram_warn_percent"]
RAM_WARN_PCT = config["monitoring"]["ram_warn_percent"]

# --- Generation progress state ---
generation_state = {
    "active": False,
    "step": 0,
    "total_steps": 0,
    "peak_vram_gb": 0.0,
    "peak_ram_gb": 0.0,
}

# Tracks which warnings are currently active (so we log once per crossing,
# not on every step) and the last OOM that happened.
alert_state = {"vram": False, "ram": False, "last_oom": None}


def start_generation(total_steps):
    generation_state.update(
        {"active": True, "step": 0, "total_steps": total_steps, "peak_vram_gb": 0.0, "peak_ram_gb": 0.0}
    )


# --- GPU (ROCm) stats ---
def _run_rocm_smi(args):
    for cmd in ("rocm-smi", "/opt/rocm/bin/rocm-smi"):
        try:
            return subprocess.check_output([cmd] + args, timeout=3, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("rocm-smi not found")


def get_gpu_stats():
    """Whole-card VRAM via rocm-smi. Safe to call while idle (no CUDA context created)."""
    try:
        out = _run_rocm_smi(["--showmeminfo", "vram", "--json"])
        data = json.loads(out)
        for card in data.values():
            if "VRAM Total Used Memory (B)" in card:
                used = int(card["VRAM Total Used Memory (B)"])
                total = int(card["VRAM Total Memory (B)"])
                return {"used_gb": round(used / 1024**3, 2), "total_gb": round(total / 1024**3, 2)}
        return {"error": "no card data"}
    except Exception as e:
        return {"error": str(e)}


def _fast_gpu_stats():
    """Cheap in-process VRAM reading for use inside the step callback (no subprocess)."""
    if RUN_ON_RAM or not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info()
        return {"used_gb": round((total - free) / 1024**3, 2), "total_gb": round(total / 1024**3, 2)}
    except Exception:
        return None


# --- System RAM stats ---
def get_ram_stats():
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].split()[0])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", 0)
        return {
            "used_gb": round((total_kb - avail_kb) / 1024**2, 2),
            "total_gb": round(total_kb / 1024**2, 2),
        }
    except Exception as e:
        return {"error": str(e)}


# --- Memory pressure warnings ---
def _pct(stats):
    if not stats or "error" in stats or not stats.get("total_gb"):
        return None
    return stats["used_gb"] / stats["total_gb"] * 100


def evaluate_pressure(gpu, ram):
    """Return a list of {"kind", "message"} for any resource over its threshold."""
    warnings = []
    pct = _pct(gpu)
    if pct is not None and pct >= VRAM_WARN_PCT:
        warnings.append({
            "kind": "vram",
            "message": f"VRAM at {pct:.0f}% ({gpu['used_gb']}/{gpu['total_gb']} GB) — out-of-memory risk",
        })
    pct = _pct(ram)
    if pct is not None and pct >= RAM_WARN_PCT:
        warnings.append({
            "kind": "ram",
            "message": f"System RAM at {pct:.0f}% ({ram['used_gb']}/{ram['total_gb']} GB) — out-of-memory risk",
        })
    return warnings


def log_pressure_transitions(warnings):
    """Log a warning the first time a threshold is crossed; re-arm once it drops back."""
    active = {w["kind"]: w["message"] for w in warnings}
    for kind in ("vram", "ram"):
        if kind in active and not alert_state[kind]:
            logger.warning(f"MEMORY WARNING: {active[kind]}")
            alert_state[kind] = True
        elif kind not in active:
            alert_state[kind] = False


def get_active_warnings(gpu, ram):
    """Warnings for /status: live pressure + the most recent unresolved OOM."""
    pressure = evaluate_pressure(gpu, ram)
    log_pressure_transitions(pressure)
    out = [w["message"] for w in pressure]
    oom = alert_state["last_oom"]
    if oom:
        out.insert(0, f"Out of memory at {oom['time']} during {oom['stage']}: {oom['detail']}")
    return out


def progress_callback(pipe, step, timestep, callback_kwargs):
    generation_state["step"] = step + 1

    gpu = _fast_gpu_stats()
    ram = get_ram_stats()
    for stats, key in ((gpu, "peak_vram_gb"), (ram, "peak_ram_gb")):
        if stats and "error" not in stats:
            generation_state[key] = max(generation_state[key], stats["used_gb"])
    log_pressure_transitions(evaluate_pressure(gpu, ram))
    return callback_kwargs


# --- OOM handling ---
_OOM_MARKERS = (
    "out of memory",
    "hiperroroutofmemory",
    "cannot allocate memory",
    "defaultcpuallocator",
    "bad_alloc",
)


def is_oom_error(exc):
    """True if exc (or anything it was raised from) looks like an out-of-memory failure."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
        if isinstance(exc, MemoryError) or (cuda_oom and isinstance(exc, cuda_oom)):
            return True
        if any(m in str(exc).lower() for m in _OOM_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def report_oom(exc, stage="generation"):
    """Log a loud, actionable OOM message with a memory snapshot and remember it for /status."""
    gpu = get_gpu_stats()
    ram = get_ram_stats()
    detail = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:200]
    alert_state["last_oom"] = {"time": time.strftime("%H:%M:%S"), "stage": stage, "detail": detail}

    logger.error(f"OUT OF MEMORY during {stage}: {detail}")
    if "error" not in gpu:
        logger.error(f"  VRAM at failure: {gpu['used_gb']}/{gpu['total_gb']} GB")
    if "error" not in ram:
        logger.error(f"  RAM at failure: {ram['used_gb']}/{ram['total_gb']} GB")
    logger.error(
        f"  Peak during this run: VRAM {generation_state['peak_vram_gb']} GB, "
        f"RAM {generation_state['peak_ram_gb']} GB"
    )
    if RUN_ON_RAM:
        logger.error("  Hint: lower the resolution, or set memory.quantize_transformer: true.")
    else:
        logger.error(
            "  Hint: lower the resolution. If memory.vae_fp32 is on, turn it off. "
            "Or set memory.vae_on_cpu: true to decode the image in system RAM, "
            "lower memory.gpu_max_memory, or set memory.run_on_ram: true (slow)."
        )


def clear_oom():
    alert_state["last_oom"] = None
