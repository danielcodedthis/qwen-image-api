import json
import logging
import subprocess
from collections import deque
from threading import Lock

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
_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_handler)


def get_recent_logs(n=50):
    with log_lock:
        return list(log_buffer)[-n:]


# --- Generation progress state ---
generation_state = {"active": False, "step": 0, "total_steps": 0}


def progress_callback(pipe, step, timestep, callback_kwargs):
    generation_state["step"] = step + 1
    return callback_kwargs


# --- GPU (ROCm) stats ---
def _run_rocm_smi(args):
    for cmd in ("rocm-smi", "/opt/rocm/bin/rocm-smi"):
        try:
            return subprocess.check_output([cmd] + args, timeout=3, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("rocm-smi not found")


def get_gpu_stats():
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
