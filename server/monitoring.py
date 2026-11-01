"""Logging, job progress, GPU/RAM stats, memory-pressure warnings and OOM detection.

Nothing here imports torch at module level: the API, the tests and the gallery work
without it. Torch is imported lazily only where an in-process VRAM reading or an
OOM type check needs it.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field

LOGGER_NAME = "qwen-image-api"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%H:%M:%S"

logger = logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------- log capture
class LogBuffer(logging.Handler):
    """Keeps the most recent log lines in memory for /status and per-job .log files.

    Every line gets a monotonically increasing sequence number, so a job can ask for
    "everything logged since I started" even after the ring buffer has wrapped.
    """

    def __init__(self, maxlen: int = 400):
        super().__init__()
        self._lines: deque[tuple[int, str]] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))

    def emit(self, record: logging.LogRecord) -> None:
        text = self.format(record)
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, text))

    def recent(self, n: int = 50) -> list[str]:
        with self._lock:
            return [line for _, line in list(self._lines)[-n:]]

    def cursor(self) -> int:
        with self._lock:
            return self._seq

    def since(self, cursor: int) -> list[str]:
        with self._lock:
            return [line for seq, line in self._lines if seq > cursor]


log_buffer = LogBuffer()


def setup_logging() -> None:
    """Attach the in-memory buffer and a stdout handler once (idempotent)."""
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if log_buffer not in logger.handlers:
        logger.addHandler(log_buffer)
    if not any(getattr(h, "_app_stdout", False) for h in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        stream._app_stdout = True  # type: ignore[attr-defined]
        logger.addHandler(stream)


setup_logging()


# ---------------------------------------------------------------- resource stats
def get_ram_stats() -> dict:
    """System RAM from /proc/meminfo (MemTotal - MemAvailable)."""
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
    except Exception as e:  # noqa: BLE001 - stats must never break a request
        return {"error": str(e)}


def _run_rocm_smi(args: list[str]) -> bytes:
    for cmd in ("rocm-smi", "/opt/rocm/bin/rocm-smi"):
        try:
            return subprocess.check_output([cmd] + args, timeout=3, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("rocm-smi not found")


_gpu_cache: dict = {"ts": 0.0, "value": None}
_gpu_cache_lock = threading.Lock()
GPU_STATS_TTL = 0.5  # seconds; the UI, MCP and several tabs may all poll /status


def get_gpu_stats() -> dict:
    """Whole-card VRAM via rocm-smi. Safe while idle (creates no GPU context)."""
    with _gpu_cache_lock:
        if _gpu_cache["value"] is not None and time.monotonic() - _gpu_cache["ts"] < GPU_STATS_TTL:
            return dict(_gpu_cache["value"])
    try:
        data = json.loads(_run_rocm_smi(["--showmeminfo", "vram", "--json"]))
        value: dict = {"error": "no card data"}
        for card in data.values():
            if "VRAM Total Used Memory (B)" in card:
                used = int(card["VRAM Total Used Memory (B)"])
                total = int(card["VRAM Total Memory (B)"])
                value = {"used_gb": round(used / 1024**3, 2), "total_gb": round(total / 1024**3, 2)}
                break
    except Exception as e:  # noqa: BLE001
        value = {"error": str(e)}
    with _gpu_cache_lock:
        _gpu_cache.update(ts=time.monotonic(), value=value)
    return dict(value)


def fast_gpu_stats() -> dict | None:
    """Cheap in-process VRAM reading for the per-step hook (no subprocess)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info()
        return {"used_gb": round((total - free) / 1024**3, 2), "total_gb": round(total / 1024**3, 2)}
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- OOM detection
_OOM_MARKERS = (
    "out of memory",
    "hiperroroutofmemory",
    "cannot allocate memory",
    "defaultcpuallocator",
    "bad_alloc",
)


def is_oom_error(exc: BaseException | None) -> bool:
    """True if exc (or anything it was raised from) looks like an out-of-memory failure."""
    try:
        import torch
        cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    except ImportError:
        cuda_oom = None
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, MemoryError) or (cuda_oom and isinstance(exc, cuda_oom)):
            return True
        if any(m in str(exc).lower() for m in _OOM_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# ---------------------------------------------------------------- job progress
@dataclass
class Progress:
    """What the server is doing right now. Read by /status, written by the job."""

    active: bool = False
    kind: str | None = None          # "generate" | "edit"
    phase: str = "idle"              # idle | loading | encoding | denoising | decoding | saving
    stem: str | None = None
    model: str | None = None
    step: int = 0
    total_steps: int = 0
    started_at: float | None = None
    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0


@dataclass
class Monitor:
    """Progress state, memory-pressure alerts and the last OOM, for one app instance."""

    vram_warn_percent: float = 92
    ram_warn_percent: float = 90
    progress: Progress = field(default_factory=Progress)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _alerting: dict = field(default_factory=lambda: {"vram": False, "ram": False}, repr=False)
    last_oom: dict | None = None

    # --- job lifecycle ---
    def begin(self, kind: str, stem: str, model: str) -> None:
        with self._lock:
            self.progress = Progress(active=True, kind=kind, phase="loading", stem=stem,
                                     model=model, started_at=time.time())

    def set_phase(self, phase: str, total_steps: int | None = None) -> None:
        with self._lock:
            self.progress.phase = phase
            if total_steps is not None:
                self.progress.total_steps = int(total_steps)
                self.progress.step = 0

    def on_step(self) -> None:
        """Called once per denoising step (the engine wraps scheduler.step)."""
        gpu, ram = fast_gpu_stats(), get_ram_stats()
        with self._lock:
            self.progress.step += 1
            if self.progress.total_steps and self.progress.step >= self.progress.total_steps:
                self.progress.phase = "decoding"
            for stats, key in ((gpu, "peak_vram_gb"), (ram, "peak_ram_gb")):
                if stats and "error" not in stats:
                    setattr(self.progress, key, max(getattr(self.progress, key), stats["used_gb"]))
        self.log_pressure_transitions(self.evaluate_pressure(gpu, ram))

    def end(self) -> None:
        with self._lock:
            self.progress.active = False
            self.progress.phase = "idle"

    def snapshot(self) -> Progress:
        with self._lock:
            return Progress(**vars(self.progress))

    # --- memory pressure ---
    @staticmethod
    def _pct(stats: dict | None) -> float | None:
        if not stats or "error" in stats or not stats.get("total_gb"):
            return None
        return stats["used_gb"] / stats["total_gb"] * 100

    def evaluate_pressure(self, gpu: dict | None, ram: dict | None) -> list[dict]:
        warnings = []
        pct = self._pct(gpu)
        if pct is not None and pct >= self.vram_warn_percent:
            warnings.append({"kind": "vram", "message":
                             f"VRAM at {pct:.0f}% ({gpu['used_gb']}/{gpu['total_gb']} GB) — out-of-memory risk"})
        pct = self._pct(ram)
        if pct is not None and pct >= self.ram_warn_percent:
            warnings.append({"kind": "ram", "message":
                             f"System RAM at {pct:.0f}% ({ram['used_gb']}/{ram['total_gb']} GB) — out-of-memory risk"})
        return warnings

    def log_pressure_transitions(self, warnings: list[dict]) -> None:
        """Log the first time a threshold is crossed; re-arm once it drops back."""
        active = {w["kind"]: w["message"] for w in warnings}
        for kind in ("vram", "ram"):
            if kind in active and not self._alerting[kind]:
                logger.warning(f"MEMORY WARNING: {active[kind]}")
                self._alerting[kind] = True
            elif kind not in active:
                self._alerting[kind] = False

    def active_warnings(self, gpu: dict, ram: dict) -> list[str]:
        """Warnings for /status: live pressure plus the most recent unresolved OOM."""
        pressure = self.evaluate_pressure(gpu, ram)
        self.log_pressure_transitions(pressure)
        out = [w["message"] for w in pressure]
        if self.last_oom:
            o = self.last_oom
            out.insert(0, f"Out of memory at {o['time']} during {o['stage']}: {o['detail']}")
        return out

    # --- OOM ---
    def report_oom(self, exc: BaseException, stage: str, memory_mode: str) -> None:
        """Log a loud, actionable OOM message with a memory snapshot; keep it for /status."""
        gpu, ram = get_gpu_stats(), get_ram_stats()
        detail = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:200]
        self.last_oom = {"time": time.strftime("%H:%M:%S"), "stage": stage, "detail": detail}
        p = self.snapshot()
        logger.error(f"OUT OF MEMORY during {stage}: {detail}")
        if "error" not in gpu:
            logger.error(f"  VRAM at failure: {gpu['used_gb']}/{gpu['total_gb']} GB")
        if "error" not in ram:
            logger.error(f"  RAM at failure: {ram['used_gb']}/{ram['total_gb']} GB")
        logger.error(f"  Peak during this run: VRAM {p.peak_vram_gb} GB, RAM {p.peak_ram_gb} GB")
        if memory_mode == "cpu":
            logger.error("  Hint: lower the resolution, or set memory.quantize_transformer: true.")
        else:
            logger.error(
                "  Hint: lower the resolution. If memory.vae_fp32 is on, turn it off. "
                "Or set memory.vae_on_cpu: true to run the VAE in system RAM, "
                "lower memory.gpu_max_memory, or set memory.run_on_ram: true (slow)."
            )
        if stage.startswith("edit"):
            logger.error("  Edits: every reference image adds to the sequence. Use fewer images, "
                         "or lower edit.resolution in the config.")

    def clear_oom(self) -> None:
        self.last_oom = None
