"""Everything under output/: images, per-job logs, edit sources, gallery thumbnails.

Layout (output/ is bind-mounted from the host):

    output/<stem>.png            the generated or edited image
    output/logs/<stem>.log       prompt, parameters, timing, peak memory, server lines
    output/sources/<stem>_<n>.png  the reference images of an edit, n = 1, 2, ...
    output/.thumbs/<stem>.jpg    gallery thumbnail cache (safe to delete)

A "stem" is the shared file name without extension. New stems look like
`<unix time>_<8 hex chars>`; the gallery also lists older or hand-copied PNGs.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .imaging import png_bytes, thumbnail_jpeg
from .monitoring import logger

MAX_STEM_LENGTH = 200
LOG_HEADER_READ_BYTES = 16_384


class InvalidName(ValueError):
    pass


def check_stem(stem: str) -> str:
    """Reject anything that could escape output/ or address a hidden/cache file."""
    if (not stem or len(stem) > MAX_STEM_LENGTH or stem.startswith(".")
            or any(c in stem for c in "/\\\x00") or stem != stem.strip()):
        raise InvalidName(f"'{stem}' is not a valid image name.")
    return stem


def stem_from_name(name: str) -> str:
    """Accept '1730000000_ab12cd34', '..._ab12cd34.png' or 'output/..._ab12cd34.png'."""
    name = name.strip()
    if name.startswith("output/"):
        name = name[len("output/"):]
    if name.lower().endswith(".png"):
        name = name[:-4]
    return check_stem(name)


@dataclass(frozen=True)
class LogRecord:
    fields: dict[str, str]
    prompt: str
    negative_prompt: str
    text: str


def parse_log(text: str) -> LogRecord:
    """Parse a per-job .log file (format written by OutputStore.write_log)."""
    fields: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].strip():
        key, sep, value = lines[i].partition(": ")
        if sep:
            fields[key.strip()] = value.strip()
        i += 1
    body = "\n" + "\n".join(lines[i:])
    prompt = negative = ""
    start = body.find("\nprompt:\n")
    if start != -1:
        rest = body[start + len("\nprompt:\n"):]
        # Written order: prompt, negative-prompt marker, negative prompt, server-log
        # section. The prompt may itself contain marker-like lines, so take the *last*
        # negative marker, and look for the log section only after it.
        neg_marker = "\n\nnegative_prompt:\n"
        neg_at = rest.rfind(neg_marker)
        if neg_at == -1:
            prompt = rest.split("\nserver logs during this generation:\n", 1)[0].strip()
        else:
            prompt = rest[:neg_at].strip()
            tail = rest[neg_at + len(neg_marker):]
            negative = tail.split("\nserver logs during this generation:\n", 1)[0].strip()
            if negative == "(none)":
                negative = ""
    return LogRecord(fields=fields, prompt=prompt, negative_prompt=negative, text=text)


class OutputStore:
    def __init__(self, output_dir: str | os.PathLike):
        self.root = Path(output_dir)
        self.logs = self.root / "logs"
        self.sources = self.root / "sources"
        self.thumbs = self.root / ".thumbs"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.logs, self.sources):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(f"Could not create {d} ({e}). Check the folder's owner (see docs/PITFALLS.md).")

    @staticmethod
    def new_stem() -> str:
        return f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------ paths
    def image_path(self, stem: str) -> Path:
        return self.root / f"{check_stem(stem)}.png"

    def log_path(self, stem: str) -> Path:
        return self.logs / f"{check_stem(stem)}.log"

    def source_path(self, stem: str, n: int) -> Path:
        """Reference image n (1-based) of an edit."""
        return self.sources / f"{check_stem(stem)}_{int(n)}.png"

    def source_count(self, stem: str) -> int:
        n = 0
        while self.source_path(stem, n + 1).is_file():
            n += 1
        return n

    def exists(self, stem: str) -> bool:
        return self.image_path(stem).is_file()

    # ------------------------------------------------------------ writing
    def save_image(self, stem: str, data: bytes) -> bool:
        """Write the PNG. A disk/permission problem is logged, not raised."""
        path = self.image_path(stem)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            logger.info(f"Saved output image to {path}")
            return True
        except OSError as e:
            logger.warning(f"Could not save the image to {self.root} ({e}). Returning it without saving.")
            return False

    def save_sources(self, stem: str, images: list[Image.Image]) -> bool:
        """Keep an edit's reference images (non-fatal if the disk refuses)."""
        try:
            self.sources.mkdir(parents=True, exist_ok=True)
            for n, image in enumerate(images, start=1):
                self.source_path(stem, n).write_bytes(png_bytes(image))
            return True
        except OSError as e:
            logger.warning(f"Could not save the edit's reference images ({e}).")
            return False

    def write_log(self, stem: str, fields: dict, prompt: str, negative: str,
                  log_lines: list[str]) -> None:
        """output/logs/<stem>.log: header fields, prompt, negative prompt, server lines."""
        lines = [f"{k}: {v}" for k, v in fields.items()]
        lines += ["", "prompt:", prompt.strip(), "", "negative_prompt:", negative.strip() or "(none)", ""]
        if log_lines:
            lines.append("server logs during this generation:")
            lines.extend(log_lines)
        try:
            self.logs.mkdir(parents=True, exist_ok=True)
            self.log_path(stem).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not write the generation log to {self.logs} ({e}).")

    # ------------------------------------------------------------ reading
    def read_log(self, stem: str) -> LogRecord | None:
        path = self.log_path(stem)
        try:
            return parse_log(path.read_text(encoding="utf-8", errors="replace"))
        except FileNotFoundError:
            return None

    def _log_header(self, stem: str) -> dict[str, str]:
        """Just the key: value header of a log (fast, for listings)."""
        try:
            with open(self.log_path(stem), encoding="utf-8", errors="replace") as f:
                head = f.read(LOG_HEADER_READ_BYTES)
        except OSError:
            return {}
        record = parse_log(head)
        out = dict(record.fields)
        if record.prompt:
            out["prompt"] = record.prompt[:300]
        return out

    def list_images(self, offset: int = 0, limit: int = 60) -> tuple[list[dict], int]:
        """Newest first. Returns (page, total)."""
        entries = []
        try:
            with os.scandir(self.root) as it:
                for e in it:
                    if not e.is_file() or not e.name.lower().endswith(".png"):
                        continue
                    try:
                        stem = check_stem(e.name[:-4])
                    except InvalidName:
                        continue
                    entries.append((e.stat().st_mtime, stem, e.stat().st_size))
        except FileNotFoundError:
            return [], 0
        entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
        page = []
        for mtime, stem, size in entries[offset:offset + limit]:
            header = self._log_header(stem)
            page.append({
                "id": stem,
                "file": f"{stem}.png",
                "created": int(mtime),
                "bytes": size,
                "has_log": bool(header) or self.log_path(stem).is_file(),
                "sources": self.source_count(stem),
                "kind": header.get("mode", "generate" if header else None),
                "status": header.get("status"),
                "size": header.get("size"),
                "model": header.get("model"),
                "seed": header.get("seed"),
                "prompt": header.get("prompt"),
            })
        return page, len(entries)

    def thumbnail(self, stem: str, max_side: int = 384) -> bytes:
        """JPEG thumbnail bytes, cached in output/.thumbs and rebuilt when the PNG is newer.

        Caching is best effort: if output/ isn't writable the thumbnail is still served.
        """
        src = self.image_path(stem)
        thumb = self.thumbs / f"{stem}.jpg"
        try:
            if thumb.stat().st_mtime >= src.stat().st_mtime:
                return thumb.read_bytes()
        except OSError:
            pass
        data = thumbnail_jpeg(src, max_side=max_side)
        try:
            self.thumbs.mkdir(parents=True, exist_ok=True)
            tmp = self.thumbs / f".{stem}.tmp"
            tmp.write_bytes(data)
            tmp.replace(thumb)
        except OSError:
            pass
        return data
