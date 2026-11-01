"""Pillow helpers: decoding uploads safely, sizing edits, encoding PNGs, thumbnails."""
from __future__ import annotations

import base64
import binascii
import io
import math
import re
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .settings import SIZE_MULTIPLE

# Hard ceiling for a decoded source image (a 50 MP photo is fine, a decompression
# bomb is not). Pillow's own guard is set to match.
MAX_SOURCE_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS

# Reference images bigger than this (longest side) are scaled down before use and
# storage. The pipeline resizes them to about edit.resolution² anyway.
MAX_REFERENCE_SIDE = 2048

_DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.ASCII)


class ImageInputError(ValueError):
    """The uploaded image can't be used (bad encoding, format, size...)."""


def decode_base64_image(value: str) -> bytes:
    """Accept raw base64 or a data URL (`data:image/png;base64,...`)."""
    text = _DATA_URL.sub("", value.strip(), count=1)
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageInputError(f"image is not valid base64 ({e}).") from None


def open_image(data: bytes, max_bytes: int) -> Image.Image:
    """Bytes -> RGB or RGBA PIL image, EXIF rotation applied.

    Transparency is kept: Qwen-Image-2.1 edits transparent images natively (the
    pipeline converts every reference to RGBA itself).
    """
    if not data:
        raise ImageInputError("image is empty.")
    if len(data) > max_bytes:
        raise ImageInputError(
            f"image is {len(data) / 1024**2:.1f} MB, over the {max_bytes / 1024**2:.0f} MB limit "
            "(edit.max_upload_mb in the config)."
        )
    try:
        with warnings.catch_warnings():
            # Pillow only *warns* between 1x and 2x its limit; treat that as an error too.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_SOURCE_PIXELS:
            raise ImageInputError(f"image is {img.width}x{img.height}, over {MAX_SOURCE_PIXELS:,} pixels.")
        img.load()
    except ImageInputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageInputError(f"image is over {MAX_SOURCE_PIXELS:,} pixels.") from None
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ImageInputError(f"could not read the image ({type(e).__name__}). "
                              "Use PNG, JPEG or WebP.") from None
    img = ImageOps.exif_transpose(img)
    return normalize_mode(img)


def has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def normalize_mode(img: Image.Image) -> Image.Image:
    """RGBA if the image has transparency, else RGB."""
    if img.mode in ("RGB", "RGBA"):
        return img
    return img.convert("RGBA" if has_alpha(img) else "RGB")


def flatten(img: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """RGB copy, transparency composited on a plain background (for previews)."""
    if not has_alpha(img):
        return img.convert("RGB")
    rgba = img.convert("RGBA")
    out = Image.new("RGB", rgba.size, background)
    out.paste(rgba, mask=rgba.getchannel("A"))
    return out


def shrink_reference(img: Image.Image, max_side: int = MAX_REFERENCE_SIDE) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def auto_edit_size(src_w: int, src_h: int, resolution: int, max_pixels: int) -> tuple[int, int]:
    """Output size for an edit with no explicit size.

    Same rule as QwenImage21Pipeline when height/width are omitted (calculate_dimensions):
    the last reference image's aspect ratio at resolution² pixels, sides rounded to 32.
    Then capped by max_pixels, shrinking the longer side.
    """
    ratio = src_w / src_h
    w = math.sqrt(resolution * resolution * ratio)
    h = w / ratio
    w = max(SIZE_MULTIPLE, round(w / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    h = max(SIZE_MULTIPLE, round(h / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    while w * h > max_pixels and max(w, h) > SIZE_MULTIPLE:
        if w >= h:
            w -= SIZE_MULTIPLE
        else:
            h -= SIZE_MULTIPLE
    return int(w), int(h)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def thumbnail_jpeg(path, max_side: int = 384, quality: int = 82) -> bytes:
    with Image.open(path) as im:
        im = flatten(ImageOps.exif_transpose(im))
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
