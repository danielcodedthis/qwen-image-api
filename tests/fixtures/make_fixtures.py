"""Regenerate the synthetic test images in this folder (they are checked in).

    python tests/fixtures/make_fixtures.py

Everything is drawn from code: no real photos, no model output.
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent


def gradient(w, h):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (int(255 * x / max(1, w - 1)), int(255 * y / max(1, h - 1)), 128)
    return img


def main():
    # Landscape source for edits (not a multiple of 16 on purpose: 200x150).
    img = gradient(200, 150)
    d = ImageDraw.Draw(img)
    d.ellipse((60, 35, 140, 115), fill=(240, 240, 240), outline=(20, 20, 20), width=3)
    img.save(HERE / "landscape_200x150.png")

    # Square source, already a multiple of 16.
    gradient(64, 64).save(HERE / "square_64.png")

    # Transparent PNG: transparency must end up white, not black.
    rgba = Image.new("RGBA", (48, 32), (255, 0, 0, 0))
    ImageDraw.Draw(rgba).rectangle((8, 8, 39, 23), fill=(0, 0, 255, 255))
    rgba.save(HERE / "transparent_48x32.png")

    # JPEG with EXIF orientation 6 (stored 40x20, displays as 20x40 portrait).
    jpg = gradient(40, 20)
    exif = Image.Exif()
    exif[0x0112] = 6
    jpg.save(HERE / "exif_rotate90_40x20.jpg", quality=90, exif=exif.tobytes())

    # Not an image at all.
    (HERE / "not_an_image.png").write_bytes(b"this is plain text pretending to be a PNG\n")


if __name__ == "__main__":
    main()
