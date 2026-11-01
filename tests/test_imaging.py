"""Pillow helpers: edit sizing, fitting, decoding guards, log parsing."""
import io

import pytest
from PIL import Image

from server import imaging
from server.storage import check_stem, parse_log, stem_from_name, InvalidName


@pytest.mark.parametrize("src, resolution, cap, expected", [
    ((200, 150), 256, 10**9, (288, 224)),
    ((1000, 1000), 1024, 10**9, (1024, 1024)),
    ((4000, 3000), 1024, 10**9, (1184, 896)),
    ((3000, 4000), 1024, 10**9, (896, 1184)),
    ((1920, 1080), 1024, 10**9, (1376, 768)),
    ((64, 64), 1024, 512 * 512, (512, 512)),     # capped by max_pixels
])
def test_auto_edit_size(src, resolution, cap, expected):
    w, h = imaging.auto_edit_size(*src, resolution, cap)
    assert (w, h) == expected
    assert w % 32 == 0 and h % 32 == 0 and w * h <= cap


@pytest.mark.parametrize("src", [(200, 150), (1920, 1080), (1080, 1920), (777, 333), (64, 64), (3000, 1000)])
@pytest.mark.parametrize("resolution", [512, 1024, 1280])
def test_auto_edit_size_matches_the_pipeline(src, resolution):
    """Without a cap, our size is exactly what QwenImage21Pipeline would pick itself."""
    pipeline_module = pytest.importorskip("diffusers.pipelines.qwenimage21.pipeline_qwenimage21")
    w, h, _ = pipeline_module.calculate_dimensions(resolution * resolution, src[0] / src[1])
    assert imaging.auto_edit_size(*src, resolution, 10**9) == (w, h)


def test_shrink_reference_keeps_aspect_and_small_images():
    small = Image.new("RGB", (800, 600))
    assert imaging.shrink_reference(small) is small
    big = imaging.shrink_reference(Image.new("RGBA", (4000, 1000)))
    assert big.size == (2048, 512) and big.mode == "RGBA"


def test_open_image_keeps_alpha_and_flatten_whitens_it():
    buf = io.BytesIO()
    Image.new("RGBA", (4, 4), (255, 0, 0, 0)).save(buf, format="PNG")
    img = imaging.open_image(buf.getvalue(), 10**6)
    assert img.mode == "RGBA"
    assert imaging.flatten(img).getpixel((0, 0)) == (255, 255, 255)
    buf = io.BytesIO()
    Image.new("L", (4, 4), 128).save(buf, format="PNG")
    assert imaging.open_image(buf.getvalue(), 10**6).mode == "RGB"


def test_open_image_rejects_huge_dimensions():
    # A tiny file that claims to be huge: rejected from the header, before decoding.
    big = Image.new("1", (8000, 7000))
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    with pytest.raises(imaging.ImageInputError, match="pixels"):
        imaging.open_image(buf.getvalue(), max_bytes=10**8)


def test_open_image_errors():
    with pytest.raises(imaging.ImageInputError, match="empty"):
        imaging.open_image(b"", 100)
    with pytest.raises(imaging.ImageInputError, match="MB limit"):
        imaging.open_image(b"x" * 200, 100)


def test_decode_base64_variants():
    assert imaging.decode_base64_image("aGVsbG8=") == b"hello"
    assert imaging.decode_base64_image("data:image/webp;base64,aGVsbG8=") == b"hello"
    with pytest.raises(imaging.ImageInputError):
        imaging.decode_base64_image("not base64!")


@pytest.mark.parametrize("bad", ["", ".hidden", "a/b", "a\\b", "..", "x\x00", " lead", "x" * 201])
def test_check_stem_rejects(bad):
    with pytest.raises(InvalidName):
        check_stem(bad)


def test_stem_from_name():
    assert stem_from_name("output/1_ab.png") == "1_ab"
    assert stem_from_name("1_ab.PNG") == "1_ab"
    assert stem_from_name("1_ab") == "1_ab"


def test_parse_log_old_format_without_mode():
    text = ("time: 2026-10-01 10:00:00\nstatus: success\nmodel: base (Qwen-Image-2.1)\nsize: 1024x1024\n\n"
            "prompt:\na red fox\n\nnegative_prompt:\n(none)\n\nserver logs during this generation:\n"
            "10:00:00 INFO Generating\n")
    rec = parse_log(text)
    assert rec.fields["status"] == "success" and "mode" not in rec.fields
    assert rec.prompt == "a red fox" and rec.negative_prompt == ""
