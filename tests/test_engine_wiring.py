"""QwenEngine (server/engine/qwen.py) on CPU, without the model.

- The engine is driven with a stand-in pipeline that has QwenImage21Pipeline.__call__'s
  signature, to check what it passes (references, sizes, seed, CFG) and how it counts steps.
- `place_vae` (vae_on_cpu / vae_fp32) is checked with a tiny random
  AutoencoderKLQwenImage21 through the REAL pipeline's `_encode_vae_image`, the code path
  every edit takes, and through decode.
Says nothing about image quality (see docs/SMOKE_TEST.md). Skipped without torch/diffusers.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from PIL import Image  # noqa: E402

from server.engine import RenderJob  # noqa: E402
from server.engine.qwen import QwenEngine, place_vae  # noqa: E402
from server.settings import load_settings  # noqa: E402


class FakeQwenPipe:
    def __init__(self, sample_sigmas=None):
        self.config = SimpleNamespace(sample_sigmas=sample_sigmas)
        self.calls = []

    def __call__(self, prompt=None, image=None, negative_prompt=None, true_cfg_scale=1.0, height=None,
                 width=None, num_inference_steps=40, sigmas=None, generator=None,
                 callback_on_step_end=None, output_resolution=1024, use_kv_cache=True):
        steps = len(self.config.sample_sigmas) if self.config.sample_sigmas else num_inference_steps
        self.calls.append(dict(prompt=prompt, image=image, negative=negative_prompt, cfg=true_cfg_scale,
                               size=(width, height), steps=num_inference_steps,
                               resolution=output_resolution, seed_draw=torch.rand(1, generator=generator).item()))
        for i in range(steps):
            callback_on_step_end(self, i, 1000 - i, {})
        return SimpleNamespace(images=[Image.new("RGB", (width, height))])


class Hooks:
    def __init__(self):
        self.phases, self.steps = [], 0

    def set_phase(self, phase, total_steps=None):
        self.phases.append((phase, total_steps))

    def on_step(self):
        self.steps += 1


def make_engine(tmp_env, sample_sigmas=None, model="base"):
    eng = QwenEngine(load_settings(tmp_env["config"], tmp_env["output"]))
    eng._pipe, eng._loaded_name = FakeQwenPipe(sample_sigmas), model  # skip from_pretrained
    return eng


def job(**kw):
    base = dict(model="base", prompt="p", negative_prompt="blurry", width=512, height=256, steps=6,
                true_cfg_scale=1.0, seed=5)
    return RenderJob(**{**base, **kw})


def test_text_to_image_passes_settings_and_counts_steps(tmp_env):
    eng = make_engine(tmp_env)
    hooks = Hooks()
    res = eng.text_to_image(job(), hooks)
    call = eng._pipe.calls[-1]
    assert res.steps_ran == hooks.steps == res.steps_planned == 6
    assert call["size"] == (512, 256) and call["steps"] == 6 and call["image"] is None
    assert call["negative"] is None and call["cfg"] == 1.0          # CFG off: no negative prompt sent
    assert hooks.phases == [("loading", None), ("denoising", 6)]
    assert res.details == {"cfg": "off"}


def test_cfg_on_sends_negative_prompt(tmp_env):
    eng = make_engine(tmp_env)
    eng.text_to_image(job(true_cfg_scale=2.0, negative_prompt=""), Hooks())
    call = eng._pipe.calls[-1]
    assert call["cfg"] == 2.0 and call["negative"] == " "           # " " = the standard empty negative


def test_seed_is_deterministic(tmp_env):
    eng = make_engine(tmp_env)
    eng.text_to_image(job(seed=11), Hooks())
    eng.text_to_image(job(seed=11), Hooks())
    eng.text_to_image(job(seed=12), Hooks())
    draws = [c["seed_draw"] for c in eng._pipe.calls]
    assert draws[0] == draws[1] != draws[2]


def test_saved_schedule_sets_planned_steps(tmp_env):
    eng = make_engine(tmp_env, sample_sigmas=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], model="turbo")
    hooks = Hooks()
    res = eng.text_to_image(job(model="turbo", steps=6), hooks)
    assert res.steps_ran == res.steps_planned == 8 and ("denoising", 8) in hooks.phases


def test_edit_passes_references_in_order_and_resolution(tmp_env):
    eng = make_engine(tmp_env)
    refs = [Image.new("RGBA", (300, 200)), Image.new("RGB", (64, 64))]
    res = eng.edit(job(), refs, 768, Hooks())
    call = eng._pipe.calls[-1]
    assert [im.size for im in call["image"]] == [(300, 200), (64, 64)]
    assert call["resolution"] == 768 and call["size"] == (512, 256)
    assert res.details["reference_images"] == 2


# ---------------------------------------------------------------- VAE placement
@pytest.fixture
def tiny_vae():
    from diffusers import AutoencoderKLQwenImage21
    torch.manual_seed(0)
    return AutoencoderKLQwenImage21(base_dim=8, decoder_base_dim=8, num_res_blocks=1).eval()


def encode_like_the_pipeline(vae, image):
    """Run QwenImage21Pipeline._encode_vae_image (the real code) with this VAE."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    fake_self = SimpleNamespace(vae=vae, latent_channels=vae.config.z_dim)
    return QwenImage21Pipeline._encode_vae_image(fake_self, image, None)


@pytest.mark.parametrize("device", ["cpu", None])  # vae_on_cpu, vae_fp32
def test_place_vae_encode_converts_in_and_out(tiny_vae, device):
    pipe = SimpleNamespace(vae=tiny_vae.to(torch.bfloat16))
    place_vae(pipe, device)
    assert next(tiny_vae.parameters()).dtype == torch.float32
    seen = {}
    real_encoder = tiny_vae._encode

    def spy(x):
        seen["dtype"] = x.dtype
        return real_encoder(x)
    tiny_vae._encode = spy

    image = torch.rand(1, 4, 1, 64, 64, dtype=torch.bfloat16) * 2 - 1   # what prepare_latents passes
    with torch.no_grad():
        latents = encode_like_the_pipeline(tiny_vae, image)
    assert seen["dtype"] == torch.float32                  # the VAE ran in float32
    assert latents.dtype == torch.bfloat16                 # handed back in the caller's dtype
    assert latents.device == image.device
    assert tuple(latents.shape) == (1, 64, 1, 4, 4)


def test_place_vae_decode_returns_float32_image(tiny_vae):
    pipe = SimpleNamespace(vae=tiny_vae.to(torch.bfloat16))
    place_vae(pipe, "cpu")
    z = torch.randn(1, 64, 1, 4, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        out = tiny_vae.decode(z, return_dict=False)[0]
    assert out.dtype == torch.float32 and tuple(out.shape)[-2:] == (64, 64)


def test_place_vae_return_dict_false(tiny_vae):
    pipe = SimpleNamespace(vae=tiny_vae.to(torch.bfloat16))
    place_vae(pipe, "cpu")
    x = torch.rand(1, 4, 1, 32, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        (dist,) = tiny_vae.encode(x, return_dict=False)
    assert dist.mode().dtype == torch.bfloat16
