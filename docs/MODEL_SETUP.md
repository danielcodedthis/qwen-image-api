# Model setup

`models/` is git-ignored (large, re-downloadable) — this is how to populate
it from scratch on a new machine.

## Requirements

```bash
pip install --user -U huggingface_hub
```

## Download the base model

```bash
huggingface-cli download Qwen/Qwen-Image-2.1 \
  --local-dir ./models/Qwen-Image-2.1
```

Check the model's license terms on Hugging Face before downloading — some
Qwen releases are gated behind a license acceptance click on the model page.
If the download fails with a 401/403, that's why: run
`huggingface-cli login`, accept the license in a browser, then retry.

Expect roughly 30GB on disk in bf16.

## Do not use pre-quantized community checkpoints as a drop-in replacement

We tried this (`Rin247/Qwen-Image-2.1-INT8`) and it silently loaded
incorrectly — see `docs/PITFALLS.md` for why. Quantization in this project
happens on-the-fly at load time instead, via `torchao`, against the base
model above — no separate download needed for that. See `server/pipeline.py`.

## MIOpen kernel cache (not a model download, but also populated locally)

The `cache/miopen/` directory (mounted via `docker-compose.yml`) holds tuned
GPU kernel selections specific to your exact GPU architecture (`gfx1201` for
the R9700). It's git-ignored because it's meaningless on different hardware
and regenerates automatically:

- First run after a fresh `cache/` directory will be noticeably slower —
  MIOpen is benchmarking kernel variants and writing the results.
- Subsequent runs read from the populated cache and are faster.
- If you switch to a different GPU, delete `cache/miopen/` and let it
  re-populate; a stale cache from a different architecture won't apply.
