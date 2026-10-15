# Model setup

`models/` is git-ignored (large, re-downloadable) — this is how to populate
it from scratch on a new machine. Both containers mount the same `./models`
folder, so each model is downloaded once.

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

## Download Turbo (for `qwen-image-turbo-api`)

```bash
huggingface-cli download Qwen/Qwen-Image-2.1-Turbo \
  --local-dir ./models/Qwen-Image-2.1-Turbo
```

(Newer `huggingface_hub` versions also provide `hf download` with the same
arguments.) Turbo is a complete pipeline folder with its own text encoder and
VAE, not a delta on top of the base model, so it takes a similar amount of disk
(check with `du -sh models/*`). At the time of release it was published under
the Qwen Research License (non-commercial research use): check the model page.

Expected layout (`/app/models/Qwen-Image-2.1-Turbo` inside the container):

```
Qwen-Image-2.1-Turbo/
├── model_index.json
├── processor/            tokenizer + chat template
├── scheduler/            scheduler_config.json
├── text_encoder/         model.safetensors, config.json
├── transformer/          2 safetensors shards + index
├── vae/
└── assets/, README.md, LICENSE   (not used by the server)
```

Turbo's 8-step sampling schedule is stored in the checkpoint itself. To see
whether your copy carries it:

```bash
grep -n sigma models/Qwen-Image-2.1-Turbo/model_index.json \
              models/Qwen-Image-2.1-Turbo/scheduler/scheduler_config.json
```

Using it requires a recent diffusers and `transformers>=5.17`, which is why
Turbo has its own `Dockerfile.turbo`. See `docs/PITFALLS.md`.

The paths the containers look for are set in the config files
(`models.<id>.path`). If a model's folder is missing, the UI shows the model as
"Not downloaded" and the API answers `400`.

## Do not use pre-quantized community checkpoints as a drop-in replacement

We tried this (`Rin247/Qwen-Image-2.1-INT8`) and it silently loaded
incorrectly — see `docs/PITFALLS.md` for why. Quantization in this project
happens on-the-fly at load time instead, via `torchao`, against the base
model above — no separate download needed for that. See `server/pipeline.py`.
(Turbo runs in plain bf16 by default.)

## MIOpen kernel cache (not a model download, but also populated locally)

The `cache/miopen/` directory (mounted via `docker-compose.yml`, and shared by
both API containers) holds tuned GPU kernel selections specific to your exact
GPU architecture (`gfx1201` for the R9700). It's git-ignored because it's
meaningless on different hardware and regenerates automatically:

- First run after a fresh `cache/` directory will be noticeably slower —
  MIOpen is benchmarking kernel variants and writing the results.
- Subsequent runs read from the populated cache and are faster.
- If you switch to a different GPU, delete `cache/miopen/` and let it
  re-populate; a stale cache from a different architecture won't apply.
- It is only used on the GPU path. With `memory.run_on_ram: true` the files
  are simply not touched (keep them for when you switch back).
