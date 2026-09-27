# Command reference

## Docker — build & run

```bash
# Rebuild — required after ANY change to server/*.py or Dockerfile.
# `restart` alone does NOT pick up code changes (COPY bakes files in at build time).
docker compose build

# Bring up everything (after a full `down`, always omit the service name
# so every service — including qwen-image-ui — comes back, not just one)
docker compose up -d --build

# Bring up one service only (only safe if siblings are already running)
docker compose up -d --build qwen-image-api
```

## Docker — stop, restart, remove

```bash
# Full teardown — use after a crash/OOM to guarantee a clean GPU/allocator state
docker compose down

# Restart existing container(s) — no rebuild, less reliable than down+up for
# clearing stuck VRAM allocations after a crash
docker compose restart

docker compose stop
docker compose start
```

## Docker — status & inspection

```bash
docker compose ps
docker stats qwen-image-api                 # live CPU/RAM/network
docker exec -it qwen-image-api bash         # shell in
docker exec qwen-image-api grep -A5 "CORSMiddleware" /app/main.py
docker inspect qwen-image-api               # mounts, env, devices
```

## Docker — logs

```bash
docker compose logs -f qwen-image-api       # follow live
docker logs -f qwen-image-api               # same, without compose context
docker logs --tail 100 qwen-image-api
docker logs -f -t qwen-image-api            # with timestamps
```

Also available: the app's own `/status` endpoint
(`curl http://localhost:8000/status`) returns recent in-app log lines plus
live GPU/RAM stats and generation progress — useful when you want structured
data rather than raw container logs.

## GPU (ROCm)

```bash
rocm-smi --showmeminfo vram                      # quick VRAM summary
watch -n 1 rocm-smi --showmeminfo vram           # live view
rocm-smi                                          # full status: temp, power, clocks

# Sanity check the GPU is visible at all, outside this project's containers
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  rocm/pytorch:latest rocm-smi
```

## System RAM

```bash
free -h
watch -n 1 free -h
ps aux --sort=-%mem | head -15
sudo dmesg | grep -i "killed process"            # confirm OOM-killer activity
sudo journalctl -k -b | grep -i oom
```

## Debugging sequence (this project, in order)

```bash
docker compose logs --tail 200 qwen-image-api    # 1. what happened
docker compose down && docker compose up -d --build   # 2. clean restart
watch -n 1 rocm-smi --showmeminfo vram           # 3a. watch VRAM
watch -n 1 free -h                                # 3b. watch RAM
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox sitting in a snowy forest", "size": "1024x1024"}' \
  -o response.json                                # 4. test request
```

## Git

```bash
git status                          # check before every commit
git add .
git commit -m "describe what changed"
git push

git log --oneline                   # quick history
git diff                            # unstaged changes
git diff --staged                   # staged changes not yet committed
```

First-time setup on a new machine:

```bash
sudo pacman -S git github-cli
gh auth login                       # browser-based device flow
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
