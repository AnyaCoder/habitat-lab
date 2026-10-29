#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=x11
export SDL_RENDER_DRIVER=software
export SDL_FRAMEBUFFER_ACCELERATION=0

exec "$repo_root/.conda/bin/python" "$repo_root/examples/go2_xpra.py"
