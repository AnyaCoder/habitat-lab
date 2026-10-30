#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${HABITAT_PYTHON:-$repo_root/.conda/bin/python}"
if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set; run this script inside Xpra or an X11 session." >&2
  exit 1
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Habitat Python not found: $python_bin" >&2
  echo "Set HABITAT_PYTHON to a Python executable with Habitat installed." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=x11
export SDL_RENDER_DRIVER=software
export SDL_FRAMEBUFFER_ACCELERATION=0

exec "$python_bin" "$repo_root/examples/go2_xpra.py"
