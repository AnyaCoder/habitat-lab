#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Launch the Go2 split-screen view inside an existing Xpra display."""

import runpy
import sys
from pathlib import Path

import pygame


def main():
    project_root = Path(__file__).resolve().parents[1]

    # SDL must initialize before habitat_sim loads the GL/X11 libraries.
    # Initializing it later can segfault when the process runs inside Xpra.
    pygame.init()
    sys.argv = [
        str(project_root / "examples" / "interactive_play.py"),
        "--cfg",
        "benchmark/rearrange/play/play_go2.yaml",
        "--disable-inverse-kinematics",
        "--go2-game",
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
