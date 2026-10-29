#!/usr/bin/env python3

import runpy
import sys
from pathlib import Path

import pygame


project_root = Path(__file__).resolve().parents[1]
pygame.init()
sys.argv = [
    str(project_root / "examples" / "interactive_play.py"),
    "--cfg",
    "benchmark/rearrange/play/play_go2.yaml",
    "--disable-inverse-kinematics",
    "--go2-game",
    "habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.width=64",
    "habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.height=64",
    "habitat.simulator.agents.main_agent.sim_sensors.head_depth_sensor.width=64",
    "habitat.simulator.agents.main_agent.sim_sensors.head_depth_sensor.height=64",
]
runpy.run_path(
    str(project_root / "examples" / "interactive_play.py"),
    run_name="__main__",
)
