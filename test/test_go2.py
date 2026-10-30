#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from pathlib import Path

import habitat
import numpy as np
import pytest

from habitat.config.default import get_agent_config, get_config
from habitat.tasks.rearrange.actions.actions import Go2BaseVelAction


def _load_interactive_play():
    path = Path(__file__).resolve().parents[1] / "examples/interactive_play.py"
    spec = importlib.util.spec_from_file_location("interactive_play", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_go2_action(*, allow_back=True):
    action = object.__new__(Go2BaseVelAction)
    action._lin_speed = 0.5
    action._ang_speed = 1.0
    action._allow_back = allow_back
    action._velocity_response_time = 0.18
    action._velocity_epsilon = 1e-4
    return action


def test_go2_config_uses_kinematic_physics_and_action():
    config = get_config("benchmark/rearrange/play/play_go2.yaml")

    assert config.habitat.simulator.step_physics is False
    assert (
        config.habitat.simulator.agents.main_agent.articulated_agent_type
        == "Go2Robot"
    )
    assert config.habitat.task.actions.base_velocity.type == "Go2BaseVelAction"


def test_go2_config_includes_head_and_split_cameras():
    interactive_play = _load_interactive_play()
    config = get_config("benchmark/rearrange/play/play_go2.yaml")

    with habitat.config.read_write(config):
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        interactive_play.configure_go2_sensors(agent_config)

        sensors = {
            sensor.uuid: sensor for sensor in agent_config.sim_sensors.values()
        }
        assert set(sensors) == {"head_rgb", "game_rgb", "free_rgb"}
        assert sensors["head_rgb"].width == 256
        assert sensors["head_rgb"].height == 256
        for name in ("game_rgb", "free_rgb"):
            assert sensors[name].width == 800
            assert sensors[name].height == 900


def test_go2_split_view_layout():
    interactive_play = _load_interactive_play()
    observations = {
        interactive_play.GO2_FOLLOW_CAMERA: np.zeros(
            (900, 800, 3), dtype=np.uint8
        ),
        interactive_play.GO2_FREE_CAMERA: np.ones(
            (900, 800, 3), dtype=np.uint8
        ),
    }

    split_view = interactive_play.get_go2_split_view(observations)

    assert split_view.shape == (900, 1600, 3)
    assert np.all(split_view[:, :800] == 0)
    assert np.all(split_view[:, 800:] == 1)


def test_go2_velocity_targets_are_limited():
    action = _make_go2_action()

    assert action._target_velocities(2.0, -2.0) == (0.5, -1.0)
    assert action._target_velocities(-2.0, 0.0) == (-0.5, 0.0)


def test_go2_velocity_targets_respect_no_reverse_setting():
    action = _make_go2_action(allow_back=False)

    assert action._target_velocities(-1.0, 0.0) == (0.0, 0.0)


def test_go2_velocity_smoothing_ramps_and_stops():
    action = _make_go2_action()

    accelerated = action._smooth_velocity(0.0, 0.5, 0.1)
    assert 0.0 < accelerated < 0.5
    assert 0.0 < action._smooth_velocity(accelerated, 0.0, 0.1) < accelerated
    assert action._smooth_velocity(1e-5, 0.0, 0.1) == 0.0
    assert action._smooth_velocity(0.0, 0.5, 0.0) == pytest.approx(0.0)
