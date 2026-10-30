#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from habitat.config.default import get_config
from habitat.tasks.rearrange.actions.actions import Go2BaseVelAction


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
