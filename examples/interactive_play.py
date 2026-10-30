#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Manually control the articulated agent to interact with the environment. Run as
```
python examples/interactive_play.py
```

To Run you need PyGame installed (to install run `pip install pygame==2.0.1`).

By default this controls with velocity control (which makes controlling the
agent hard). To use IK control instead add the `--add-ik` command line argument.

Controls:
- For velocity control
    - 1-7 to increase the motor target for the articulated agent arm joints
    - Q-U to decrease the motor target for the articulated agent arm joints
- For IK control
    - W,S,A,D to move side to side
    - E,Q to move up and down
- I,J,K,L to move the articulated agent base around
- PERIOD to print the current world coordinates of the articulated agent base.
- Z to toggle the camera to free movement mode. When in free camera mode:
    - Arrow keys and PageUp/PageDown to translate the camera
    - I,J,K,L,U,O to rotate the camera
    - B to reset the camera position
- X to change the articulated agent that is being controlled (if there are multiple articulated agents).

Change the task with `--cfg benchmark/rearrange/close_cab.yaml` (choose any task under the `habitat-lab/habitat/config/task/rearrange/` folder).

Change the grip type:
- Suction gripper `task.actions.arm_action.grip_controller "SuctionGraspAction"`

For the Go2 split-screen mode:
- `W/S` move forward and backward.
- `A/D` turn left and right.
- Arrow keys move the free camera, with `PageUp/PageDown` for elevation.
- `I/K`, `U/O`, and `J/L` rotate the free camera.

To record a video: `--save-obs` This will save the video to file under `data/vids/` specified by `--save-obs-fname` (by default `vid.mp4`).

Record and play back trajectories:
- To record a trajectory add `--save-actions --save-actions-count 200` to
  record a truncated episode length of 200.
- By default the trajectories are saved to data/interactive_play_replays/play_actions.txt
- Play the trajectories back with `--load-actions data/interactive_play_replays/play_actions.txt`
"""

import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List

import cv2
import magnum as mn
import numpy as np

import habitat
import habitat_sim
import habitat.tasks.rearrange.rearrange_task
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    HeadRGBSensorConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut

try:
    import pygame
except ImportError:
    pygame = None

# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


def step_env(env, action_name, action_args):
    return env.step({"action": action_name, "action_args": action_args})


def get_input_vel_ctlr(
    skip_pygame,
    cfg,
    arm_action,
    env,
    not_block_input,
    agent_to_control,
    control_humanoid,
    humanoid_controller,
):
    if skip_pygame:
        return step_env(env, "empty", {}), None, False
    multi_agent = len(env._sim.agents_mgr) > 1

    if multi_agent:
        agent_k = f"agent_{agent_to_control}_"
    else:
        agent_k = ""
    arm_action_name = f"{agent_k}arm_action"

    if control_humanoid:
        base_action_name = f"{agent_k}humanoidjoint_action"
        base_key = "human_joints_trans"
    else:
        if "spot" in cfg:
            base_action_name = f"{agent_k}base_velocity_non_cylinder"
        else:
            base_action_name = f"{agent_k}base_velocity"
        arm_key = "arm_action"
        grip_key = "grip_action"
        base_key = "base_vel"

    if arm_action_name in env.action_space.spaces:
        arm_action_space = env.action_space.spaces[arm_action_name].spaces[
            arm_key
        ]
        arm_ctrlr = env.task.actions[arm_action_name].arm_ctrlr
        base_action = None
    elif "stretch" in cfg:
        arm_action_space = np.zeros(10)
        arm_ctrlr = None
        base_action = [0, 0]
    else:
        arm_action_space = np.zeros(7)
        arm_ctrlr = None
        base_action = [0, 0]

    if arm_action is None:
        arm_action = np.zeros(arm_action_space.shape[0])
        given_arm_action = False
    else:
        given_arm_action = True

    end_ep = False
    magic_grasp = None

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        return None, None, False
    elif keys[pygame.K_m]:
        end_ep = True
    elif keys[pygame.K_n]:
        env._sim.navmesh_visualization = not env._sim.navmesh_visualization

    if not_block_input:
        # Base control
        if "go2" in cfg:
            left, right, back, forward = (
                pygame.K_a,
                pygame.K_d,
                pygame.K_s,
                pygame.K_w,
            )
        else:
            left, right, back, forward = (
                pygame.K_j,
                pygame.K_l,
                pygame.K_k,
                pygame.K_i,
            )

        if keys[left]:
            base_action = [0, 1]
        elif keys[right]:
            base_action = [0, -1]
        elif keys[back]:
            base_action = [-1, 0]
        elif keys[forward]:
            base_action = [1, 0]

        if arm_action_space.shape[0] == 7:
            # Velocity control. A different key for each joint
            if keys[pygame.K_q]:
                arm_action[0] = 1.0
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0

            elif keys[pygame.K_w]:
                arm_action[1] = 1.0
            elif keys[pygame.K_2]:
                arm_action[1] = -1.0

            elif keys[pygame.K_e]:
                arm_action[2] = 1.0
            elif keys[pygame.K_3]:
                arm_action[2] = -1.0

            elif keys[pygame.K_r]:
                arm_action[3] = 1.0
            elif keys[pygame.K_4]:
                arm_action[3] = -1.0

            elif keys[pygame.K_t]:
                arm_action[4] = 1.0
            elif keys[pygame.K_5]:
                arm_action[4] = -1.0

            elif keys[pygame.K_y]:
                arm_action[5] = 1.0
            elif keys[pygame.K_6]:
                arm_action[5] = -1.0

            elif keys[pygame.K_u]:
                arm_action[6] = 1.0
            elif keys[pygame.K_7]:
                arm_action[6] = -1.0

        elif arm_action_space.shape[0] == 4:
            # Velocity control. A different key for each joint
            # This is for Spot robot which a user can only control the effective arm in the real robot
            if keys[pygame.K_q]:
                arm_action[0] = 1.0
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0

            elif keys[pygame.K_w]:
                arm_action[1] = 1.0
            elif keys[pygame.K_2]:
                arm_action[1] = -1.0

            elif keys[pygame.K_e]:
                arm_action[2] = 1.0
            elif keys[pygame.K_3]:
                arm_action[2] = -1.0

            elif keys[pygame.K_r]:
                arm_action[3] = 1.0
            elif keys[pygame.K_4]:
                arm_action[3] = -1.0

        elif arm_action_space.shape[0] == 10:
            # Velocity control. A different key for each joint
            if keys[pygame.K_q]:
                arm_action[0] = 1.0
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0

            elif keys[pygame.K_w]:
                arm_action[4] = 1.0
            elif keys[pygame.K_2]:
                arm_action[4] = -1.0

            elif keys[pygame.K_e]:
                arm_action[5] = 1.0
            elif keys[pygame.K_3]:
                arm_action[5] = -1.0

            elif keys[pygame.K_r]:
                arm_action[6] = 1.0
            elif keys[pygame.K_4]:
                arm_action[6] = -1.0

            elif keys[pygame.K_t]:
                arm_action[7] = 1.0
            elif keys[pygame.K_5]:
                arm_action[7] = -1.0

            elif keys[pygame.K_y]:
                arm_action[8] = 1.0
            elif keys[pygame.K_6]:
                arm_action[8] = -1.0

            elif keys[pygame.K_u]:
                arm_action[9] = 1.0
            elif keys[pygame.K_7]:
                arm_action[9] = -1.0

        elif isinstance(arm_ctrlr, ArmEEAction):
            EE_FACTOR = 0.5
            # End effector control
            if keys[pygame.K_d]:
                arm_action[1] -= EE_FACTOR
            elif keys[pygame.K_a]:
                arm_action[1] += EE_FACTOR
            elif keys[pygame.K_w]:
                arm_action[0] += EE_FACTOR
            elif keys[pygame.K_s]:
                arm_action[0] -= EE_FACTOR
            elif keys[pygame.K_q]:
                arm_action[2] += EE_FACTOR
            elif keys[pygame.K_e]:
                arm_action[2] -= EE_FACTOR
        else:
            raise ValueError("Unrecognized arm action space")

        if keys[pygame.K_p]:
            logger.info("[play.py]: Unsnapping")
            # Unsnap
            magic_grasp = -1
        elif keys[pygame.K_o]:
            # Snap
            logger.info("[play.py]: Snapping")
            magic_grasp = 1

    if control_humanoid:
        if humanoid_controller is None:
            # Add random noise to human arms but keep global transform
            (
                joint_trans,
                root_trans,
            ) = env._sim.articulated_agent.get_joint_transform()
            # Divide joint_trans by 4 since joint_trans has flattened quaternions
            # and the dimension of each quaternion is 4
            num_joints = len(joint_trans) // 4
            root_trans = np.array(root_trans)
            index_arms_start = 10
            joint_trans_quat = [
                mn.Quaternion(
                    mn.Vector3(joint_trans[(4 * index) : (4 * index + 3)]),
                    joint_trans[4 * index + 3],
                )
                for index in range(num_joints)
            ]
            rotated_joints_quat = []
            for index, joint_quat in enumerate(joint_trans_quat):
                random_vec = np.random.rand(3)
                # We allow for maximum 10 angles per step
                random_angle = np.random.rand() * 10
                rotation_quat = mn.Quaternion.rotation(
                    mn.Rad(random_angle), mn.Vector3(random_vec).normalized()
                )
                if index > index_arms_start:
                    joint_quat *= rotation_quat
                rotated_joints_quat.append(joint_quat)
            joint_trans = np.concatenate(
                [
                    np.array(list(quat.vector) + [quat.scalar])
                    for quat in rotated_joints_quat
                ]
            )
            base_action = np.concatenate(
                [joint_trans.reshape(-1), root_trans.transpose().reshape(-1)]
            )
        else:
            # Use the controller
            relative_pos = mn.Vector3(base_action[0], 0, base_action[1])
            humanoid_controller.calculate_walk_pose(relative_pos)
            base_action = humanoid_controller.get_pose()

    if keys[pygame.K_PERIOD]:
        # Print the current position of the articulated agent, useful for debugging.
        pos = [
            float("%.3f" % x)
            for x in env._sim.articulated_agent.sim_obj.translation
        ]
        rot = env._sim.articulated_agent.sim_obj.rotation
        ee_pos = env._sim.articulated_agent.ee_transform().translation
        logger.info(
            f"Agent state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
        )
    elif keys[pygame.K_COMMA]:
        # Print the current arm state of the articulated agent, useful for debugging.
        joint_state = [
            float("%.3f" % x) for x in env._sim.articulated_agent.arm_joint_pos
        ]
        logger.info(f"Agent arm joint state: {joint_state}")

    args: Dict[str, Any] = {}

    if base_action is not None and base_action_name in env.action_space.spaces:
        name = base_action_name
        args = {base_key: base_action}
    else:
        name = arm_action_name
        if given_arm_action:
            # The grip is also contained in the provided action
            args = {
                arm_key: arm_action[:-1],
                grip_key: arm_action[-1],
            }
        else:
            args = {arm_key: arm_action, grip_key: magic_grasp}

    if magic_grasp is None:
        arm_action = [*arm_action, 0.0]
    else:
        arm_action = [*arm_action, magic_grasp]

    return step_env(env, name, args), arm_action, end_ep


class FreeCamHelper:
    _MOVE_STEP = 0.1
    _ROTATE_STEP = 0.1

    def __init__(self, sensor_name):
        self._is_free_cam_mode = False
        self._last_pressed = 0
        self._initial_transform = None
        self._free_position = mn.Vector3.zero_init()
        self._free_quat = mn.Quaternion()
        self._pressed = set()
        self._sensor_name = sensor_name

    @property
    def is_free_cam_mode(self):
        return self._is_free_cam_mode

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                self._pressed.add(event.key)
            elif event.type == pygame.KEYUP:
                self._pressed.discard(event.key)
            elif event.type == pygame.WINDOWFOCUSLOST:
                self._pressed.clear()

    def update(self, env, update_idx):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_z] and (update_idx - self._last_pressed) > 60:
            self._is_free_cam_mode = not self._is_free_cam_mode
            logger.info(f"Switching camera mode to {self._is_free_cam_mode}")
            self._last_pressed = update_idx
            if self._is_free_cam_mode:
                self._set_to_sensor(env)

        if self._is_free_cam_mode:
            offset_rpy = np.zeros(3)
            if keys[pygame.K_u]:
                offset_rpy[1] += 1
            elif keys[pygame.K_o]:
                offset_rpy[1] -= 1
            elif keys[pygame.K_i]:
                offset_rpy[2] += 1
            elif keys[pygame.K_k]:
                offset_rpy[2] -= 1
            elif keys[pygame.K_j]:
                offset_rpy[0] += 1
            elif keys[pygame.K_l]:
                offset_rpy[0] -= 1

            offset_xyz = np.zeros(3)
            if pygame.K_PAGEUP in self._pressed:
                offset_xyz[1] += 1
            if pygame.K_PAGEDOWN in self._pressed:
                offset_xyz[1] -= 1
            if pygame.K_UP in self._pressed:
                offset_xyz[2] -= 1
            if pygame.K_DOWN in self._pressed:
                offset_xyz[2] += 1
            if pygame.K_LEFT in self._pressed:
                offset_xyz[0] -= 1
            if pygame.K_RIGHT in self._pressed:
                offset_xyz[0] += 1
            if np.any(offset_rpy):
                rotation_delta = euler_to_quat(
                    offset_rpy * self._ROTATE_STEP
                )
                self._free_quat = self._free_quat * rotation_delta
            if np.any(offset_xyz):
                local_step = mn.Vector3(*offset_xyz)
                self._free_position += (
                    self._free_quat.transform_vector(local_step)
                    * self._MOVE_STEP
                )
            if keys[pygame.K_b]:
                self._free_position = mn.Vector3(
                    self._initial_transform.translation
                )
                self._free_quat = mn.Quaternion.from_matrix(
                    self._initial_transform.rotation()
                )

            self.apply_transform(env)

    def apply_transform(self, env):
        camera_transform = mn.Matrix4.from_(
            self._free_quat.to_matrix(), self._free_position
        )
        set_visual_sensor_transform(env, self._sensor_name, camera_transform)

    def set_view(self, env, camera_transform):
        self._initial_transform = camera_transform
        self._free_position = mn.Vector3(camera_transform.translation)
        self._free_quat = mn.Quaternion.from_matrix(
            camera_transform.rotation()
        )
        self._is_free_cam_mode = True
        self.apply_transform(env)

    def _set_to_sensor(self, env):
        sensor = env._sim.get_sensor(self._sensor_name).sensor_object
        self._initial_transform = (
            sensor.render_camera.camera_matrix.inverted()
        )
        self._free_position = mn.Vector3(
            self._initial_transform.translation
        )
        self._free_quat = mn.Quaternion.from_matrix(
            self._initial_transform.rotation()
        )
        self.apply_transform(env)


GO2_GAME_WIDTH = 800
GO2_GAME_HEIGHT = 900
GO2_RENDER_WIDTH = GO2_GAME_WIDTH
GO2_RENDER_HEIGHT = GO2_GAME_HEIGHT
GO2_FOLLOW_SENSOR = "game_rgb"
GO2_FREE_SENSOR = "free_rgb"
GO2_SENSORS = (GO2_FOLLOW_SENSOR, GO2_FREE_SENSOR)


def add_go2_sensor_configs(agent_config):
    agent_config.sim_sensors["head_rgb_sensor"] = HeadRGBSensorConfig()
    agent_config.sim_sensors.update(
        {
            f"{sensor_uuid}_sensor": ThirdRGBSensorConfig(
                uuid=sensor_uuid,
                width=GO2_RENDER_WIDTH,
                height=GO2_RENDER_HEIGHT,
            )
            for sensor_uuid in GO2_SENSORS
        }
    )


def set_visual_sensor_transform(
    env, sensor_uuid, global_camera_transform
):
    sensor_object = env._sim.get_sensor(sensor_uuid)._sensor_object
    render_camera = sensor_object.render_camera
    cumulative_transform = mn.Matrix4.identity_init()
    node = render_camera.node.parent
    while node is not None and hasattr(node, "transformation"):
        cumulative_transform @= node.transformation
        node = node.parent
    render_camera.node.transformation = (
        cumulative_transform.inverted() @ global_camera_transform
    )


def get_sensor_observations(env):
    env._sim.start_async_render()
    return env._sim.get_sensor_observations_async_finish()


def get_go2_view(obs, sensor_uuid):
    view = obs[sensor_uuid][:, :, :3]
    if view.shape[:2] != (GO2_GAME_HEIGHT, GO2_GAME_WIDTH):
        view = cv2.resize(
            view,
            (GO2_GAME_WIDTH, GO2_GAME_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
    return view


def get_go2_split_view(obs):
    return np.concatenate(
        [get_go2_view(obs, sensor_uuid) for sensor_uuid in GO2_SENSORS],
        axis=1,
    )


class Go2SplitPresenter:
    """Write both camera views directly into preallocated display targets."""

    def __init__(self, screen):
        self._targets = tuple(
            screen.subsurface(
                (
                    index * GO2_GAME_WIDTH,
                    0,
                    GO2_GAME_WIDTH,
                    GO2_GAME_HEIGHT,
                )
            )
            for index in range(len(GO2_SENSORS))
        )

    def draw(self, obs):
        for sensor_uuid, target in zip(GO2_SENSORS, self._targets):
            view = get_go2_view(obs, sensor_uuid)
            pygame.surfarray.blit_array(
                target, np.transpose(view, (1, 0, 2))
            )
        pygame.display.update()


def place_go2_at_scene_center(env):
    robot = env._sim.articulated_agent
    lower_bound, upper_bound = env._sim.pathfinder.get_bounds()
    scene_center = (lower_bound + upper_bound) * 0.5
    scene_center.y = robot.base_pos.y
    snapped_position = env._sim.pathfinder.snap_point(scene_center)
    if np.all(
        np.isfinite(
            [
                snapped_position.x,
                snapped_position.y,
                snapped_position.z,
            ]
        )
    ):
        snapped_position.y = robot.base_pos.y
        robot.base_pos = snapped_position
        robot.align_to_ground()


def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    use_arm_actions = None
    if args.load_actions is not None:
        with open(args.load_actions, "rb") as f:
            use_arm_actions = np.load(f)
            logger.info("Loaded arm actions")

    obs = env.reset()
    free_cam_sensor_name = (
        GO2_FREE_SENSOR
        if args.go2_game
        else ("third_rgb" if "third_rgb" in obs else "head_rgb_sensor")
    )
    free_cam = FreeCamHelper(free_cam_sensor_name)
    go2_presenter = None

    if not args.no_render:
        if args.go2_game:
            screen = pygame.display.set_mode(
                [GO2_GAME_WIDTH * 2, GO2_GAME_HEIGHT]
            )
            go2_presenter = Go2SplitPresenter(screen)
        else:
            draw_obs = observations_to_image(obs, {})
            screen = pygame.display.set_mode(
                [draw_obs.shape[1], draw_obs.shape[0]]
            )

    if args.go2_game:
        place_go2_at_scene_center(env)
        robot = env._sim.articulated_agent
        robot.update()
        free_cam.set_view(env, robot.follow_camera_transform)
        obs = get_sensor_observations(env)

    update_idx = 0
    target_fps = 30.0 if args.go2_game else 60.0
    prev_time = time.time()
    fps_measure_start = prev_time
    fps_measure_frames = 0
    all_obs = []
    total_reward = 0
    all_arm_actions: List[float] = []
    agent_to_control = 0

    gfx_measure = env.task.measurements.measures.get(
        GfxReplayMeasure.cls_uuid, None
    )
    is_multi_agent = len(env._sim.agents_mgr) > 1

    humanoid_controller = None
    if args.use_humanoid_controller:
        humanoid_controller = HumanoidRearrangeController(args.walk_pose_path)
        humanoid_controller.reset(env._sim.articulated_agent.base_pos)

    while True:
        if not args.no_render:
            free_cam.handle_events(pygame.event.get())
        if (
            args.save_actions
            and len(all_arm_actions) > args.save_actions_count
        ):
            # quit the application when the action recording queue is full
            break
        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        if args.no_render:
            keys = defaultdict(lambda: False)
        else:
            keys = pygame.key.get_pressed()
            free_cam.update(env, update_idx)

        if not args.no_render and is_multi_agent and keys[pygame.K_x]:
            agent_to_control += 1
            agent_to_control = agent_to_control % len(env._sim.agents_mgr)
            logger.info(
                f"Controlled agent changed. Controlling agent {agent_to_control}."
            )

        allow_robot_input = args.go2_game or not free_cam.is_free_cam_mode
        step_result, arm_action, end_ep = get_input_vel_ctlr(
            args.no_render,
            args.cfg,
            use_arm_actions[update_idx]
            if use_arm_actions is not None
            else None,
            env,
            allow_robot_input,
            agent_to_control,
            args.control_humanoid,
            humanoid_controller=humanoid_controller,
        )

        if (
            not args.no_render
            and keys[pygame.K_c]
            and "pddl_apply_action" in env.task.actions
        ):
            pddl_action = env.task.actions["pddl_apply_action"]
            logger.info("Actions:")
            actions = pddl_action._action_ordering
            for i, action in enumerate(actions):
                logger.info(f"{i}: {action}")
            entities = pddl_action._entities_list
            logger.info("Entities")
            for i, entity in enumerate(entities):
                logger.info(f"{i}: {entity}")
            action_sel = input("Enter Action Selection: ")
            entity_sel = input("Enter Entity Selection: ")
            action_sel = int(action_sel)
            entity_sel = [int(x) + 1 for x in entity_sel.split(",")]
            ac = np.zeros(pddl_action.action_space["pddl_action"].shape[0])
            ac_start = pddl_action.get_pddl_action_start(action_sel)
            ac[ac_start : ac_start + len(entity_sel)] = entity_sel

            step_env(env, "pddl_apply_action", {"pddl_action": ac})

        if (
            not args.no_render
            and keys[pygame.K_g]
            and "all_predicates" in env.task.sensor_suite.sensors
        ):
            pred_list = env.task.sensor_suite.sensors[
                "all_predicates"
            ]._predicates_list
            pred_values = step_result["all_predicates"]
            logger.info("\nPredicate Truth Values:")
            for i, (pred, pred_value) in enumerate(
                zip(pred_list, pred_values)
            ):
                logger.info(f"{i}: {pred.compact_str} = {pred_value}")

        if step_result is None:
            break

        if end_ep:
            total_reward = 0
            # Clear the saved keyframes.
            if gfx_measure is not None:
                gfx_measure.get_metric(force_get=True)
            env.reset()

        all_arm_actions.append(arm_action)
        update_idx += 1
        if use_arm_actions is not None and update_idx >= len(use_arm_actions):
            break

        obs = step_result
        info = env.get_metrics()
        reward_key = [k for k in info if "reward" in k]
        if len(reward_key) > 0:
            reward = info[reward_key[0]]
        else:
            reward = 0.0

        total_reward += reward
        info["Total Reward"] = total_reward

        use_ob = None
        if not args.go2_game and free_cam.is_free_cam_mode:
            cam = obs[free_cam_sensor_name][:, :, :3]
            use_ob = cv2.resize(
                cam,
                (draw_obs.shape[1], draw_obs.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        elif not args.go2_game:
            use_ob = observations_to_image(obs, info)
            if not args.skip_render_text:
                use_ob = overlay_frame(use_ob, info)

        if not args.no_render:
            if args.go2_game:
                go2_presenter.draw(obs)
            else:
                surface = pygame.surfarray.make_surface(
                    np.transpose(use_ob, (1, 0, 2))
                )
                screen.blit(surface, (0, 0))
                pygame.display.update()
        if args.save_obs:
            if args.go2_game:
                saved_obs = np.transpose(
                    get_go2_split_view(obs), (1, 0, 2)
                )
            elif args.no_render:
                saved_obs = use_ob
            else:
                saved_obs = np.transpose(use_ob, (1, 0, 2))
            all_obs.append(saved_obs)  # type: ignore[arg-type]

        if not args.no_render:
            pygame.event.pump()
        if env.episode_over:
            total_reward = 0
            env.reset()

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        time.sleep(delay)
        prev_time = curr_time
        fps_measure_frames += 1
        measured_elapsed = time.time() - fps_measure_start
        if not args.no_render and measured_elapsed >= 1.0:
            measured_fps = fps_measure_frames / measured_elapsed
            pygame.display.set_caption(
                f"Habitat interactive play - {measured_fps:.0f} FPS"
            )
            fps_measure_start = time.time()
            fps_measure_frames = 0

    if args.save_actions:
        if len(all_arm_actions) < args.save_actions_count:
            raise ValueError(
                f"Only did {len(all_arm_actions)} actions but {args.save_actions_count} are required"
            )
        all_arm_actions = all_arm_actions[: args.save_actions_count]
        os.makedirs(SAVE_ACTIONS_DIR, exist_ok=True)
        save_path = osp.join(SAVE_ACTIONS_DIR, args.save_actions_fname)
        with open(save_path, "wb") as f:
            np.save(f, all_arm_actions)
        logger.info(f"Saved actions to {save_path}")
        pygame.quit()
        return

    if args.save_obs:
        all_obs = np.array(all_obs)  # type: ignore[assignment]
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))  # type: ignore[assignment]
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(
            np.expand_dims(all_obs, 1),
            0,
            "color",
            osp.join(SAVE_VIDEO_DIR, args.save_obs_fname),
        )
    if gfx_measure is not None:
        gfx_str = gfx_measure.get_metric(force_get=True)
        write_gfx_replay(
            gfx_str, config.habitat.task, env.current_episode.episode_id
        )

    if not args.no_render:
        pygame.quit()


def has_pygame():
    return pygame is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument(
        "--save-actions-fname", type=str, default="play_actions.txt"
    )
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="""
            The number of steps the saved action trajectory is clipped to. NOTE
            the episode must be at least this long or it will terminate with
            error.
            """,
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--go2-game",
        action="store_true",
        default=False,
        help="Show a split view with a Go2 follow camera and a free camera.",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
    )

    parser.add_argument(
        "--control-humanoid",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--use-humanoid-controller",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )

    args = parser.parse_args()
    if not has_pygame() and not args.no_render:
        raise ImportError(
            "Need to install PyGame (run `pip install pygame==2.0.1`)"
        )
    if not args.no_render:
        pygame.init()

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task
        agent_config = get_agent_config(sim_config=sim_config)

        if args.go2_game:
            add_go2_sensor_configs(agent_config)

        if not args.same_task:
            sim_config.debug_render = True
            if not args.go2_game:
                agent_config.sim_sensors.update(
                    {
                        "third_rgb_sensor": ThirdRGBSensorConfig(
                            height=args.play_cam_res,
                            width=args.play_cam_res,
                        )
                    }
                )
            if "pddl_success" in task_config.measurements:
                task_config.measurements.pddl_success.must_call_stop = False
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if args.control_humanoid:
            args.disable_inverse_kinematics = True

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"
        if task_config.type == "RearrangePddlTask-v0":
            task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()

    with habitat.Env(config=config) as env:
        play_env(env, args, config)
