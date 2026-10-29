# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.physics import JointMotorSettings, MotionType

from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
    MobileManipulatorParams,
)


class Go2Robot(MobileManipulator):
    """Habitat wrapper for the Unitree Go2 quadruped model."""

    _leg_names = (
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
    )
    _STAND_THIGH = 0.8
    _STAND_KNEE = -1.6
    _LEG_SEGMENT_LENGTH = 0.213
    _GAIT_FREQUENCY = 1.8
    _MAX_STRIDE = 0.08
    _FOOT_LIFT = 0.045
    _stand_pose = [0.0, _STAND_THIGH, _STAND_KNEE] * 4
    _foot_link_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    def __init__(self, agent_cfg, sim, limit_robo_joints=True, fixed_base=True):
        self._gait_phase = 0.0
        self._gait_pose = np.array(self._stand_pose, dtype=np.float32)
        params = MobileManipulatorParams(
            arm_joints=[],
            gripper_joints=[],
            # Rearrange's grasp manager expects one transform even when no
            # grasp action is configured; the front-left foot is a placeholder.
            ee_offset=[mn.Vector3.zero_init()],
            ee_links=[3],
            ee_constraint=np.zeros((1, 2, 1), dtype=np.float32),
            cameras={
                "head": ArticulatedAgentCameraParams(
                    attached_link_id=-1,
                    cam_offset_pos=mn.Vector3(0.24, 0.0, 0.16),
                )
            },
            gripper_closed_state=np.zeros(0, dtype=np.float32),
            gripper_open_state=np.zeros(0, dtype=np.float32),
            gripper_state_eps=0.01,
            arm_mtr_pos_gain=0.0,
            arm_mtr_vel_gain=0.0,
            arm_mtr_max_impulse=0.0,
            base_offset=mn.Vector3(0.0, -0.30, 0.0),
            base_link_names={"base"},
            arm_init_params=np.zeros(0, dtype=np.float32),
            gripper_init_params=np.zeros(0, dtype=np.float32),
            leg_joints=[],
            leg_init_params=[],
            leg_mtr_pos_gain=25.0,
            leg_mtr_vel_gain=3.0,
            leg_mtr_max_impulse=500.0,
            ee_count=1,
        )
        super().__init__(
            params,
            agent_cfg,
            sim,
            limit_robo_joints=limit_robo_joints,
            fixed_base=fixed_base,
            base_type="leg",
        )

    @property
    def base_rot(self) -> float:
        forward = self.sim_obj.transformation.transform_vector(
            mn.Vector3.x_axis()
        )
        return float(np.arctan2(-forward.z, forward.x))

    @base_rot.setter
    def base_rot(self, rotation_y_rad: float) -> None:
        yaw = mn.Quaternion.rotation(
            mn.Rad(float(rotation_y_rad)), mn.Vector3(0, 1, 0)
        )
        urdf_y_up = mn.Quaternion.rotation(
            mn.Deg(-90), mn.Vector3(1, 0, 0)
        )
        self.sim_obj.rotation = yaw * urdf_y_up

    @property
    def base_pos(self):
        return self.sim_obj.translation + mn.Vector3(0.0, -0.30, 0.0)

    @base_pos.setter
    def base_pos(self, position: mn.Vector3) -> None:
        self.sim_obj.translation = position - mn.Vector3(0.0, -0.30, 0.0)

    def foot_positions(self):
        positions = []
        for link_name in self._foot_link_names:
            link_id = self.sim_obj.get_link_id_from_name(link_name)
            positions.append(
                mn.Vector3(
                    self.sim_obj.get_link_scene_node(link_id).translation
                )
            )
        return positions

    def align_to_ground(self) -> None:
        """Put the four level feet on the support surface below them."""
        foot_positions = self.foot_positions()
        robot_object_ids = set(self.sim_obj.link_object_ids)
        support_heights = []
        down = -mn.Vector3.y_axis()
        for foot_position in foot_positions:
            ray = habitat_sim.geo.Ray(
                foot_position + mn.Vector3.y_axis(2.0), down
            )
            for hit in self._sim.cast_ray(ray, max_distance=5.0).hits:
                if hit.object_id not in robot_object_ids:
                    support_heights.append(float(hit.point.y))
                    break

        if not support_heights:
            return

        support_height = float(np.median(support_heights))
        foot_height = float(
            np.median([position.y for position in foot_positions])
        )
        target_translation = mn.Vector3(self.sim_obj.translation)
        target_translation.y += support_height - foot_height
        self.sim_obj.transformation = mn.Matrix4.from_(
            self.sim_obj.transformation.rotation(),
            target_translation,
        )

    def reconfigure(self):
        super().reconfigure()
        self.sim_obj.motion_type = MotionType.KINEMATIC
        self.params.leg_joints = [
            self.sim_obj.get_link_id_from_name(name)
            for name in self._leg_names
        ]
        self.params.leg_init_params = list(self._stand_pose)

        for joint, target in zip(
            self.params.leg_joints, self.params.leg_init_params
        ):
            motor_id = self.joint_motors[joint][0]
            settings = JointMotorSettings(
                target,
                self.params.leg_mtr_pos_gain,
                0.0,
                self.params.leg_mtr_vel_gain,
                self.params.leg_mtr_max_impulse,
            )
            self.sim_obj.update_joint_motor(motor_id, settings)

        self._update_motor_settings_cache()
        self.leg_joint_pos = self.params.leg_init_params

    def reset(self):
        self._gait_phase = 0.0
        self._gait_pose = np.array(self._stand_pose, dtype=np.float32)
        super().reset()

    def advance_gait(self, forward, turn, dt):
        """Animate a diagonal trot using planar leg inverse kinematics.

        Commands are metres/second and radians/second. This animates the
        legs of the kinematically moved base, rather than balancing it.
        """
        # Each side has a different stride when turning. FL/RR and FR/RL
        # are opposite phase, so a diagonal pair lifts together.
        speeds = np.array(
            [forward - turn * 0.14, forward + turn * 0.14] * 2
        )
        activity = min(1.0, float(np.max(np.abs(speeds))) / 0.15)
        if activity > 0.0:
            self._gait_phase = (
                self._gait_phase
                + dt * 2.0 * np.pi * self._GAIT_FREQUENCY
            ) % (2.0 * np.pi)
            phases = self._gait_phase + np.array([0.0, np.pi, np.pi, 0.0])
            stride = np.clip(
                speeds / (4.0 * self._GAIT_FREQUENCY),
                -self._MAX_STRIDE,
                self._MAX_STRIDE,
            )
            foot_x = stride * np.cos(phases)
            lift = (
                self._FOOT_LIFT
                * activity
                * np.maximum(0.0, np.sin(phases))
            )
            foot_down = (
                2.0
                * self._LEG_SEGMENT_LENGTH
                * np.cos(self._STAND_THIGH)
                - lift
            )
            cosine = np.clip(
                (
                    foot_x**2
                    + foot_down**2
                    - 2.0 * self._LEG_SEGMENT_LENGTH**2
                )
                / (2.0 * self._LEG_SEGMENT_LENGTH**2),
                -1.0,
                1.0,
            )
            knee = -np.arccos(cosine)
            thigh = np.arctan2(-foot_x, foot_down) - knee / 2.0
            target = np.column_stack((np.zeros(4), thigh, knee)).ravel()
        else:
            target = np.array(self._stand_pose, dtype=np.float32)
            self._gait_phase = 0.0

        # Smooth starts/stops and changes of direction.
        blend = 1.0 - np.exp(-dt / 0.06)
        self._gait_pose += blend * (target - self._gait_pose)
        indices = [self.joint_pos_indices[j] for j in self.params.leg_joints]
        lower, upper = (np.asarray(limit) for limit in self.joint_limits)
        self._gait_pose = np.clip(
            self._gait_pose, lower[indices], upper[indices]
        )
        self.leg_joint_pos = self._gait_pose


__all__ = ["Go2Robot"]
