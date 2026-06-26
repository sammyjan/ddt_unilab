from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
)

D1_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RR_foot",
)
NUM_D1_ACTIONS = len(D1_JOINT_NAMES)

D1_FOOT_INDICES = np.asarray([3, 7, 11, 15], dtype=np.int32)
D1_HIP_INDICES = np.asarray([0, 4, 8, 12], dtype=np.int32)
D1_LEG_INDICES = np.asarray(
    [i for i in range(NUM_D1_ACTIONS) if i not in D1_FOOT_INDICES],
    dtype=np.int32,
)

DEFAULT_D1_ANGLES = np.asarray(
    [
        0.1, 0.8, -1.5, 0.0,   # FL
        -0.1, 0.8, -1.5, 0.0,  # FR
        0.1, 1.0, -1.5, 0.0,   # RL
        -0.1, 1.0, -1.5, 0.0,  # RR
    ],
    dtype=np.float64,
)


@dataclass
class NoiseConfig(BaseNoiseConfig):
    pass


@dataclass
class ControlConfig(PdControlConfig):
    action_scale: float = 0.25
    hip_scale_reduction: float = 0.5
    foot_Kp: float = 10.0
    foot_Kd: float = 0.0
    clip_actions: float = 1.0


@dataclass
class Asset:
    base_name = "base_link"
    foot_name = "foot"
    ground = "floor"


@dataclass
class D1BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.0025
    ctrl_dt: float = 0.02


class D1BaseEnv(LocomotionBaseEnv):
    _cfg: D1BaseCfg

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_D1_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.default_angles = np.asarray(DEFAULT_D1_ANGLES, dtype=self.default_angles.dtype)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
            dtype=self.default_angles.dtype,
        )
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(clipped_actions)
        )
        state.info["current_actions"] = clipped_actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else clipped_actions
        )

        actions_scaled = exec_actions * self._cfg.control_config.action_scale
        actions_scaled[:, D1_HIP_INDICES] *= self._cfg.control_config.hip_scale_reduction

        targets = actions_scaled + self.default_angles
        return targets

    def _pre_step_motor_control(self, backend, policy_ctrl: np.ndarray) -> np.ndarray:
        joint_pos = self.get_dof_pos()
        joint_vel = self.get_dof_vel()

        kp = self._cfg.control_config.Kp
        kd = self._cfg.control_config.Kd
        kp_foot = self._cfg.control_config.foot_Kp
        kd_foot = self._cfg.control_config.foot_Kd

        torque = np.empty_like(policy_ctrl)
        # leg joints: kp*(target - pos) - kd*vel
        torque[:, D1_LEG_INDICES] = (
            kp * (policy_ctrl[:, D1_LEG_INDICES] - joint_pos[:, D1_LEG_INDICES])
            - kd * joint_vel[:, D1_LEG_INDICES]
        )
        # foot joints: kp_foot*(target - pos) only (kd_foot = 0)
        torque[:, D1_FOOT_INDICES] = (
            kp_foot * (policy_ctrl[:, D1_FOOT_INDICES] - joint_pos[:, D1_FOOT_INDICES])
        )
        if kd_foot > 0:
            torque[:, D1_FOOT_INDICES] -= kd_foot * joint_vel[:, D1_FOOT_INDICES]

        # clip to actuator ctrl range (used as torque limits)
        ctrl_range = np.asarray(backend.get_actuator_ctrl_range(), dtype=torque.dtype)
        np.clip(torque, ctrl_range[:, 0], ctrl_range[:, 1], out=torque)
        return torque
