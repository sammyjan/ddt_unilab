from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.d1.base import (
    D1BaseCfg,
    D1BaseEnv,
    D1_FOOT_INDICES,
    D1_HIP_INDICES,
    NUM_D1_ACTIONS,
)


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.60]


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float


@dataclass
class JoystickSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    gravity = "upvector"


@dataclass
class D1DomainRandConfig(DomainRandConfig):
    randomize_init_yaw: bool = True
    init_z_range: list[float] = field(default_factory=lambda: [0.0, 0.2])
    init_roll_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    init_pitch_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    init_yaw_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])


@registry.envcfg("D1Flat")
@dataclass
class D1FlatCfg(D1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "d1" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)
    domain_rand: D1DomainRandConfig = field(default_factory=D1DomainRandConfig)


class D1FlatDomainRandomizationProvider(LocomotionDRProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> Any:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        yaw = np.random.uniform(-np.pi, np.pi, size=(num_reset,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        qvel[:, 0:6] = np.random.uniform(-0.5, 0.5, size=(num_reset, 6))

        commands = self._sample_commands(env, num_reset)
        info_updates = {
            "commands": commands,
            "current_actions": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "last_actions": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
        from unilab.dr.dr_utils import zero_actions
        info_updates["current_actions"] = zero_actions(num_reset, env._num_action)
        info_updates["last_actions"] = zero_actions(num_reset, env._num_action)
        from unilab.dr import ResetPlan
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel),
        )


@registry.env("D1Flat", sim_backend="mujoco")
@registry.env("D1Flat", sim_backend="motrix")
class D1FlatTask(D1BaseEnv):
    _cfg: D1FlatCfg

    def __init__(self, cfg: D1FlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        self._init_reward_functions()
        self._init_domain_randomization(D1FlatDomainRandomizationProvider())
        self._backend.set_pre_step_control(self._pre_step_motor_control)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + commands(3) + dof_diff(16) + dof_vel(16) + last_actions(16) = 57
        return {"obs": 57, "critic": 60}

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "orientation": rewards.orientation,
            "upward": rewards.upward,
            "similar_to_default": rewards.similar_to_default,
            "dof_acc": self._reward_dof_acc,
            "torques": self._reward_torques,
            "collision": self._reward_collision,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.gravity)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        base_height = self._backend.get_base_pos()[:, 2]
        return (gravity[:, 2] <= 0.5) | (base_height < 0)

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))

        obs = np.concatenate(
            [
                noisy_gyro,
                -noisy_gravity,
                command,
                noisy_diff,
                noisy_dof_vel,
                last_actions,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [gyro, -gravity, command, diff, dof_vel, last_actions, linvel],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def _compute_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=cfg.tracking_sigma,
            base_height_target=cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
        )
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _reward_dof_acc(self, ctx: RewardContext) -> np.ndarray:
        # Penalize dof accelerations
        if not hasattr(self, "_last_dof_vel"):
            self._last_dof_vel = np.zeros((self._num_envs, NUM_D1_ACTIONS), dtype=get_global_dtype())
        dof_acc = (ctx.info.get("dof_vel", self._last_dof_vel) - self._last_dof_vel) / self._cfg.ctrl_dt
        self._last_dof_vel = ctx.info.get("dof_vel", self._last_dof_vel).copy()
        return np.sum(np.square(dof_acc), axis=1)

    def _reward_torques(self, ctx: RewardContext) -> np.ndarray:
        torques = ctx.info.get("torques", np.zeros((self._num_envs, NUM_D1_ACTIONS), dtype=get_global_dtype()))
        return np.sum(np.square(torques), axis=1)

    def _reward_collision(self, ctx: RewardContext) -> np.ndarray:
        # Placeholder: needs contact force sensors configured in MJCF
        return np.zeros(self._num_envs, dtype=get_global_dtype())
