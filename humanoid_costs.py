"""Transparent simulator safety proxies for pinned HumanoidBench H1 walking.

These are research specifications, not validated hardware safety thresholds.
Signals are sampled at control-step endpoints (not every physics substep).
The wrapper leaves reward/termination intact. A trainer using fixed-horizon
costs must separately account for the absorbing state after physical failure.
"""
from dataclasses import asdict, dataclass

import gymnasium as gym
import mujoco
import numpy as np


@dataclass(frozen=True)
class HumanoidCostConfig:
    head_height_min: float = 1.35
    torso_angle_max_degrees: float = 45.0
    joint_margin_fraction: float = 0.05


class HumanoidCostWrapper(gym.Wrapper):
    """Expose three nonnegative raw costs in ``info['costs']``.

    Intended initial use is H1/H1Hand walking. Height thresholds are task and
    morphology specific: do not reuse unmodified for crawling, G1 or Digit.
    Model force limits must be present; missing limits fail loudly rather than
    silently interpreting position control commands as torques.
    """

    cost_names = ("fall_risk", "joint_limit", "actuator_load")
    default_budget_rates = np.array([0.05, 0.05, 0.20], dtype=np.float32)

    def __init__(self, env, config=None):
        super().__init__(env)
        self.config = config or HumanoidCostConfig()
        if not (0 < self.config.joint_margin_fraction < 0.5):
            raise ValueError("joint_margin_fraction must lie in (0, 0.5)")
        if self.config.head_height_min <= 0:
            raise ValueError("head_height_min must be positive")
        if not (0 < self.config.torso_angle_max_degrees < 90):
            raise ValueError("torso_angle_max_degrees must lie in (0, 90)")
        model = self.unwrapped.model
        scalar_joint = np.isin(model.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE,
                                              mujoco.mjtJoint.mjJNT_SLIDE])
        # Select only robot joints, excluding free roots and later object DoFs.
        robot_qpos_end = self.unwrapped.robot.dof
        mask = scalar_joint & model.jnt_limited.astype(bool) & (
            model.jnt_qposadr < robot_qpos_end)
        self.joint_ids = np.flatnonzero(mask)
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.joint_bounds = model.jnt_range[self.joint_ids].copy()
        if not len(self.joint_ids) or np.any(np.diff(self.joint_bounds, axis=1) <= 0):
            raise ValueError("Expected nonempty robot joints with finite positive ranges")
        self.force_scales = np.abs(model.actuator_forcerange).max(axis=1)
        if (not model.nu or not np.all(model.actuator_forcelimited)
                or not np.all(np.isfinite(self.force_scales))
                or np.any(self.force_scales <= 0)):
            raise ValueError("Every actuator needs a finite enforced force limit")
        self.upright_min = np.cos(np.deg2rad(self.config.torso_angle_max_degrees))

    def safety_signals(self, terminated=False):
        """Read current physics state; no mutation or extra simulation."""
        base = self.unwrapped
        data = base.data
        head_height = float(base.robot.head_height())
        torso_upright = float(base.robot.torso_upright())
        fall_risk = float(terminated or head_height < self.config.head_height_min
                          or torso_upright < self.upright_min)
        q = data.qpos[self.qpos_ids]
        lower, upper = self.joint_bounds.T
        relative_margin = np.minimum(q - lower, upper - q) / (upper - lower)
        joint_limit = float(np.mean(relative_margin < self.config.joint_margin_fraction))
        normalized_force = data.actuator_force / self.force_scales
        actuator_load = float(np.mean(np.square(normalized_force)))
        costs = np.asarray([fall_risk, joint_limit, actuator_load], dtype=np.float32)
        if not np.all(np.isfinite(costs)):
            raise FloatingPointError("Non-finite simulator safety signals")
        # Actuator force times transmission velocity is mechanical power;
        # summing absolute powers is not signed net power or battery draw.
        absolute_mechanical_power = float(np.sum(np.abs(
            data.actuator_force * data.actuator_velocity)))
        metrics = {
            "safety/head_height_m": head_height,
            "safety/torso_upright": torso_upright,
            "safety/min_joint_margin_fraction": float(relative_margin.min()),
            "safety/max_actuator_load_fraction": float(np.abs(normalized_force).max()),
            "safety/absolute_mechanical_power_w": absolute_mechanical_power,
            "safety/physical_termination": float(terminated),
            **{f"cost/{name}": float(value) for name, value in zip(self.cost_names, costs)},
        }
        return costs, metrics

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        costs, metrics = self.safety_signals(terminated=terminated)
        return obs, reward, terminated, truncated, {**info, **metrics, "costs": costs}

    def cost_metadata(self):
        return {"names": list(self.cost_names), "thresholds": asdict(self.config),
                "budget_rates": self.default_budget_rates.tolist(),
                "sampling": "control-step endpoint", "units": "dimensionless",
                "safety_claim": "unvalidated simulator proxies"}
