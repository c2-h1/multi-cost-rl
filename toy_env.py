"""A small two-cost continuous navigation CMDP, not HumanoidBench.

Costs are hazard occupancy and squared control effort. Episodes have an
intrinsic finite horizon, including zero-cost/reward absorption after success.
Time is observed, so a terminal value of zero is mathematically appropriate.
"""
from __future__ import annotations

import numpy as np


class TwoCostNavigation:
    cost_names = ("hazard", "effort")
    obs_dim = 7
    action_dim = 2

    def __init__(self, num_envs=64, horizon=64, seed=0):
        self.num_envs, self.horizon = num_envs, horizon
        self.rng = np.random.default_rng(seed)
        self.budgets = np.array([0.03, 0.12], dtype=np.float32) * horizon
        self.t = 0

    def reset(self):
        self.pos = np.stack([
            self.rng.uniform(-1.15, -0.95, self.num_envs),
            self.rng.uniform(-0.12, 0.12, self.num_envs)], axis=-1).astype(np.float32)
        self.goal = np.stack([
            self.rng.uniform(0.95, 1.15, self.num_envs),
            self.rng.uniform(-0.12, 0.12, self.num_envs)], axis=-1).astype(np.float32)
        self.active = np.ones(self.num_envs, dtype=bool)
        self.success = np.zeros(self.num_envs, dtype=bool)
        self.t = 0
        return self.observation()

    def observation(self):
        return np.concatenate([
            self.pos, self.goal - self.pos,
            np.linalg.norm(self.pos, axis=-1, keepdims=True),
            np.full((self.num_envs, 1), self.t / self.horizon),
            self.active[:, None]], axis=-1).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.num_envs, 2) or not np.isfinite(action).all():
            raise ValueError("Expected finite (num_envs, 2) actions")
        action = np.clip(action, -1, 1)
        old_distance = np.linalg.norm(self.goal - self.pos, axis=-1)
        active = self.active.copy()
        self.pos = np.clip(self.pos + 0.10 * action * active[:, None], -1.5, 1.5)
        distance = np.linalg.norm(self.goal - self.pos, axis=-1)
        reached = (distance < 0.13) & active
        reward = (2.0 * (old_distance - distance) - 0.015) * active + 2.0 * reached
        costs = np.stack([
            (np.linalg.norm(self.pos, axis=-1) < 0.34) * active,
            np.mean(action ** 2, axis=-1) * active], axis=-1).astype(np.float32)
        self.success |= reached
        self.active &= ~reached
        self.t += 1
        return self.observation(), reward.astype(np.float32), costs

    def diagnostics(self):
        return {"success_rate": float(self.success.mean()),
                "final_distance": float(np.linalg.norm(self.goal - self.pos, axis=-1).mean())}

    def close(self):
        pass

