"""Small synchronous batch adapter for the actual HumanoidBench simulator.

The pilot's intrinsic horizon is deliberately shorter than the benchmark.
After a physical termination, the remaining horizon is absorbing with fall-risk
cost one, other costs zero and reward zero. This prevents falling early from
escaping the fall-risk budget. Time and absorption are included in observation.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import numpy as np


class HumanoidBatch:
    def __init__(self, num_envs, horizon, seed, env_id="h1-walk-v0"):
        root = Path(__file__).resolve().parent / "external/humanoid-bench"
        if not root.is_dir():
            raise FileNotFoundError("Clone the upstream repository into external/humanoid-bench; see README.md")
        sys.path.insert(0, str(root))
        import humanoid_bench  # noqa: F401 registers task IDs
        import gymnasium as gym
        from humanoid_costs import HumanoidCostWrapper, HumanoidCostConfig

        self.num_envs, self.horizon = num_envs, horizon
        self.rng = np.random.default_rng(seed)
        config = HumanoidCostConfig()
        self.envs = [HumanoidCostWrapper(gym.make(env_id, render_mode=None), config=config)
                     for _ in range(num_envs)]
        self.cost_names = self.envs[0].cost_names
        self.budgets = np.array([.05, .05, .2], dtype=np.float32) * horizon
        self.obs_dim = int(np.prod(self.envs[0].observation_space.shape)) + 2
        self.action_dim = int(np.prod(self.envs[0].action_space.shape))
        self.physical_steps_total = 0
        self.metadata = {"upstream_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                         "env_id": env_id, "cost_config": asdict(config),
                         "budget_rates": (self.budgets / horizon).tolist(),
                         "budget_feasibility_established": False,
                         "physical_termination_padding": [1., 0., 0.],
                         "observation_transform": "raw_state_plus_elapsed_fraction_and_absorbed_flag"}

    def observation(self):
        return np.concatenate([self.obs,
            np.full((self.num_envs, 1), self.t / self.horizon),
            self.absorbed[:, None]], axis=-1).astype(np.float32)

    def reset(self):
        self.obs = np.stack([env.reset(seed=int(self.rng.integers(0, 2**31-1)))[0]
                             for env in self.envs]).astype(np.float32)
        self.absorbed = np.zeros(self.num_envs, dtype=bool)
        self.fell = np.zeros(self.num_envs, dtype=bool)
        self.t = 0
        self.episode_physical_steps = 0
        self.start_x = np.array([e.unwrapped.data.qpos[0] for e in self.envs])
        return self.observation()

    def step(self, actions):
        reward = np.zeros(self.num_envs, dtype=np.float32)
        costs = np.zeros((self.num_envs, len(self.cost_names)), dtype=np.float32)
        for i, env in enumerate(self.envs):
            if self.absorbed[i]:
                costs[i, 0] = float(self.fell[i])
                continue
            obs, r, terminated, truncated, info = env.step(actions[i])
            self.obs[i] = obs
            reward[i] = r
            costs[i] = info["costs"]
            self.physical_steps_total += 1
            self.episode_physical_steps += 1
            if terminated:
                self.absorbed[i] = True
                self.fell[i] = True
                costs[i, 0] = 1
            elif truncated and self.t + 1 < self.horizon:
                raise RuntimeError("Upstream TimeLimit precedes requested intrinsic horizon; use horizon <= upstream limit")
        self.t += 1
        if not np.isfinite(self.obs).all() or not np.isfinite(costs).all():
            raise FloatingPointError("Nonfinite MuJoCo observation/cost")
        return self.observation(), reward, costs

    def diagnostics(self):
        final_x = np.array([e.unwrapped.data.qpos[0] for e in self.envs])
        return {"fall_rate": float(self.fell.mean()),
                "forward_displacement": float((final_x - self.start_x).mean()),
                "physical_steps_total": self.physical_steps_total,
                "physical_fraction": self.episode_physical_steps / (self.num_envs * self.horizon)}

    def close(self):
        for env in self.envs:
            env.close()
