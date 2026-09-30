"""Synchronous batch adapter for Safety-Gymnasium goal tasks with two costs.

Costs are per-timestep indicators built from the component fields in `info`:
    hazard = 1[cost_hazards > 0]
    vase   = 1[cost_vases_contact > 0 or cost_vases_velocity > 0]
so episode costs count affected timesteps, not incidents, and one timestep may
count toward both components. The environment's own `cost_sum` is not used.

Episodes are complete: every environment runs the native horizon. A genuine
early termination becomes an absorbing state (reward/cost zero, env no longer
stepped). A simulator exception or missing cost field is an error, never a
zero-cost step. Observations append elapsed fraction and the absorbed flag.
"""
from __future__ import annotations

import numpy as np

REQUIRED_FIELDS = ("cost_hazards", "cost_vases_contact", "cost_vases_velocity")


def cost_indicators(info):
    """Map one step's info dict to (c_hazard, c_vase); raises on missing fields."""
    if "cost_exception" in info:
        raise RuntimeError(f"Simulator exception reported by environment: {info}")
    missing = [k for k in REQUIRED_FIELDS if k not in info]
    if missing:
        raise KeyError(f"Missing cost fields {missing}; info keys: {sorted(info)}")
    hazard = float(info["cost_hazards"] > 0)
    vase = float(info["cost_vases_contact"] > 0 or info["cost_vases_velocity"] > 0)
    return hazard, vase


class SafetyGoalBatch:
    cost_names = ("hazard", "vase")

    def __init__(self, num_envs, seed, env_id="SafetyPointGoal2-v0", budgets=(1.0, 1.0)):
        import safety_gymnasium

        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.envs = [safety_gymnasium.make(env_id) for _ in range(num_envs)]
        env = self.envs[0]
        self.horizon = int(env.spec.max_episode_steps)
        if self.horizon != env.unwrapped.task.num_steps:
            raise RuntimeError(f"TimeLimit {self.horizon} != task.num_steps {env.unwrapped.task.num_steps}")
        self.budgets = np.asarray(budgets, dtype=np.float32)
        if self.budgets.shape != (2,) or not (self.budgets > 0).all():
            raise ValueError(f"Need two positive budgets, got {budgets}")
        self.obs_dim = int(np.prod(env.observation_space.shape)) + 2
        self.action_dim = int(np.prod(env.action_space.shape))
        self.metadata = {"env_id": env_id, "horizon": self.horizon,
                         "safety_gymnasium": safety_gymnasium.__version__,
                         "cost_definitions": {"hazard": "1[cost_hazards>0]",
                                              "vase": "1[cost_vases_contact>0 or cost_vases_velocity>0]"},
                         "budgets": self.budgets.tolist(), "budget_units": "affected timesteps per episode",
                         "observation_transform": "native_plus_elapsed_fraction_and_absorbed_flag"}
        self.layout_seeds = np.zeros(num_envs, dtype=np.int64)

    def observation(self):
        return np.concatenate([self.obs, np.full((self.num_envs, 1), self.t / self.horizon),
                               self.absorbed[:, None]], axis=-1).astype(np.float32)

    def reset(self):
        self.layout_seeds = self.rng.integers(0, 2**31 - 1, self.num_envs)
        self.obs = np.stack([env.reset(seed=int(s))[0] for env, s in zip(self.envs, self.layout_seeds)]).astype(np.float32)
        self.absorbed = np.zeros(self.num_envs, dtype=bool)
        self.length = np.zeros(self.num_envs, dtype=np.int64)
        self.goals = np.zeros(self.num_envs, dtype=np.int64)
        self.raw = np.zeros((self.num_envs, len(REQUIRED_FIELDS)))
        self.t = 0
        return self.observation()

    def step(self, actions):
        if self.t >= self.horizon:
            raise RuntimeError("Episode already complete; call reset()")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite {(self.num_envs, self.action_dim)} actions")
        reward = np.zeros(self.num_envs, dtype=np.float32)
        costs = np.zeros((self.num_envs, 2), dtype=np.float32)
        last = self.t + 1 == self.horizon
        for i, env in enumerate(self.envs):
            if self.absorbed[i]:
                continue
            obs, r, _, terminated, truncated, info = env.step(actions[i])
            costs[i] = cost_indicators(info)
            reward[i] = r
            self.obs[i] = obs
            self.raw[i] += [info[k] for k in REQUIRED_FIELDS]
            self.goals[i] += bool(info.get("goal_met", False))
            self.length[i] += 1
            if terminated:
                self.absorbed[i] = True
            elif truncated != last:
                raise RuntimeError(f"Unexpected truncation={truncated} at step {self.t + 1}/{self.horizon}")
        self.t += 1
        if not np.isfinite(self.obs).all() or not np.isfinite(reward).all():
            raise FloatingPointError("Nonfinite observation/reward")
        return self.observation(), reward, costs

    def diagnostics(self):
        return {"goals_per_episode": float(self.goals.mean()),
                "early_termination_rate": float((self.length < self.horizon).mean()),
                "effective_length": float(self.length.mean()),
                **{f"raw_{k}": float(self.raw[:, j].mean()) for j, k in enumerate(REQUIRED_FIELDS)}}

    def close(self):
        for env in self.envs:
            env.close()
