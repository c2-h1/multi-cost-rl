"""Check that toy budgets admit a goal-reaching reference controller.

This controller is a feasibility diagnostic, not a trained baseline or a safety
certificate. It is never used to imitate/pretrain or choose test checkpoints.
"""
import json
from pathlib import Path
import numpy as np
from toy_env import TwoCostNavigation


def main():
    env = TwoCostNavigation(512, seed=9876)
    env.reset()
    total_cost = np.zeros((env.num_envs, 2))
    total_reward = np.zeros(env.num_envs)
    phase = np.zeros(env.num_envs, dtype=int)
    for _ in range(env.horizon):
        first = np.tile([-.50, .55], (env.num_envs, 1))
        second = np.tile([.50, .55], (env.num_envs, 1))
        target = np.where((phase == 0)[:, None], first,
                          np.where((phase == 1)[:, None], second, env.goal))
        delta = target - env.pos
        phase += ((np.linalg.norm(delta, axis=-1) < .08) & (phase < 2)).astype(int)
        action = delta / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-6) * .55
        _, reward, cost = env.step(action)
        total_reward += reward
        total_cost += cost
    result = {"type": "handwritten_feasibility_controller_not_RL", "episodes": env.num_envs,
              "seed": 9876, "budgets": env.budgets.tolist(),
              "return": float(total_reward.mean()), "costs": total_cost.mean(0).tolist(),
              **env.diagnostics()}
    Path("results").mkdir(exist_ok=True)
    Path("results/toy-budget-calibration.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
