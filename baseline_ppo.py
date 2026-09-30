"""Safety Gym paper baseline (Ray, Achiam & Amodei 2019): PPO and PPO-Lagrangian.

PyTorch port of openai/safety-starter-agents `run_polopt_agent` with the
`scripts/experiment.py` settings (Point robots): 1e7 steps, 30,000 steps/epoch,
separate 256x256 tanh pi/V/Vc MLPs, gamma=lam=0.99/0.97 for reward and cost,
80 pi iterations (lr 3e-4, clip 0.2, early stop at KL > 1.2*0.01), 80 value
iterations (lr 1e-3), cost_lim=25 on the environment's scalar indicator cost.
PPO-Lagrangian: penalty=softplus(p), init 1, one Adam step (lr 5e-2) per epoch
on -p*(mean EpCost - 25) before the policy step; objective
(surr_adv - penalty*surr_cost)/(1+penalty). Deliberately independent of train.py.

Differences from the paper that cannot be removed: Safety-Gymnasium 1.0.0 +
MuJoCo 2.3.3 instead of safety_gym + mujoco-py, PyTorch instead of TF1/MPI.
Environment stepping is spread across worker processes (no effect on the
algorithm: every epoch is still 30 x 1000-step episodes).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import memguard


def worker(remote, env_id, num_envs, seed):
    import safety_gymnasium
    envs = [safety_gymnasium.make(env_id) for _ in range(num_envs)]
    rng = np.random.default_rng(seed)
    reset = lambda env: env.reset(seed=int(rng.integers(0, 2**31 - 1)))[0]
    try:
        while True:
            command, data = remote.recv()
            if command == "reset":
                remote.send(np.stack([reset(env) for env in envs]))
            elif command == "step":
                out = []
                for env, action in zip(envs, data):
                    obs, reward, cost, terminated, truncated, info = env.step(action)
                    if "cost_exception" in info:
                        raise RuntimeError(f"Simulator exception: {info}")
                    final = obs
                    if terminated or truncated:
                        obs = reset(env)
                    out.append((obs, final, reward, cost, terminated, truncated))
                remote.send(out)
            elif command == "close":
                break
    finally:
        for env in envs:
            env.close()
        remote.close()


class VecEnv:
    """Lockstep envs in worker processes; auto-reset, returning the final obs."""

    def __init__(self, env_id, num_envs, num_workers, seed):
        ctx = mp.get_context("spawn")
        sizes = [len(c) for c in np.array_split(np.arange(num_envs), num_workers)]
        self.remotes, self.procs = [], []
        for i, size in enumerate(sizes):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(child, env_id, size, seed * 1000 + i), daemon=True)
            proc.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(proc)
        self.splits = np.cumsum(sizes)[:-1]

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.concatenate([remote.recv() for remote in self.remotes]).astype(np.float32)

    def step(self, actions):
        for remote, chunk in zip(self.remotes, np.split(actions, self.splits)):
            remote.send(("step", chunk))
        rows = [row for remote in self.remotes for row in remote.recv()]
        obs, final, reward, cost, terminated, truncated = (np.array(x) for x in zip(*rows))
        return (obs.astype(np.float32), final.astype(np.float32), reward.astype(np.float32),
                cost.astype(np.float32), terminated, truncated)

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for proc in self.procs:
            proc.join(timeout=10)


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.xavier_uniform_(linear.weight)  # TF1 dense default (glorot uniform, zero bias)
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()] if i < len(sizes) - 2 else [linear]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.pi = mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.v = mlp([obs_dim, *hidden, 1])
        self.vc = mlp([obs_dim, *hidden, 1])

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def values(self, obs):
        return self.v(obs).squeeze(-1), self.vc(obs).squeeze(-1)


def discounted_targets(signal, values, next_values, ends, gamma, lam):
    """Per-env GAE advantages and discounted reward-to-go, both reset at episode ends.

    next_values[t] is V(s_{t+1}) inside an episode, V(final obs) at a timeout or
    the epoch cut, and 0 at a genuine termination (reference `finish_path`).
    """
    adv, ret = np.zeros_like(signal), np.zeros_like(signal)
    last_adv = np.zeros(signal.shape[1], dtype=np.float32)
    last_ret = np.zeros(signal.shape[1], dtype=np.float32)
    for t in reversed(range(len(signal))):
        cut = ends[t]
        last_adv = np.where(cut, 0, last_adv)
        last_ret = np.where(cut, next_values[t], last_ret)
        delta = signal[t] + gamma * next_values[t] - values[t]
        last_adv = delta + gamma * lam * last_adv
        last_ret = signal[t] + gamma * last_ret
        adv[t], ret[t] = last_adv, last_ret
    return adv, ret


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=("ppo", "ppo_lagrangian"), required=True)
    p.add_argument("--env-id", default="SafetyPointGoal2-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--steps-per-epoch", type=int, default=30_000)
    p.add_argument("--max-ep-len", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", required=True)
    p.add_argument("--skip-memory-check", action="store_true")
    # Reference defaults; exposed only for smoke tests.
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.97)
    p.add_argument("--cost-lim", type=float, default=25.0)
    p.add_argument("--target-kl", type=float, default=0.01)
    p.add_argument("--pi-iters", type=int, default=80)
    p.add_argument("--vf-iters", type=int, default=80)
    args = p.parse_args()

    num_envs = args.steps_per_epoch // args.max_ep_len
    assert num_envs * args.max_ep_len == args.steps_per_epoch
    # ~30 x 0.2 GB of envs + one torch import per worker: ~8-10 GiB per run.
    if not args.skip_memory_check:
        memguard.check(memguard.estimate_gib(num_envs, 1 + args.workers, args.device == "cuda"), "baseline_ppo.py")
    memguard.start_watchdog()
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "baseline_ppo.py")

    epochs = args.total_steps // args.steps_per_epoch
    envs = VecEnv(args.env_id, num_envs, args.workers, args.seed)
    obs = envs.reset()
    obs_dim, act_dim = obs.shape[1], 2

    ac = ActorCritic(obs_dim, act_dim).to(device)
    pi_opt = torch.optim.Adam([*ac.pi.parameters(), ac.log_std], lr=3e-4, eps=1e-8)
    vf_opt = torch.optim.Adam([*ac.v.parameters(), *ac.vc.parameters()], lr=1e-3, eps=1e-8)
    lagrangian = args.algo == "ppo_lagrangian"
    penalty_param = torch.tensor(float(np.log(np.expm1(1.0))), requires_grad=True)
    penalty_opt = torch.optim.Adam([penalty_param], lr=5e-2, eps=1e-8)

    import safety_gymnasium, mujoco
    config = {**vars(args), "epochs": epochs, "num_envs": num_envs, "clip_ratio": 0.2, "pi_lr": 3e-4,
              "vf_lr": 1e-3, "penalty_init": 1.0, "penalty_lr": 5e-2, "hidden": [256, 256],
              "kl_margin": 1.2, "log_std_init": -0.5, "python": platform.python_version(),
              "torch": torch.__version__, "mujoco": mujoco.__version__,
              "safety_gymnasium": safety_gymnasium.__version__}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    log = (output / "progress.jsonl").open("w", buffering=1)

    ep_ret = np.zeros(num_envs)
    ep_cost = np.zeros(num_envs)
    ep_len = np.zeros(num_envs, dtype=int)
    cumulative_cost = 0.0
    start = time.time()
    T = args.max_ep_len
    for epoch in range(epochs):
        buf = {k: np.zeros((T, num_envs, *s), dtype=np.float32) for k, s in
               [("obs", (obs_dim,)), ("act", (act_dim,)), ("logp", ()), ("rew", ()), ("cost", ()),
                ("v", ()), ("vc", ()), ("next_v", ()), ("next_vc", ())]}
        ends = np.zeros((T, num_envs), dtype=bool)
        finished = {"ret": [], "cost": [], "len": []}
        for t in range(T):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device)
                dist = ac.dist(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
                v, vc = ac.values(obs_t)
            action = action.cpu().numpy()
            next_obs, final_obs, reward, cost, terminated, truncated = envs.step(action)
            for key, value in [("obs", obs), ("act", action), ("logp", logp.cpu().numpy()), ("rew", reward),
                               ("cost", cost), ("v", v.cpu().numpy()), ("vc", vc.cpu().numpy())]:
                buf[key][t] = value
            cumulative_cost += float(cost.sum())
            ep_ret += reward
            ep_cost += cost
            ep_len += 1
            timeout = truncated | (ep_len == args.max_ep_len)
            done = terminated | timeout
            cut = done | (t == T - 1)
            # Bootstrap from the final observation unless the episode genuinely terminated.
            boot_obs = np.where(done[:, None], final_obs, next_obs)
            with torch.no_grad():
                nv, nvc = ac.values(torch.as_tensor(boot_obs, device=device))
            dead = terminated & ~timeout
            buf["next_v"][t] = np.where(dead, 0, nv.cpu().numpy())
            buf["next_vc"][t] = np.where(dead, 0, nvc.cpu().numpy())
            ends[t] = cut
            for i in np.flatnonzero(done):
                finished["ret"].append(ep_ret[i])
                finished["cost"].append(ep_cost[i])
                finished["len"].append(ep_len[i])
                ep_ret[i] = ep_cost[i] = ep_len[i] = 0
            obs = next_obs
        if not (ep_len == 0).all():
            print(f"Warning: {int((ep_len > 0).sum())} trajectories cut off by epoch", flush=True)

        adv, ret = discounted_targets(buf["rew"], buf["v"], buf["next_v"], ends, args.gamma, args.lam)
        cadv, cret = discounted_targets(buf["cost"], buf["vc"], buf["next_vc"], ends, args.gamma, args.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalize reward advantage
        cadv = cadv - cadv.mean()                        # center, do not rescale, cost advantage
        flat = lambda x: torch.as_tensor(x.reshape(-1, *x.shape[2:]), device=device)
        o, a, logp_old = flat(buf["obs"]), flat(buf["act"]), flat(buf["logp"])
        adv_t, cadv_t, ret_t, cret_t = flat(adv), flat(cadv), flat(ret), flat(cret)
        with torch.no_grad():
            old = ac.dist(o)
            old_mu, old_std = old.mean.clone(), old.stddev.clone()

        ep_cost_mean = float(np.mean(finished["cost"]))
        if lagrangian:
            penalty_opt.zero_grad()
            (-penalty_param * (ep_cost_mean - args.cost_lim)).backward()
            penalty_opt.step()
        penalty = float(nn.functional.softplus(penalty_param.detach())) if lagrangian else 0.0

        for i in range(args.pi_iters):
            dist = ac.dist(o)
            ratio = (dist.log_prob(a).sum(-1) - logp_old).exp()
            min_adv = torch.where(adv_t > 0, 1.2 * adv_t, 0.8 * adv_t)
            surr_adv = torch.minimum(ratio * adv_t, min_adv).mean()
            surr_cost = (ratio * cadv_t).mean()
            objective = (surr_adv - penalty * surr_cost) / (1 + penalty) if lagrangian else surr_adv
            kl = torch.distributions.kl_divergence(dist, torch.distributions.Normal(old_mu, old_std)).sum(-1).mean()
            pi_opt.zero_grad()
            (-objective).backward()
            pi_opt.step()
            if kl.item() > 1.2 * args.target_kl:
                break
        stop_iter = i
        for _ in range(args.vf_iters):
            v, vc = ac.values(o)
            loss_v, loss_vc = (ret_t - v).square().mean(), (cret_t - vc).square().mean()
            vf_opt.zero_grad()
            (loss_v + loss_vc).backward()
            vf_opt.step()

        steps = (epoch + 1) * args.steps_per_epoch
        row = {"epoch": epoch, "TotalEnvInteracts": steps, "EpRet": float(np.mean(finished["ret"])),
               "EpRetStd": float(np.std(finished["ret"])), "EpCost": ep_cost_mean,
               "EpCostStd": float(np.std(finished["cost"])), "EpLen": float(np.mean(finished["len"])),
               "NumEpisodes": len(finished["ret"]), "CumulativeCost": cumulative_cost,
               "CostRate": cumulative_cost / steps, "Penalty": penalty, "KL": kl.item(),
               "StopIter": stop_iter, "LossV": loss_v.item(), "LossVC": loss_vc.item(),
               "SurrCost": surr_cost.item(), "Entropy": float(ac.dist(o[:1]).entropy().sum()),
               "Time": time.time() - start}
        log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if epoch % 50 == 0 or epoch == epochs - 1:
            torch.save({"model": ac.state_dict(), "penalty_param": penalty_param.detach(), "config": config,
                        "epoch": epoch}, output / "model.pt")

    rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    last5 = rows[-5:]
    # Paper footnote 3: return/cost averaged over the last five epochs, cost rate from the final epoch.
    summary = {"algo": args.algo, "seed": args.seed, "env_id": args.env_id, "total_steps": rows[-1]["TotalEnvInteracts"],
               "EpRet_last5": float(np.mean([r["EpRet"] for r in last5])),
               "EpCost_last5": float(np.mean([r["EpCost"] for r in last5])),
               "CostRate_final": rows[-1]["CostRate"], "Penalty_final": rows[-1]["Penalty"],
               "elapsed_seconds": time.time() - start}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("COMPLETE " + json.dumps(summary), flush=True)
    envs.close()


if __name__ == "__main__":
    main()
