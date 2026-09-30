"""Memory budgeting so runs fail loudly instead of OOM-killing the host.

Measured on SafetyPointGoal2-v0 (Safety-Gymnasium 1.0.0, MuJoCo 2.3.3): each
env holds ~185 MB (dense constraint buffers for njmax=3000), flat over time;
importing torch costs ~0.5 GB per process. One baseline_ppo.py run (30 envs,
4 spawned workers) peaks at ~8.8 GiB on CPU; six at once exhausted 62 GB.

`available_bytes` honors cgroup limits (SLURM --mem, containers) as well as
/proc/meminfo, so the same check works on a laptop and on the cluster.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

GIB = 2**30
ENV_GIB = 0.2          # one Safety-Gymnasium env
PROCESS_GIB = 0.7      # python + torch + safety_gymnasium imports
CUDA_GIB = 1.5         # host-side CUDA context


def estimate_gib(num_envs, processes=1, cuda=False):
    return num_envs * ENV_GIB + processes * PROCESS_GIB + (CUDA_GIB if cuda else 0.0)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cgroup_headroom():
    """Bytes left under this process's cgroup memory limit, or None if unlimited."""
    for line in (_read("/proc/self/cgroup") or "").splitlines():
        _, controllers, rel = line.split(":", 2)
        if controllers == "":  # cgroup v2
            base = Path("/sys/fs/cgroup") / rel.lstrip("/")
            limit, usage = _read(base / "memory.max"), _read(base / "memory.current")
        elif "memory" in controllers.split(","):  # cgroup v1
            base = Path("/sys/fs/cgroup/memory") / rel.lstrip("/")
            limit, usage = _read(base / "memory.limit_in_bytes"), _read(base / "memory.usage_in_bytes")
        else:
            continue
        if limit and usage and limit != "max" and int(limit) < 2**60:
            return int(limit) - int(usage)
    return None


def available_bytes():
    meminfo = _read("/proc/meminfo")
    if meminfo is None:  # not Linux: no check
        return None
    host = next(int(l.split()[1]) * 1024 for l in meminfo.splitlines() if l.startswith("MemAvailable:"))
    cgroup = _cgroup_headroom()
    return host if cgroup is None else min(host, cgroup)


def check(needed_gib, what, reserve_gib=2.0):
    """Exit before allocating if `needed_gib` (+ reserve) does not fit."""
    available = available_bytes()
    if available is None:
        return
    if needed_gib + reserve_gib > available / GIB:
        sys.exit(f"Refusing to start {what}: needs ~{needed_gib:.1f} GiB + {reserve_gib:.1f} GiB reserve, "
                 f"only {available / GIB:.1f} GiB available. Run fewer jobs at once or use fewer envs "
                 f"(--skip-memory-check to override).")


def start_watchdog(min_free_gib=1.5, interval=5.0):
    """Kill this process if free memory drops below `min_free_gib`.

    A last resort for when something else on the machine eats memory: losing
    one run is better than the kernel OOM-killer taking down the host.
    Spawned workers see their pipe close and exit on their own.
    """
    if available_bytes() is None:
        return

    def watch():
        while True:
            time.sleep(interval)
            available = available_bytes() / GIB
            if available < min_free_gib:
                print(f"MEMORY WATCHDOG: only {available:.2f} GiB free (< {min_free_gib} GiB); aborting run",
                      file=sys.stderr, flush=True)
                os._exit(137)

    threading.Thread(target=watch, daemon=True, name="memory-watchdog").start()
