"""Stream already-running runs' JSONL logs to W&B without touching the trainers.

Follows every `<root>/*/progress.jsonl` (baseline_ppo.py) or `metrics.jsonl`
(train.py), one W&B run per directory with a stable id, so restarting this
script resumes the same W&B runs instead of duplicating them. Exits once every
run directory has summary.json and all rows are logged.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="runs/paper-repro")
    p.add_argument("--project", default="multi-cost-rl")
    p.add_argument("--entity", default=None)
    p.add_argument("--group", default=None, help="defaults to the root directory name")
    p.add_argument("--mode", default="online", choices=("online", "offline"))
    p.add_argument("--interval", type=float, default=60)
    args = p.parse_args()
    root = Path(args.root).resolve()
    group = args.group or root.name
    for key, sub in [("WANDB_CACHE_DIR", "cache"), ("WANDB_CONFIG_DIR", "config"), ("WANDB_DATA_DIR", "data")]:
        os.environ.setdefault(key, str(Path(".cache/wandb", sub).resolve()))
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb

    state = {}  # run dir -> (wandb run, rows logged)
    while True:
        dirs = sorted(d for d in root.iterdir() if d.is_dir() and
                      ((d / "progress.jsonl").exists() or (d / "metrics.jsonl").exists()))
        for d in dirs:
            log = d / "progress.jsonl" if (d / "progress.jsonl").exists() else d / "metrics.jsonl"
            if d not in state:
                config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
                run_id = hashlib.sha1(f"{group}/{d.name}".encode()).hexdigest()[:12]
                run = wandb.init(project=args.project, entity=args.entity, group=group, name=d.name,
                                 id=run_id, resume="allow", config=config, mode=args.mode,
                                 dir=str(root), reinit="create_new")
                state[d] = [run, 0]
            run, done = state[d]
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            for row in rows[done:]:
                step = int(row.get("TotalEnvInteracts", row.get("env_steps", 0)))
                run.log({k: v for k, v in row.items() if isinstance(v, (int, float, bool))}, step=step)
            state[d][1] = len(rows)
            summary = d / "summary.json"
            if summary.exists() and run is not None:
                run.summary.update(json.loads(summary.read_text()))
                run.finish()
                state[d][0] = None
        if dirs and all(state[d][0] is None for d in dirs):
            print("All runs complete and logged.", flush=True)
            return
        print(time.strftime("%Y-%m-%d %H:%M:%S %Z"), {d.name: state[d][1] for d in dirs}, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
