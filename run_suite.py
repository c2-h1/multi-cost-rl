"""Run the same fixed toy budget for every method and seed; fail on errors."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from train import METHODS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--group", default="toy-comparison-v1")
    args = p.parse_args()
    Path("runs").mkdir(exist_ok=True)
    manifest = {"group": args.group, "updates": args.updates, "seeds": args.seeds,
                "methods": args.methods, "runs": []}
    manifest_path = Path("runs") / f"{args.group}-manifest.json"
    for seed in args.seeds:
        for method in args.methods:
            output = Path("runs") / f"{args.group}-{method}-s{seed}"
            if (output / "summary.json").exists():
                print(f"Already complete: {output}", flush=True)
                manifest["runs"].append(str(output))
                continue
            if output.exists():
                raise RuntimeError(f"Incomplete run exists: {output}; choose another group or inspect it")
            command = [sys.executable, "train.py", "--method", method, "--seed", str(seed),
                       "--updates", str(args.updates), "--wandb-mode", args.wandb_mode,
                       "--group", args.group, "--output", str(output)]
            print("Starting " + " ".join(command), flush=True)
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((output / "summary.json").read_text())
            print(json.dumps({k: summary[k] for k in ["method", "seed", "env_steps", "final/return_mean",
                                                     "final/cost_hazard", "final/cost_effort",
                                                     "final/success_rate", "final/all_mean_costs_feasible"]}), flush=True)
            manifest["runs"].append(str(output))
            manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

