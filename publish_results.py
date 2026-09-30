"""Publish this experiment's results and final model artifacts to W&B.

Uses the existing SDK authentication; never reads or prints credentials.
"""
import json
import os
from pathlib import Path

import wandb


def main():
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_CACHE_DIR", str(Path(".cache/wandb/cache").resolve()))
    os.environ.setdefault("WANDB_DATA_DIR", str(Path(".cache/wandb/data").resolve()))
    root = Path("results")
    run = wandb.init(project="multi-cost-rl", name="session-results-2026-09-20", job_type="analysis",
                     dir=str(root), mode="online", settings=wandb.Settings(init_timeout=40))
    columns = ["environment", "group", "method", "seed", "training_transitions", "return", "all_mean_costs_feasible", "run_url"]
    data = []
    for file in sorted(Path("runs").glob("*/summary.json")):
        summary = json.loads(file.read_text())
        config = json.loads((file.parent / "config.json").read_text())
        data.append([summary["env"], config["group"], summary["method"], summary["seed"],
                     summary["env_steps"], summary["final/return_mean"],
                     summary["final/all_mean_costs_feasible"], summary["wandb"].get("url")])
    run.log({"completed_runs": wandb.Table(columns=columns, data=data)})
    for name in ["toy-comparison-v1-learning-curves.png", "feasible-sets.png", "toy-trajectories.png"]:
        if (root / name).exists():
            run.log({name.removesuffix(".png"): wandb.Image(str(root / name))})
    artifact = wandb.Artifact("multi-cost-rl-session-results", type="experiment-results",
                              description="All pilot results including failed feasibility; no claim of solved humanoid walking")
    for pattern in ["*.md", "*.png", "*.pdf", "*.json", "*.txt"]:
        for file in root.glob(pattern):
            artifact.add_file(str(file), name="results/" + file.name)
    for name in ["README.md", "exp-setup.md", "recent-work.md", "background.md", "training-report.md", "requirements-lock.txt"]:
        if Path(name).exists():
            artifact.add_file(name)
    run.log_artifact(artifact)
    model_names = [p.parent.name for p in sorted(Path("runs").glob("toy-comparison-v1-*/summary.json"))]
    model_names.append("humanoid-pilot-multi-s0")
    for name in model_names:
        directory = Path("runs") / name
        model = wandb.Artifact(name, type="model", description="Final-budget experimental policy; consult summary for constraint violations")
        for filename in ["checkpoint.pt", "config.json", "summary.json", "evaluation.npz"]:
            model.add_file(str(directory / filename))
        for file in (directory / "source").glob("*.py"):
            model.add_file(str(file), name="source/" + file.name)
        run.log_artifact(model)
    url = run.url
    run.finish()
    (root / "wandb-analysis.json").write_text(json.dumps({"url": url}, indent=2))
    print(url)


if __name__ == "__main__":
    main()
