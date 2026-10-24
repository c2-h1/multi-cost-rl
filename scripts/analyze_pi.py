"""E11 (PI multipliers) analysis: late-training windows, cumulative excess, validation tables, figures.
Never trains or evaluates; reads metrics.jsonl and validation/ck*.npz. Works on incomplete runs.

    python scripts/analyze_pi.py --output reports/spg-v2-pi

Writes <output>/analysis.md and the figures pi-curves.png, pi-validation.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = {"A": "spg-v2-pi-a-ki04", "B": "spg-v2-pi-b-ki02", "C": "spg-v2-pi-c-ki02-kp1", "D": "spg-v2-pi-d-ki02-kp4"}
LABEL = {"A": "A  K_I 0.04", "B": "B  K_I 0.02", "C": "C  K_I 0.02, K_P 1", "D": "D  K_I 0.02, K_P 4"}
SEED_COLOR = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference categorical slots 1-3
COND_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]  # slots 1-4 (validation figure: color = condition)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"
WINDOWS = [(4.0e6, 4.25e6), (4.25e6, 4.5e6), (4.5e6, 4.75e6), (4.75e6, 5.0e6)]
VALIDATION = ["004000000", "004252000", "004500000", "004752000", "005000000"]
BUDGETS = np.array([31.9, 73.9])
COSTS = ("hazard", "vase")


def runs(root):
    out = {}
    for cond, group in CONDITIONS.items():
        for run in sorted((root / group).glob("multi-s?")):
            rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()] \
                if (run / "metrics.jsonl").exists() else []
            rows = [r for r in rows if "train/return_mean" in r]
            if rows:
                out[(cond, int(run.name[-1]))] = (run, rows)
    return out


def col(rows, key):
    return np.array([r[key] for r in rows], float)


def fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def training_tables(data):
    lines = ["## Training: late windows (0.25M steps each, training batches of 4 episodes)", "",
             "Window mean of the per-update values; `noise` = mean within-window std of the raw per-update x "
             "(batch noise); `swing` = max − min of the 4 window means (slow variation). λ slope is a least-squares "
             "fit over 4–5M steps, per 1M steps.", "",
             "| cond | seed | updates | x_h windows | x_v windows | λ_h windows | λ_v windows | return windows | "
             "x noise (h / v) | x swing (h / v) | λ slope (h / v) |", "|" + "---|" * 11]
    for (cond, seed), (run, rows) in sorted(data.items()):
        steps = col(rows, "env_steps")
        win = {}
        for key in ("train/normalized_hazard", "train/normalized_vase", "dual/lambda_0", "dual/lambda_1",
                    "train/return_mean"):
            vals = col(rows, key)
            win[key] = [vals[(steps > lo) & (steps <= hi)] for lo, hi in WINDOWS]
        means = {k: [v.mean() if len(v) else np.nan for v in w] for k, w in win.items()}
        noise = [np.nanmean([v.std() if len(v) > 1 else np.nan for v in win[k]])
                 for k in ("train/normalized_hazard", "train/normalized_vase")]
        swing = [np.nanmax(means[k]) - np.nanmin(means[k]) if not np.all(np.isnan(means[k])) else np.nan
                 for k in ("train/normalized_hazard", "train/normalized_vase")]
        late = steps > 4.0e6
        slope = [np.polyfit(steps[late] / 1e6, col(rows, k)[late], 1)[0] if late.sum() > 10 else np.nan
                 for k in ("dual/lambda_0", "dual/lambda_1")]
        w = lambda k, d=2: " / ".join(fmt(x, d) for x in means[k])
        lines.append(f"| {cond} | {seed} | {len(rows)} | {w('train/normalized_hazard')} | {w('train/normalized_vase')} | "
                     f"{w('dual/lambda_0')} | {w('dual/lambda_1')} | {w('train/return_mean', 1)} | "
                     f"{fmt(noise[0])} / {fmt(noise[1])} | {fmt(swing[0])} / {fmt(swing[1])} | "
                     f"{fmt(slope[0])} / {fmt(slope[1])} |")
    lines += ["", "## Training: cumulative positive excess over all updates", "",
              "`batch` = Σ_k max(0, x̂_i[k] − 1) over updates (what the controller sees, budget units × updates). "
              "`episode` = Σ over training episodes of max(0, C_i − d_i) / d_i.", "",
              "| cond | seed | updates | batch excess h | batch excess v | episode excess h | episode excess v |",
              "|---|---|---|---|---|---|---|"]
    for (cond, seed), (run, rows) in sorted(data.items()):
        bx = [np.maximum(col(rows, f"train/normalized_{c}") - 1, 0).sum() for c in COSTS]
        ex = [rows[-1][f"train/cumulative_episode_excess_{c}"] / b for c, b in zip(COSTS, BUDGETS)]
        lines.append(f"| {cond} | {seed} | {len(rows)} | {bx[0]:.1f} | {bx[1]:.1f} | {ex[0]:.1f} | {ex[1]:.1f} |")
    return lines


def validation(root, data):
    table = {}
    for (cond, seed), (run, rows) in data.items():
        for ck in VALIDATION:
            f = run / "validation" / f"ck{ck}.npz"
            if f.exists():
                z = np.load(f)
                x = z["costs"] / BUDGETS
                n = len(x)
                table[(cond, seed, ck)] = {
                    "n": n, "ret": z["returns"].mean(), "ret_se": z["returns"].std(ddof=1) / np.sqrt(n),
                    "x": x.mean(0), "x_se": x.std(0, ddof=1) / np.sqrt(n), "joint": float((x <= 1).all(1).mean())}
    lines = ["", "## Validation (200 episodes, env seed 700000, action seed 700001, chunk 10; stochastic)", "",
             "Per checkpoint; ± = standard error over the 200 episodes (episode uncertainty). "
             "Joint pass = fraction of episodes with both C_i ≤ d_i (auxiliary; not the expected-cost constraint).", "",
             "| cond | seed | ckpt (M steps) | return | x_h | x_v | both mean ≤ 1 | joint pass |", "|---|---|---|---|---|---|---|---|"]
    for key in sorted(table):
        v = table[key]
        lines.append(f"| {key[0]} | {key[1]} | {int(key[2]) / 1e6:.3f} | {v['ret']:.1f} ± {v['ret_se']:.1f} | "
                     f"{v['x'][0]:.2f} ± {v['x_se'][0]:.2f} | {v['x'][1]:.2f} ± {v['x_se'][1]:.2f} | "
                     f"{'yes' if (v['x'] <= 1).all() else 'no'} | {v['joint']:.2f} |")
    lines += ["", "### Mean of the per-checkpoint validation metrics (4.0–5.0M, 5 checkpoints)", "",
              "This is the **mean of the checkpoint-wise evaluation metrics**, not one policy's performance and "
              "not an average of network weights. Seed spread = std over training seeds of the per-seed means.", "",
              "| cond | seeds with all 5 ckpts | return | x_h | x_v | joint pass | return seed spread | x_h seed spread | x_v seed spread |",
              "|---|---|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        per_seed = []
        for seed in range(3):
            vals = [table.get((cond, seed, ck)) for ck in VALIDATION]
            if all(vals):
                per_seed.append([np.mean([v["ret"] for v in vals]), *np.mean([v["x"] for v in vals], 0),
                                 np.mean([v["joint"] for v in vals])])
        if per_seed:
            a = np.array(per_seed)
            sd = a.std(0, ddof=1) if len(a) > 1 else np.full(4, np.nan)
            lines.append(f"| {cond} | {len(a)} | {a[:, 0].mean():.1f} | {a[:, 1].mean():.2f} | {a[:, 2].mean():.2f} | "
                         f"{a[:, 3].mean():.2f} | {fmt(sd[0], 1)} | {fmt(sd[1])} | {fmt(sd[2])} |")
        else:
            lines.append(f"| {cond} | 0 | — | — | — | — | — | — | — |")
    return lines, table


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def smooth(v, w):
    return np.convolve(v, np.ones(w) / w, mode="valid") if len(v) >= w else np.array([])


def curves_figure(data, path, window=50):
    rows_spec = [("dual/lambda_0", "λ_hazard", False), ("dual/lambda_1", "λ_vase", False),
                 ("train/normalized_hazard", f"x_hazard ({window}-upd mean)", True),
                 ("train/normalized_vase", f"x_vase ({window}-upd mean)", True),
                 ("train/return_mean", f"return ({window}-upd mean)", True)]
    fig, axes = plt.subplots(len(rows_spec), 4, figsize=(15, 12), sharex=True, sharey="row")
    for c, cond in enumerate(CONDITIONS):
        axes[0, c].set_title(LABEL[cond], fontsize=10, color=INK)
        for seed in range(3):
            if (cond, seed) not in data:
                continue
            rows = data[(cond, seed)][1]
            steps = col(rows, "env_steps") / 1e6
            for r, (key, _, sm) in enumerate(rows_spec):
                v = col(rows, key)
                if sm:
                    y = smooth(v, window)
                    x = steps[window // 2: window // 2 + len(y)]
                else:
                    x, y = steps, v
                axes[r, c].plot(x, y, color=SEED_COLOR[seed], lw=1.4, label=f"seed {seed}")
        for r, (key, label, _) in enumerate(rows_spec):
            style(axes[r, c])
            if key.startswith("train/normalized"):
                axes[r, c].axhline(1, color=INK, lw=1, ls=":")
            for lo, _ in WINDOWS[1:]:
                axes[r, c].axvline(lo / 1e6, color=GRID, lw=0.8)
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=9)
        axes[-1, c].set_xlabel("env steps (M)", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def validation_figure(table, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for c, cond in enumerate(CONDITIONS):
        for m, (title, get) in enumerate([("x_hazard", lambda v: (v["x"][0], v["x_se"][0])),
                                          ("x_vase", lambda v: (v["x"][1], v["x_se"][1])),
                                          ("return", lambda v: (v["ret"], v["ret_se"]))]):
            for seed in range(3):
                pts = [(int(ck) / 1e6, *get(table[(cond, seed, ck)])) for ck in VALIDATION if (cond, seed, ck) in table]
                if pts:
                    p = np.array(pts)
                    axes[m].errorbar(p[:, 0] + (c - 1.5) * 0.03, p[:, 1], yerr=p[:, 2], color=COND_COLOR[c],
                                     marker="os^"[seed], ms=5, lw=1, capsize=2,
                                     label=cond if seed == 0 else None, alpha=0.9)
            axes[m].set_title(title, fontsize=10)
    for ax in axes[:2]:
        ax.axhline(1, color=INK, lw=1, ls=":")
    for ax in axes:
        style(ax)
        ax.set_xlabel("checkpoint (M steps); marker = seed (o 0, s 1, ^ 2)", fontsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = runs(args.root)
    status = ["## Run status", "", "| cond | seed | updates logged | complete |", "|---|---|---|---|"]
    for cond in CONDITIONS:
        for seed in range(3):
            if (cond, seed) in data:
                run, rows = data[(cond, seed)]
                status.append(f"| {cond} | {seed} | {rows[-1]['update']}/1250 | {'yes' if (run / 'summary.json').exists() else 'no'} |")
            else:
                status.append(f"| {cond} | {seed} | not started | no |")
    val_lines, table = validation(args.root, data)
    text = "\n".join(["# E11 analysis (generated by scripts/analyze_pi.py)", ""] + status + [""] +
                     training_tables(data) + val_lines) + "\n"
    (args.output / "analysis.md").write_text(text)
    curves_figure(data, args.output / "pi-curves.png")
    if table:
        validation_figure(table, args.output / "pi-validation.png")
    print(text)


if __name__ == "__main__":
    main()
