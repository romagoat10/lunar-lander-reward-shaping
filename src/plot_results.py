"""Plot task performance, not the modified training reward."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "baseline": "Built-in reward",
    "hover_bonus": "+1 per frame (flawed)",
    "potential": "Potential difference",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("artifacts/results/episodes.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/plots/reward_comparison.png"))
    args = parser.parse_args()
    if not args.results.exists():
        raise FileNotFoundError(f"No evaluation data at {args.results}; train/evaluate agents first.")

    episodes = pd.read_csv(args.results)
    if episodes.empty:
        raise ValueError("The evaluation file is empty.")

    by_run = episodes.groupby(["experiment", "train_seed"], as_index=False).agg(
        mean_task_return=("task_return", "mean"),
        solve_rate=("solved", "mean"),
    )
    order = [name for name in LABELS if name in by_run["experiment"].unique()]
    means = by_run.groupby("experiment").mean(numeric_only=True).reindex(order)
    spreads = by_run.groupby("experiment").std(numeric_only=True).reindex(order).fillna(0)
    labels = [LABELS[name] for name in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = range(len(order))
    axes[0].bar(x, means["mean_task_return"], color="#315f50", yerr=spreads["mean_task_return"], capsize=4)
    axes[0].axhline(200, color="#b86b44", linestyle="--", linewidth=1.4, label="Solved threshold (200)")
    axes[0].set_title("Original task return")
    axes[0].set_ylabel("Mean episode return ± SD across training seeds")
    axes[0].legend(frameon=False)
    axes[1].bar(x, means["solve_rate"] * 100, color="#739a82", yerr=spreads["solve_rate"] * 100, capsize=4)
    axes[1].set_title("Landing success rate")
    axes[1].set_ylabel("Episodes scoring ≥ 200 (%)")
    axes[1].set_ylim(0, 100)

    for axis in axes:
        axis.set_xticks(list(x), labels, rotation=12, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    fig.suptitle("Reward shaping can change what PPO learns", fontsize=14, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
