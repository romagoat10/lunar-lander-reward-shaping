"""Summarize episode results by training seed and reward design."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/multiseed_evaluation.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()
    if not args.results.exists():
        raise FileNotFoundError(f"No evaluation data at {args.results}.")
    episodes = pd.read_csv(args.results)
    if episodes.empty:
        raise ValueError("The evaluation file is empty.")
    per_seed = episodes.groupby(["experiment", "train_seed"], as_index=False).agg(
        mean_task_return=("task_return", "mean"),
        solve_rate=("solved", "mean"),
        episodes=("episode", "count"),
    )
    summary = per_seed.groupby("experiment").agg(
        training_seeds=("train_seed", "count"),
        episodes_per_seed=("episodes", "min"),
        mean_task_return=("mean_task_return", "mean"),
        task_return_sd_across_seeds=("mean_task_return", "std"),
        mean_solve_rate=("solve_rate", "mean"),
        solve_rate_sd_across_seeds=("solve_rate", "std"),
    ).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, float_format="%.4f")
    print(f"Saved {args.output}")
    print(per_seed.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    print("\nMean and sample SD across training seeds:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

if __name__ == "__main__":
    main()
