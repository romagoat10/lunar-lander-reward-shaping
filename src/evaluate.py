"""Evaluate saved policies on the original LunarLander task reward."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from rocket_lander import make_env

FIELDS = ["experiment", "train_seed", "episode", "task_return", "training_return", "steps", "solved"]

def evaluate_model(model: PPO, design: str, train_seed: int, episodes: int, seed: int, video_path: Path | None = None) -> list[dict]:
    env = make_env(design, render_mode="rgb_array" if video_path else None)
    rows: list[dict] = []
    video_frames = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            task_return = 0.0
            training_return = 0.0
            steps = 0
            if video_path:
                video_frames.append(env.render())
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                task_return += info["base_reward"]
                training_return += reward
                steps += 1
                done = terminated or truncated
                if video_path:
                    video_frames.append(env.render())
            rows.append({"experiment": design, "train_seed": train_seed, "episode": episode,
                         "task_return": task_return, "training_return": training_return,
                         "steps": steps, "solved": int(task_return >= 200)})
        if video_path:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, video_frames, fps=50)
    finally:
        env.close()
    return rows

def write_results(rows: list[dict], path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    continue_file = append and path.exists()
    with path.open("a" if continue_file else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not continue_file:
            writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("baseline", "hover_bonus", "potential", "all"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=100_000,
                        help="Starting environment seed shared across models for paired evaluation")
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final",
                        help="Use final checkpoints for a consistent comparison or best checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--record-video", action="store_true", help="Save all evaluation episodes for each model into one MP4")
    parser.add_argument("--append", action="store_true", help="Append rows instead of replacing the results CSV")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    designs = ("baseline", "hover_bonus", "potential") if args.experiment == "all" else (args.experiment,)
    results_path = args.output_dir / "results" / "episodes.csv"
    first_result = True
    for design in designs:
        for train_seed in args.seeds:
            if args.checkpoint == "best":
                model_path = args.output_dir / "models" / design / f"seed_{train_seed}" / "best" / "best_model.zip"
            else:
                model_path = args.output_dir / "models" / design / f"seed_{train_seed}" / "final_model.zip"
            if not model_path.exists():
                raise FileNotFoundError(f"No saved model for {design}, seed {train_seed}: train this run first")
            model = PPO.load(model_path)
            video_path = args.output_dir / "videos" / f"{design}_seed_{train_seed}.mp4" if args.record_video else None
            rows = evaluate_model(model, design, train_seed, args.episodes, args.eval_seed, video_path)
            write_results(rows, results_path, append=args.append or not first_result)
            first_result = False
            print(f"{design} seed {train_seed}: task return {np.mean([row['task_return'] for row in rows]):.1f}; solve rate {np.mean([row['solved'] for row in rows]):.0%}")
    print(f"Saved episode metrics to {results_path}")

if __name__ == "__main__":
    main()
