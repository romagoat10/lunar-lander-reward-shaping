"""Train PPO agents under comparable LunarLander reward designs."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rocket_lander import GAMMA, make_env

DESIGNS = ("baseline", "hover_bonus", "potential")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(*DESIGNS, "all"), default="all")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def build_vector_env(design: str, seed: int, monitor_path: Path) -> DummyVecEnv:
    def factory():
        env = make_env(design)
        return Monitor(env, filename=str(monitor_path))

    vector_env = DummyVecEnv([factory])
    vector_env.seed(seed)
    return vector_env


def main() -> None:
    args = parse_args()
    designs = DESIGNS if args.experiment == "all" else (args.experiment,)
    for design in designs:
        for seed in args.seeds:
            run_dir = args.output_dir / "runs" / design / f"seed_{seed}"
            model_dir = args.output_dir / "models" / design / f"seed_{seed}"
            monitor_dir = run_dir / "monitor"
            best_dir = model_dir / "best"
            for directory in (monitor_dir, best_dir, run_dir):
                directory.mkdir(parents=True, exist_ok=True)

            train_env = build_vector_env(design, seed, monitor_dir / "train.csv")
            eval_env = build_vector_env(design, seed + 10_000, monitor_dir / "eval.csv")
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(run_dir / "evaluation"),
                eval_freq=max(args.eval_freq, 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                verbose=0,
            )
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                gamma=GAMMA,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                policy_kwargs={"net_arch": [64, 64]},
                seed=seed,
                verbose=1,
            )

            config = {
                "environment": "LunarLander-v3",
                "reward_design": design,
                "seed": seed,
                "timesteps": args.timesteps,
                "gamma": GAMMA,
                "algorithm": "PPO",
                "policy": "MlpPolicy [64, 64]",
                "python_version": platform.python_version(),
                "gymnasium_version": gym.__version__,
                "stable_baselines3_version": stable_baselines3.__version__,
            }
            (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            print(f"\n=== Training {design}, seed {seed} ===")
            try:
                model.learn(total_timesteps=args.timesteps, callback=eval_callback)
                model.save(model_dir / "final_model")
            finally:
                train_env.close()
                eval_env.close()
            print(f"Saved checkpoints and logs under {args.output_dir}")


if __name__ == "__main__":
    main()
