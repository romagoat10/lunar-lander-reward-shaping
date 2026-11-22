"""Show a random LunarLander-v3 episode before training."""

from __future__ import annotations

from rocket_lander import make_env


def main() -> None:
    env = make_env("baseline", render_mode="human")
    observation, info = env.reset(seed=7)
    print("Observation order: x, y, vx, vy, angle, angular_velocity, left_leg, right_leg")
    print("First observation:", observation)
    print("Actions: 0 coast, 1 left engine, 2 main engine, 3 right engine")

    episode_return = 0.0
    try:
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated
    finally:
        env.close()

    print(f"Random-policy episode return: {episode_return:.1f}")
    print("A random policy is only a sanity check; it is not expected to land reliably.")


if __name__ == "__main__":
    main()
