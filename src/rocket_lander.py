"""LunarLander reward wrappers for the reward-shaping experiment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


ENV_ID = "LunarLander-v3"
GAMMA = 0.99


class RewardDesignWrapper(gym.Wrapper):
    """Expose base reward and apply one of three reward designs.

    ``hover_bonus`` is deliberately flawed: it adds +1 on every frame, even
    when the lander makes no progress. ``potential`` adds the discounted
    change in a state potential and sets terminal potential to zero.
    """

    def __init__(self, env: gym.Env, design: str, gamma: float = GAMMA):
        super().__init__(env)
        if design not in {"baseline", "hover_bonus", "potential"}:
            raise ValueError(f"Unknown reward design: {design}")
        self.design = design
        self.gamma = gamma
        self.previous_potential = 0.0

    @staticmethod
    def potential(observation: np.ndarray) -> float:
        """A small negative cost for being far, fast, or poorly aligned."""
        x, y, vx, vy, angle, angular_velocity, *_ = observation
        distance = float(np.hypot(x, max(float(y), 0.0)))
        speed = float(np.hypot(vx, vy))
        return -(
            0.50 * distance
            + 0.20 * speed
            + 0.15 * abs(float(angle))
            + 0.05 * abs(float(angular_velocity))
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(observation)
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        ended = terminated or truncated

        if self.design == "hover_bonus":
            extra_reward = 1.0
        elif self.design == "potential":
            next_potential = 0.0 if ended else self.potential(observation)
            extra_reward = self.gamma * next_potential - self.previous_potential
            self.previous_potential = next_potential
        else:
            extra_reward = 0.0

        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["shaping_reward"] = float(extra_reward)
        return observation, float(base_reward + extra_reward), terminated, truncated, info


def make_env(design: str, render_mode: str | None = None) -> gym.Env:
    """Create LunarLander-v3 wrapped with the selected reward design."""
    env = gym.make(ENV_ID, render_mode=render_mode)
    return RewardDesignWrapper(env, design=design, gamma=GAMMA)
