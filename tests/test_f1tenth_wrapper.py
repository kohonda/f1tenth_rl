import time
from typing import Tuple

import gymnasium as gym
import numpy as np

from f1tenth_wrapper.env import F1TenthWrapper
from stable_baselines3.common.utils import set_random_seed


def main():

    seed = 42
    set_random_seed(seed)

    env = gym.make(
        "f1tenth-RL-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "original"},
            "params": {"mu": 1.0},
            "reset_config": {"type": "rl_random_static"},
            "seed": seed,
        },
        render_mode="human",
    )

    for _ in range(3):
        obs, info = env.reset()
        done = False
        env.render()

        while not done:
            action = env.action_space.sample()
            obs, step_reward, done, truncated, info = env.step(action)
            frame = env.render()


if __name__ == "__main__":
    main()
