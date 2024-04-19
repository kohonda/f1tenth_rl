from typing import Tuple, List
import fire
import time
import copy
import os
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import SAC
from f1tenth_wrapper.env import F1TenthWrapper


class RLWrapper:
    def __init__(
        self, model_path: str, original_env: F1TenthWrapper = None, horizon: int = 10
    ):
        self.model = SAC.load(model_path)

        if original_env is not None:
            self.env = original_env.unwrapped.clone(render_mode="rgb_array")
            _, _ = self.env.reset()
            self._predictive_state = np.zeros((horizon, 2))
            self._predictive_action = np.zeros((horizon, 2))
            self._horizon = horizon

    def get_dist_params(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tensor_obs, _ = self.model.policy.actor.obs_to_tensor(obs)

        with torch.no_grad():
            # get mean and std from the actor policy
            mean, log_std, _ = self.model.policy.actor.get_action_dist_params(
                tensor_obs
            )
            std = log_std.exp()

            # scale the mean and std
            # tanh squashes the mean and std
            squashed_mean = torch.tanh(mean)
            squashed_val = torch.tanh(mean + std)

        # unscale the mean and std based on the action space
        unscale_means = self.model.policy.actor.unscale_action(
            squashed_mean.detach().cpu().numpy()
        )
        unscale_vals = self.model.policy.actor.unscale_action(
            squashed_val.detach().cpu().numpy()
        )
        unscale_std = unscale_vals - unscale_means

        return unscale_means, unscale_std

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        action = self.model.predict(obs, deterministic=True)[0]
        return action

    def get_predictive_state(
        self, original_env: F1TenthWrapper, current_action: np.ndarray
    ) -> None:

        # sync the environment
        self.env.sync_env(original_env)

        obs, step_reward, done, truncated, info = self.env.step(current_action)

        # save the state and action for the current step
        self._predictive_action[0] = current_action
        x = info["original_obs"]["poses_x"][0]
        y = info["original_obs"]["poses_y"][0]
        # yaw = info["original_obs"]["poses_theta"][0]
        # self._predictive_state.append([x, y, yaw])
        self._predictive_state[0] = [x, y]

        for i in range(self._horizon - 1):
            action = self.get_action(obs)
            obs, step_reward, done, truncated, info = self.env.step(action)

            # save the state and action
            self._predictive_action[i + 1] = action
            x = info["original_obs"]["poses_x"][0]
            y = info["original_obs"]["poses_y"][0]
            # yaw = info["original_obs"]["poses_theta"][0]
            # self._predictive_state.append([x, y, yaw])
            self._predictive_state[i + 1] = [x, y]

    def render_predictive_state(self, e) -> None:
        e.render_lines(self._predictive_state, color=(0, 0, 255), size=1)


def main(model_path: str, video_recording: bool = False):
    seed = 1
    set_random_seed(seed)

    # create the environment
    if video_recording:
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
            render_mode="rgb_array",
        )

        # check folder exists
        if not os.path.exists("videos"):
            os.makedirs("videos")

        video_path = os.path.join("videos", f"RL_{time.time()}")

        env = gym.wrappers.RecordVideo(env, video_path)
    else:
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

    # RL model
    rl_wrapper = RLWrapper(model_path, original_env=env, horizon=30)
    env.unwrapped.add_render_callback(rl_wrapper.render_predictive_state)

    track = env.unwrapped.track
    env.unwrapped.add_render_callback(track.raceline.render_waypoints)

    obs, info = env.reset()

    done = False
    if video_recording:
        frames = [env.render()]
    else:
        env.render()

    laptime = 0.0
    time_step = env.unwrapped.core.config["timestep"]
    if video_recording:
        end_time = 15.0
    else:
        end_time = float("inf")

    while laptime < end_time and not done:
        action = rl_wrapper.get_action(obs)
        obs, step_reward, done, truncated, info = env.step(action)
        # rl_wrapper.get_predictive_state(env, action)
        laptime += time_step

        if video_recording:
            frame = env.render()
            frames.append(frame)
        else:
            frame = env.render()

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
