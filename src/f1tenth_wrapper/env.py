"""
Kohei Honda, 2024
"""

from typing import Tuple, List
import gymnasium as gym
import numpy as np

from utils.waypoints_handler import nearest_point_on_trajectory, calculate_curvatures
from f1tenth_gym.envs import F110Env


class F1TenthWrapper(gym.Env):
    """
    Wrapper for the F1Tenth Gym environment to use in RL training.
    Reward and observation design is based on the article:
    https://arxiv.org/pdf/2008.07971.pdf
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, config: dict = None, render_mode=None, **kwargs):

        if config["num_agents"] != 1:
            raise ValueError("Only one agent is supported in this wrapper.")

        if config["observation_config"]["type"] != "original":
            raise ValueError(
                "Only original observation type is supported in this wrapper."
            )

        self.core = F110Env(config=config, render_mode=render_mode, **kwargs)
        self.render_mode = render_mode

        # waypoints
        self.track = self.core.unwrapped.track
        waypoints = np.stack(
            [self.track.raceline.xs, self.track.raceline.ys, self.track.raceline.vxs]
        ).T
        self._waypoints = waypoints[:, :2]

        # space definition
        self.action_space = self.core.action_space
        self.DIM_DEPTH = 18  # number of depth sensors used in the observation
        self.DIM_OBS = 28
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.DIM_OBS,), dtype=np.float32
        )

        # inner variables
        self.prev_vels = np.zeros(3)
        self.prev_steer_angle = 0.0
        self._current_waypoint = np.zeros(2)
        self._current_index = 0
        self.prev_waypoint = np.zeros(2)

    def step(self, action):
        """
        Returns the next observation, reward, done, info tuple.
        observation: [v_x, v_y, dv_x, dv_y, yaw, scan x18 (distance to the wall), prev_steer, path_curvature]
        reward: (progress of the reference path) - (penalty when the car hits the track)
        """
        original_obs, step_reward, done, truncated, info = self.core.step(action)

        # calculate current waypoint and index
        self._current_waypoint, self._current_index = self.calc_current_waypoint(
            original_obs
        )

        obs = self._observation(original_obs)
        reward = self._reward(original_obs)

        # take over the previous values
        self._take_over(original_obs, action)

        info = {"original_obs": original_obs, "info": info}

        return obs, reward, done, truncated, info

    def _take_over(self, original_obs, action):
        self.prev_steer_angle = action[0][0]
        self.prev_vels = np.array(
            [
                original_obs["linear_vels_x"],
                original_obs["linear_vels_y"],
                original_obs["ang_vels_z"],
            ]
        )
        self.prev_waypoint = self._current_waypoint

    def reset(self, seed=None, options=None):
        original_obs, info = self.core.reset(seed=seed, options=options)
        obs = self._observation(original_obs)

        self.prev_vels = np.zeros(3)
        self.prev_steer_angle = 0.0
        self._current_waypoint, self._current_index = self.calc_current_waypoint(
            original_obs
        )
        self.prev_waypoint = self._current_waypoint

        info = {"original_obs": original_obs, "info": info}

        return obs, info

    def clone(self, render_mode=None):
        if render_mode is None:
            return F1TenthWrapper(config=self.core.config, render_mode=self.render_mode)
        else:
            return F1TenthWrapper(config=self.core.config, render_mode=render_mode)

    def sync_env(self, original_env):
        self.core.sync_env(original_env.unwrapped.core)
        self.prev_vels = original_env.unwrapped.prev_vels.copy()
        self.prev_steer_angle = original_env.unwrapped.prev_steer_angle
        self.prev_waypoint = original_env.unwrapped.prev_waypoint.copy()

    def calc_current_waypoint(self, original_obs):
        current_pos = np.array([original_obs["poses_x"][0], original_obs["poses_y"][0]])
        nearest_point, _, _, index = nearest_point_on_trajectory(
            point=current_pos, trajectory=self._waypoints
        )
        return nearest_point, index

    def add_render_callback(self, callback_func):
        self.core.add_render_callback(callback_func)

    def render(self, mode="human"):
        return self.core.render()

    def close(self):
        return self.core.close()

    def _observation(self, original_obs) -> np.ndarray:
        # velocity
        vx = original_obs["linear_vels_x"]
        vy = original_obs["linear_vels_y"]
        ang_v = original_obs["ang_vels_z"]

        # delta velocity
        dvx = vx - self.prev_vels[0]
        dvy = vy - self.prev_vels[1]
        dang_v = ang_v - self.prev_vels[2]

        # get nearest point on the wayopints
        index = self._current_index

        # yaw angle deviation from the waypoint
        yaw = original_obs["poses_theta"]
        next_index = (index + 1) % self._waypoints.shape[0]
        dx = self._waypoints[next_index, 0] - self._waypoints[index, 0]
        dy = self._waypoints[next_index, 1] - self._waypoints[index, 1]
        yaw_ref = np.arctan2(dy, dx)
        yaw_dev = yaw - yaw_ref

        # disttances from depth sensors
        sparse_idx = np.linspace(
            0, original_obs["scans"].shape[1] - 1, num=self.DIM_DEPTH, dtype=int
        )
        depths = original_obs["scans"][:, sparse_idx]

        # calculate curvature from N points from the current index
        N = 10
        ahead_points = np.zeros((N, 2))
        for i in range(N):
            ahead_points[i, 0] = self._waypoints[
                (index + i) % self._waypoints.shape[0], 0
            ]
            ahead_points[i, 1] = self._waypoints[
                (index + i) % self._waypoints.shape[0], 1
            ]
        curvatures = calculate_curvatures(ahead_points)
        path_curvature = curvatures[0]

        # collision
        collisions = original_obs["collisions"]

        # set observation
        obs = np.ndarray(self.DIM_OBS, dtype=np.float32)
        obs[0] = vx
        obs[1] = vy
        obs[2] = ang_v
        obs[3] = dvx
        obs[4] = dvy
        obs[5] = dang_v
        obs[6] = yaw_dev
        obs[7:25] = depths
        obs[25] = self.prev_steer_angle
        obs[26] = path_curvature
        obs[27] = collisions[0]

        # chack inf or nan
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print("obs: ", obs)
            raise ValueError("Invalid observation")

        return obs

    def _reward(self, original_obs) -> float:
        # distance between the current waypoint and the previous waypoint
        dist = np.linalg.norm(self._current_waypoint - self.prev_waypoint)

        # collision penalty with speed penalty
        collisions = original_obs["collisions"]
        if collisions[0] > 0:
            collision_efficient = 5e-2
            speed_norm = (self.prev_vels[0] ** 2 + self.prev_vels[1] ** 2)[0]
            collision_penalty = -collision_efficient * speed_norm
        else:
            collision_penalty = 0.0

        reward = dist + collision_penalty

        # check inf or nan
        if np.isnan(reward) or np.isinf(reward):
            print("reward: ", reward)
            raise ValueError("Invalid reward")

        return reward
