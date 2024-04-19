import gymnasium as gym
import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from f1tenth_wrapper.env import F1TenthWrapper

run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

seed = 42
set_random_seed(seed)

wandb_config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 3000000,
    "env_id": "f1tenth-RL-v0",
    "seed": seed,
}

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


model = SAC(
    wandb_config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run_id}"
)

model.learn(
    total_timesteps=wandb_config["total_timesteps"],
    progress_bar=True,
)

model_path = f"models/{run_id}/model.zip"
model.save(model_path)
