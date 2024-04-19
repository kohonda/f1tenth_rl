import gymnasium as gym
import wandb
import datetime
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from f1tenth_wrapper.env import F1TenthWrapper

run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

seed = 42
set_random_seed(seed)

wandb_config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_id": "f1tenth-RL-v0",
    "seed": seed,
}


run = wandb.init(
    project="F1Tenth_SAC",
    config=wandb_config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
    id=run_id,
)

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
    wandb_config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}"
)

model.learn(
    total_timesteps=wandb_config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

model_path = f"models/{run.id}/model.zip"
wandb.save(model_path)

run.finish()
