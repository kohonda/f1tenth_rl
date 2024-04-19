import gymnasium as gym


gym.register(
    id="f1tenth-RL-v0",
    entry_point="f1tenth_wrapper.env:F1TenthWrapper",
)
