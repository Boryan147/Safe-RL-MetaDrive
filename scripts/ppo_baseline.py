from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

# custom callback to log cost during training
class CostLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cumulative_cost = 0

    def _on_step(self) -> bool:
        # cost from info at every step
        for info in self.locals["infos"]:
            if "cost" in info:
                self.cumulative_cost += info["cost"]
        self.logger.record("train/cumulative_cost", self.cumulative_cost)
        return True

def train():
    config = {
        "use_render": False,
        "traffic_density": 0.15,
        "map": "C",  
        "accident_prob": 0.5,
    }
    env = SafeMetaDriveEnv(config)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/sb3_baseline/",
        device="auto" 
    )

    print("Starting training: PPO Naive Baseline...")
    model.learn(
        total_timesteps=100000, 
        callback=CostLoggingCallback(),
        tb_log_name="ppo_safe_env"
    )

    os.makedirs("results/models/", exist_ok=True)
    model.save("results/models/ppo_metadrive_baseline")
    print("Training finished and model saved!")

if __name__ == "__main__":
    train()