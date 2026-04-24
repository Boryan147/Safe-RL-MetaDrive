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
            self.cumulative_cost += info["cost"]
        self.logger.record("ppo_baseline/cost", self.cumulative_cost)
        return True

def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "../logs/sb3ppo_baseline/")
    results_dir = os.path.join(script_dir, "../results/models/")
    
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
        tensorboard_log=logs_dir,
        device="auto" 
    )

    print("Starting training: PPO Naive Baseline...")
    model.learn(
        total_timesteps=100000, 
        callback=CostLoggingCallback(),
        tb_log_name="ppo_baseline"
    )

    os.makedirs(results_dir, exist_ok=True)
    model.save(os.path.join(results_dir, "ppo_baseline"))
    print("Training finished and model saved!")

if __name__ == "__main__":
    train()