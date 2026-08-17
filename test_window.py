import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from metadrive.envs.metadrive_env import MetaDriveEnv

def make_env():
    env = MetaDriveEnv(dict(
        use_render=True, 
        image_observation=False,
        num_scenarios=10,  
        start_seed=0
    ))
    return env

if __name__ == "__main__":
    # Wrap the env in a DummyVecEnv for SB3
    env = DummyVecEnv([make_env])

    # Initialize the PPO agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)

    print("Starting training...")
    # Train for a small number of steps to test connectivity
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("ppo_metadrive_test")
    print("Training complete and model saved!")