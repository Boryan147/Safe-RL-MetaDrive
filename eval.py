import time
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO

# 1. Configure the environment for 3D GUI rendering
eval_config = {
    "use_render": True,          # Launch the Panda3D window
    "manual_control": False,      # Policy controls the vehicle
    "num_scenarios": 5,           # Test across 5 procedural maps
    "start_seed": 100,            # Unseen map seeds
    "traffic_density": 0.05,      # Background traffic density
    "decision_repeat": 5,
}

env = MetaDriveEnv(eval_config)

# 2. Load the trained SB3 PPO checkpoint
model = PPO.load("ppo_metadrive_test")

num_episodes = 3

try:
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        total_cost = 0.0

        print(f"\n================ Episode {ep + 1} Starting ================")
        print(f"Initial Observation Shape: {obs.shape}")

        while not done:
            # Predict action from trained policy
            action, _states = model.predict(obs, deterministic=True)

            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            cost = info.get("cost", 0.0)
            total_reward += reward
            total_cost += cost
            step += 1

            # Print shapes, reward, and safety cost per step
            print(
                f"[Step {step:03d}] "
                f"Obs Shape: {obs.shape} | "
                f"Action Shape: {action.shape} | "
                f"Action: [{action[0]:+.2f}, {action[1]:+.2f}] | "
                f"Reward: {reward:+.3f} | "
                f"Cost: {cost:.1f}"
            )

            done = terminated or truncated
            time.sleep(0.02)  # Maintain smooth visual playback rate (~50 FPS)

        # Episode Outcome Summary
        outcome = (
            "Arrived at Destination" if info.get("arrive_dest", False)
            else "Crashed" if info.get("crash", False)
            else "Out of Road" if info.get("out_of_road", False)
            else "Max Steps / Truncated"
        )
        print(f"\n--- Episode {ep + 1} Summary ---")
        print(f"Total Steps:  {step}")
        print(f"Total Reward: {total_reward:.3f}")
        print(f"Total Cost:   {total_cost:.1f}")
        print(f"Outcome:      {outcome}")

finally:
    env.close()