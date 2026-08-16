from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy

# Initialize safe driving environment with hazardous obstacles
env = SafeMetaDriveEnv(
    dict(
        use_render=False,
        agent_policy=ExpertPolicy,  # Built-in expert driver
        start_seed=1000,
        num_scenarios=5,
        accident_prob=0.8,
    )
)

obs, info = env.reset()
total_reward = 0.0
total_cost = 0.0

for step in range(500):
    # Pass dummy action; ExpertPolicy calculates steering/throttle internally
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])

    total_reward += reward
    total_cost += info.get("cost", 0.0)

    if terminated or truncated:
        print(
            f"Episode Finished | Steps: {step+1} | "
            f"Total Reward: {total_reward:.2f} | Total Cost: {total_cost:.2f} | "
            f"Arrived Destination: {info.get('arrive_dest', False)}",
            flush=True
        )
        obs, info = env.reset()
        total_reward = 0.0
        total_cost = 0.0

env.close()