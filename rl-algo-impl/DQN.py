import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# --- 1. The Neural Network ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # TODO: Define a simple Multi-Layer Perceptron (MLP)
        # Input: state_dim -> Hidden layers -> Output: action_dim (Q-values)

    def forward(self, x):
        # TODO: Implement the forward pass
        pass

# --- 2. The Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Return a random batch of transitions as Tensors
        pass

# --- 3. The Algorithm Logic ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.99
        self.epsilon = 1.0

    def select_action(self, state):
        # TODO: Implement Epsilon-Greedy selection
        # With probability epsilon, pick random; else, pick argmax Q(s, a)
        pass

    def train(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return

        # TODO: THE CORE LOGIC
        # 1. Sample a batch from memory
        # 2. Compute current Q values: Q(s, a)
        # 3. Compute Target Q values: Reward + gamma * max(Q_target(s', a')) * (1 - done)
        # 4. Compute Loss (MSE or Huber) and perform Backprop
        pass

# --- 4. Main Training Loop ---
env = gym.make("CartPole-v1")
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.memory.push(state, action, reward, next_state, done)
        agent.train(batch_size=64)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
            
    # TODO: Update Target Network every N episodes
    print(f"Episode {episode}, Reward: {total_reward}")