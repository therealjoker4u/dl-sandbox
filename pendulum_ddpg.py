import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

_env_kwargs = {
    "render_mode" : "human"
}
env = gym.make("Pendulum-v1", **_env_kwargs)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = float(env.action_space.high[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_bound

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.relu(self.fc1(torch.cat([x, a], dim=-1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.float32, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device)
        )

    def size(self):
        return len(self.buffer)


gamma = 0.99
tau = 0.005
lr_actor = 0.0001
lr_critic = 0.0003
batch_size = 64
max_episodes = 1000
max_steps = 200
start_steps = 1000

# Set a value greater than 0.0 to explore the environment (for example : 0.1)
exploration_noise = 0.0


actor = Actor(obs_dim, action_dim, action_bound).to(device)
if os.path.exists("./models/pendulum_ddpg_actor.pth"):
    actor.load_state_dict(torch.load("./models/pendulum_ddpg_actor.pth"))
    print("Pretrained Actor model loaded")
    
critic = Critic(obs_dim, action_dim).to(device)
if os.path.exists("./models/pendulum_ddpg_critic.pth"):
    critic.load_state_dict(torch.load("./models/pendulum_ddpg_critic.pth"))
    print("Pretrained Critic model loaded")

target_actor = Actor(obs_dim, action_dim, action_bound).to(device)
target_critic = Critic(obs_dim, action_dim).to(device)

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)


replay_buffer = ReplayBuffer()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


for episode in range(max_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if replay_buffer.size() < start_steps:
            action = env.action_space.sample()
        else:
            action = actor(state_tensor).cpu().detach().numpy()[0]
            action += exploration_noise * np.random.randn(action_dim)
            action = np.clip(action, -action_bound, action_bound)

        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated

        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if replay_buffer.size() >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            with torch.no_grad():
                target_actions = target_actor(next_states)
                target_q = target_critic(next_states, target_actions)
                target_q = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * target_q

            q_values = critic(states, actions)
            critic_loss = nn.MSELoss()(q_values, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(states, actor(states)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            soft_update(target_actor, actor, tau)
            soft_update(target_critic, critic, tau)

        if done:
            break

    print(f"Episode {episode+1}, Reward: {episode_reward}")
    torch.save(actor.state_dict(), "./models/pendulum_ddpg_actor.pth")
    torch.save(critic.state_dict(), "./models/pendulum_ddpg_critic.pth")

env.close()
