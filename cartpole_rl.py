import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


env = gym.make('CartPole-v1', render_mode="human")


policy_net = PolicyNetwork(state_dim=4, action_dim=2)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)


def select_action(policy_net, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = policy_net(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)


def finish_episode(saved_log_probs, rewards, gamma=0.99):
    R = 0
    policy_loss = []
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    for log_prob, R in zip(saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()


for episode in range(1000):
    state, info = env.reset()
    saved_log_probs = []
    rewards = []
    for t in range(10000):
        action, log_prob = select_action(policy_net, state)
        saved_log_probs.append(log_prob)
        state, reward, done, terminated, info = env.step(action)
        rewards.append(reward)
        if done:
            break
    finish_episode(saved_log_probs, rewards)
