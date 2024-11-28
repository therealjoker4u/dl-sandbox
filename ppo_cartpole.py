import os
import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from tensordict import TensorDict
import matplotlib.pyplot as plt

from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedReplayBuffer
import gymnasium as gym

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

ic(torch.cuda.is_available())
ic(device)
device = "cpu"


ACTOR_MODEL_FILE = "./models/ppo_cartpole.pth"
CRITIC_MODEL_FILE = "./models/ppo_cartpole_critic.pth"

_env_kwargs = {
    "render_mode": "human"
}

env = gym.make('CartPole-v1',
               **_env_kwargs
               )


class PolicyNetwork(nn.Module):
    def __init__(self, num_actions=2, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act = nn.SELU(inplace=False)

        self.fc1 = nn.LazyLinear(128, device=device)

        self.fc2 = nn.LazyLinear(128, device=device)

        self.fc3 = nn.LazyLinear(128, device=device)

        self.out_layer = nn.LazyLinear(num_actions, device=device)

    def forward(self, y: torch.Tensor):
        y = self.act(self.fc1(y))
        y = self.act(self.fc2(y))
        y = self.act(self.fc3(y))
        y = self.out_layer(y)
        y = F.softmax(y, 0 if len(y.shape) == 1 else -1)

        return y


class ValueNetwork(nn.Module):
    def __init__(self,  device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act = nn.SELU(inplace=False)

        self.fc1 = nn.LazyLinear(128, device=device)

        self.fc2 = nn.LazyLinear(128, device=device)

        self.fc3 = nn.LazyLinear(128, device=device)

        self.out_layer = nn.LazyLinear(1, device=device)

    def forward(self, y: torch.Tensor):
        y = self.act(self.fc1(y))
        y = self.act(self.fc2(y))
        y = self.act(self.fc3(y))
        y = self.out_layer(y)
        return y


action_spec_size = 2
policy = PolicyNetwork(action_spec_size, device=device).eval()
critic = ValueNetwork(device=device).eval()


def dump_gradients(module: nn.Module):
    result = ""
    for name, param in module.named_parameters():
        if param.grad is not None and not "bias" in name:
            result += f"{name}: {round(param.grad.norm().item(), 4)}\n"
    return result


is_training = True
if "render_mode" in _env_kwargs and _env_kwargs["render_mode"] == "human":
    policy.eval()
    is_training = False

if os.path.exists(ACTOR_MODEL_FILE):
    policy.load_state_dict(torch.load(ACTOR_MODEL_FILE, weights_only=False))
    print("Pretrained actor loaded")
    policy.eval()

if os.path.exists(CRITIC_MODEL_FILE):
    critic.load_state_dict(torch.load(CRITIC_MODEL_FILE, weights_only=False))
    print("Pretrained critic loaded")
    critic.eval()


policy_optimizer = torch.optim.AdamW(
    policy.parameters(), lr=0.0002, amsgrad=True, weight_decay=0.001)

critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.002)

optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=policy_optimizer, gamma=0.999)


reward_history = []
loss_history = []
loss_history_critic = []
rb = ReplayBuffer(
    storage=LazyTensorStorage(max_size=10000, device=device)
)


def select_action(policy: PolicyNetwork, observation: TensorDict, differentiable=False):
    with torch.set_grad_enabled(differentiable):
        y: torch.Tensor = policy(observation)
    if is_training:
        dist = torch.distributions.Categorical(y)
        action = dist.sample((1, ))
        prob = y[action.item()].detach()
    else:
        action = y.argmax(-1, keepdim=True)
        prob = None
    return (action, prob)


def optimize_policy_and_critic(policy: PolicyNetwork, sample: TensorDict, eps=0.2):
    policy_optimizer.zero_grad()
    probs: torch.Tensor = policy(
        sample["observation"]).gather(-1, sample["action"]).squeeze()
    p_ratio = probs / sample["probs"]

    clip_p_ratio = p_ratio.clip(1 - eps, 1 + eps)
    loss1 = (p_ratio * sample["advantage"].squeeze()).mean().unsqueeze(0)
    loss2 = (clip_p_ratio * sample["advantage"].squeeze()).mean().unsqueeze(0)
    loss = -torch.cat([loss1, loss2]).min()

    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 0.6)
    policy_optimizer.step()

    return (loss, )


max_reward_ever = 0
losses_history = []
gamma = 0.99
lmbda = 0.96

for episode in range(2000):
    obs, _ = env.reset()
    obs = torch.from_numpy(obs).to(
        dtype=torch.float32, device=device)

    episode_rewards = []
    transitions = []

    for t in range(500):
        action, prob = select_action(policy, obs,)
        obs_next, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_rewards.append(-10.0 if terminated else 1.0)

        obs_next = torch.from_numpy(obs_next).to(
            dtype=torch.float32, device=device)

        if is_training:
            next = TensorDict(
                {"observation": obs_next}, device=device)

            transition = TensorDict({
                "action": action,
                "observation": obs,
                "done": torch.tensor(done, dtype=torch.bool, device=device),
                "next": next,
                "probs": prob
            }, device=device)

            transitions.append(transition)

        obs = obs_next

        if done:
            break

    total_reward = sum(episode_rewards)
    if total_reward > max_reward_ever:
        max_reward_ever = total_reward

    if is_training:
        critic_optimizer.zero_grad()
        returns = []
        estimated_values = []
        advantages = []
        for t, (transition, reward) in enumerate(zip(transitions, episode_rewards)):
            for i in range(t + 1, len(episode_rewards)):
                reward += episode_rewards[i] * (gamma ** (i - t))

            estimated_value = critic(transition["observation"])
            returns.append(reward)
            estimated_values.append(estimated_value)
            advantage = reward - estimated_value
            transition["advantage"] = advantage.detach()
            rb.add(transition)

        # Calculating the loss of values between real returns and estimated values
        returns = torch.tensor(returns, device=device).unsqueeze(1).detach()
        estimated_values = torch.cat(estimated_values).unsqueeze(1)
        critic_td_error = F.smooth_l1_loss(returns, estimated_values)
        # Optimizing the critic
        critic_td_error.backward()
        critic_optimizer.step()

        loss_history_critic.append(critic_td_error.item())
        torch.save(critic.state_dict(), CRITIC_MODEL_FILE)

    loss = None
    if is_training and len(rb) >= 256:
        sample, info = rb.sample(256, return_info=True)
        avg10_loss = sum(losses_history) / \
            len(losses_history) if len(losses_history) > 0 else 1.0

        policy.train()
        (loss, ) = optimize_policy_and_critic(policy, sample)
        policy.eval()

        losses_history.append(loss.item())
        if len(losses_history) > 10:
            losses_history = losses_history[-10:]
        optimizer_scheduler.step()
        torch.save(policy.state_dict(), ACTOR_MODEL_FILE)

    with torch.no_grad():
        avg_loss = round(loss.item(), 4) if loss != None else 0
        total_reward = round(total_reward, 4)

    print(
        f"Episode #{episode} (Max steps : {t}) | Avg Loss : {avg_loss} | Total Reward : {total_reward} | LR : {optimizer_scheduler.get_last_lr()[0]:8f}")

    if avg_loss > 0:
        loss_history.append(avg_loss)
        reward_history.append(total_reward)

    if is_training:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        ax1.plot(range(len(loss_history)), loss_history)
        ax1.set_title("Policy Loss")

        ax2.plot(range(len(loss_history_critic)), loss_history_critic)
        ax2.set_title("Critic Loss")

        ax3.plot(range(len(reward_history)), reward_history)
        ax3.set_title("Total Reward")

        plt.tight_layout()
        plt.savefig("./.tmp/ppo_cartpole.png")
        plt.close()
