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


ACTOR_MODEL_FILE = "./models/pg_cartpole.pth"

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
        # self.norm1 = nn.BatchNorm1d(128, device=device)

        self.fc2 = nn.LazyLinear(128, device=device)
        # self.norm2 = nn.BatchNorm1d(128, device=device)

        self.fc3 = nn.LazyLinear(128, device=device)
        # self.norm3 = nn.BatchNorm1d(128, device=device)

        self.out_layer = nn.LazyLinear(num_actions, device=device)

    def forward(self, y: torch.Tensor):
        batched_here = False
        if len(y.shape) == 1:
            batched_here = True
            y = y.unsqueeze(0)
        y = self.act(self.fc1(y))
        y = self.act(self.fc2(y))
        y = self.act(self.fc3(y))

        y = F.softmax(self.out_layer(y), -1)
        if batched_here:
            y = y.squeeze()

        return y


action_spec_size = 2
policy = PolicyNetwork(action_spec_size, device=device).eval()


def dump_policy_gradients():
    result = ""
    for name, param in policy.named_parameters():
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


optimizer = torch.optim.Adam(
    policy.parameters(), lr=0.0005, maximize=False)
optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer, gamma=0.999)


reward_history = []
loss_history = []
rb = ReplayBuffer(
    storage=LazyTensorStorage(max_size=10000, device=device)
)


def select_action(observation: TensorDict, differentiable=False):
    y: torch.Tensor = policy(observation)
    dist = torch.distributions.Categorical(y)
    action = dist.sample((1, ))
    return action


def optimize_policy(policy: PolicyNetwork, sample: TensorDict):
    optimizer.zero_grad()
    probs: torch.Tensor = policy(sample["observation"])
    log_probs = probs.gather(-1, sample["action"]).log()
    loss = -(log_probs * sample["next", "reward"].unsqueeze(1))
    loss = loss.squeeze().mean()

    loss.backward()
    # nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
    optimizer.step()

    return (loss, )


max_reward_ever = 0
losses_history = []

for episode in range(2000):
    obs, _ = env.reset()
    obs = torch.from_numpy(obs).to(
        dtype=torch.float32, device=device)

    episode_rewards = []
    transitions = []

    for t in range(500):
        action = select_action(obs,)
        obs_next, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_rewards.append(reward)

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
            }, device=device)

            transitions.append(transition)

        obs = obs_next

        if done:
            break

    total_reward = sum(episode_rewards)
    if total_reward > max_reward_ever:
        max_reward_ever = total_reward
    if is_training:
        torch.save(policy.state_dict(), ACTOR_MODEL_FILE)

    for transition in transitions:
        transition["next", "reward"] = torch.tensor(
            total_reward, dtype=torch.float32, device=device)
        rb.add(transition)

    loss = None
    if is_training and len(rb) >= 256:
        sample, info = rb.sample(256, return_info=True)
        avg10_loss = sum(losses_history) / \
            len(losses_history) if len(losses_history) > 0 else 1.0

        policy.train()
        (loss, ) = optimize_policy(policy, sample)
        policy.eval()

        losses_history.append(loss.item())
        if len(losses_history) > 10:
            losses_history = losses_history[-10:]
        optimizer_scheduler.step()

    with torch.no_grad():
        avg_loss = round(loss.item(), 4) if loss != None else 0
        total_reward = round(total_reward, 4)

    print(
        f"Episode #{episode} (Max steps : {t}) | Avg Loss : {avg_loss} | Total Reward : {total_reward} | LR : {optimizer_scheduler.get_last_lr()[0]:8f}")

    if avg_loss > 0:
        loss_history.append(avg_loss)
        reward_history.append(total_reward)

    if is_training:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.plot(range(len(loss_history)), loss_history)
        ax1.set_title("Avg Loss")

        ax2.plot(range(len(reward_history)), reward_history)
        ax2.set_title("Total Reward")

        plt.tight_layout()
        plt.savefig("./.tmp/pg_cartpole.png")
        plt.close()

    # if total_reward >= 500:
    #     break
