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


ACTOR_MODEL_FILE = "./models/dqn_cartpole_sb.pth"

_env_kwargs = {
    "render_mode": "human"
}

env = gym.make('CartPole-v1',
               **_env_kwargs
               )


class QNetwork(nn.Module):
    def __init__(self, num_actions=2, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.LazyLinear(128, device=device)

        self.fc2 = nn.LazyLinear(128, device=device)

        self.fc3 = nn.LazyLinear(128, device=device)

        self.fc4 = nn.LazyLinear(num_actions, device=device)

    def forward(self, observation: torch.Tensor):
        y = F.relu(self.fc1(observation))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))

        y = self.fc4(y)
        return y


def soft_update(target_network: nn.Module, source_network: nn.Module, tau: float):
    """
    Soft updates the target network's parameters towards the source network's parameters.

    Args:
        target_network (nn.Module): The network to be updated.
        source_network (nn.Module): The network providing the new parameters.
        tau (float): The interpolation parameter. Should be between 0 and 1.
                    tau = 0 means no update, tau = 1 means copy the source network.
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data)


action_spec_size = 2
q_net = QNetwork(action_spec_size, device=device).train()

exploration_eps = 1.0
min_exploration_eps = 0.0
exploration_decay = 0.998
is_training = True
if "render_mode" in _env_kwargs and _env_kwargs["render_mode"] == "human":
    q_net.eval()
    exploration_eps = 0.0
    min_exploration_eps = 0.0
    is_training = False

if os.path.exists(ACTOR_MODEL_FILE):
    q_net.load_state_dict(torch.load(ACTOR_MODEL_FILE, weights_only=False))
    print("Pretrained actor loaded")
    exploration_eps = min_exploration_eps

target_q_net = QNetwork(action_spec_size, device=device).eval()
target_q_net.load_state_dict(q_net.state_dict().copy())

optimizer = torch.optim.Adam(
    q_net.parameters(), lr=0.001)


reward_history = []
loss_history = []
rb = PrioritizedReplayBuffer(
    alpha=0.6,
    beta=0.6,
    storage=LazyTensorStorage(max_size=10000, device=device)
)


def select_action(observation: TensorDict, eps: float, differentiable=False):
    exploration_dist = torch.distributions.Bernoulli(eps)
    if eps > 0 and exploration_dist.sample().item() == 1:
        action = torch.randint(0, action_spec_size, (1, ),
                               device=device)
    else:
        with torch.set_grad_enabled(differentiable):
            q_values: torch.Tensor = q_net(observation)
            action = q_values.argmax().unsqueeze(-1)
    del exploration_dist
    return action


mse = nn.MSELoss()


def optimize_q_net(q_net: QNetwork, target_q_net: QNetwork, sample: TensorDict, update_target_net: bool, gamma=0.99, tau=0.01):
    with torch.no_grad():
        non_terminal_coef = (sample["done"] == False).to(dtype=torch.float32)
        target_q_values = target_q_net(
            sample["next", "observation"]).max(dim=-1).values

        target_q_values = gamma * target_q_values * non_terminal_coef
        target_q_values += sample["next", "reward"]

        target_q_values = target_q_values.unsqueeze(-1)

    optimizer.zero_grad()

    q_values: torch.Tensor = q_net(sample["observation"])
    selected_q_values = q_values.gather(-1, sample["action"])
    loss = F.smooth_l1_loss(selected_q_values, target_q_values)
    loss.backward()
    optimizer.step()
    if update_target_net:
        soft_update(target_q_net, q_net, tau)

    td: torch.Tensor = selected_q_values - target_q_values
    return (loss, td.squeeze().abs().detach())


max_reward_ever = 0

for episode in range(2000):
    obs, _ = env.reset()
    obs = torch.from_numpy(obs).to(
        dtype=torch.float32, device=device)

    episode_rewards = []
    transitions = []

    for t in range(500):
        action = select_action(obs, eps=exploration_eps)
        obs_next, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_rewards.append(reward)

        obs_next = torch.from_numpy(obs_next).to(
            dtype=torch.float32, device=device)

        if is_training:
            next = TensorDict(
                {"observation": obs_next, "reward": torch.tensor(-1 if done else 0, dtype=torch.float32, device=device)}, device=device)

            transition = TensorDict({
                "action": action,
                "observation": obs,
                "done": torch.tensor(done, dtype=torch.bool, device=device),
                "next": next,
            }, device=device)
            rb.add(transition)

        obs = obs_next

        if done:
            break

    total_reward = sum(episode_rewards)
    if total_reward > max_reward_ever:
        max_reward_ever = total_reward
    if is_training:
        torch.save(q_net.state_dict(), ACTOR_MODEL_FILE)

    loss = None
    if is_training and len(rb) >= 256:
        sample, info = rb.sample(256, return_info=True)
        loss, priorities = optimize_q_net(
            q_net,
            target_q_net,
            sample.to(device=device),
            update_target_net=episode % 8 == 0,
            tau=0.005
        )
        rb.update_priority(info["index"], priorities)

    with torch.no_grad():
        avg_loss = round(loss.item(), 4) if loss != None else 0
        total_reward = round(total_reward, 4)

    print(
        f"Episode #{episode} (Max steps : {t}) | Avg Loss : {avg_loss} | Total Reward : {total_reward} | E-Greedy epsilon : {round(exploration_eps, ndigits=2)}")

    if avg_loss > 0:
        loss_history.append(avg_loss)
        reward_history.append(total_reward)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(range(len(loss_history)), loss_history)
    ax1.set_title("Avg Loss")

    ax2.plot(range(len(reward_history)), reward_history)
    ax2.set_title("Total Reward")

    plt.tight_layout()
    plt.savefig("./.tmp/dqn_cartpole_sb.png")
    plt.close()

    if exploration_eps > min_exploration_eps:
        exploration_eps *= exploration_decay
        if exploration_eps < min_exploration_eps:
            exploration_eps = min_exploration_eps
