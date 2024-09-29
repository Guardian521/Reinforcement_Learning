import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# 定义价值网络
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化环境和网络
env = gym.make("ALE/Breakout-v5", render_mode='human')
env = AtariPreprocessing(env, frame_skip=1)
env = FrameStack(env, num_stack=4)

input_dim = env.observation_space.shape
output_dim = env.action_space.n
policy_network = PolicyNetwork(input_dim[0], output_dim)
value_network = ValueNetwork(input_dim[0])
optimizer_policy = optim.Adam(policy_network.parameters(), lr=0.0001)
optimizer_value = optim.Adam(value_network.parameters(), lr=0.0001)

# 定义超参数
gamma = 0.99  # 折扣因子
epochs = 1000  # 迭代次数
batch_size = 2000  # 每次优化步骤的批量大小
clip_param = 0.1  # PPO裁剪参数
value_loss_coef = 0.5  # 价值损失系数
entropy_coef = 0.01  # 熵损失系数
max_grad_norm = 0.5  # 最大梯度范数
epsilon = 0.1  # 探索策略

# PPO算法主循环
for epoch in range(epochs):
    states = []
    actions = []
    rewards = []
    values = []
    masks = []

    # 收集数据
    state, info_dict = env.reset()
    state = np.array(state, dtype=np.float32) / 255.0  # 归一化输入
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    while len(states) < batch_size:
        action_probs = policy_network(state)

        # 检查action_probs是否包含无效值
        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or (action_probs < 0).any():
            print("Invalid action_probs detected, replacing with uniform distribution")
            action_probs = torch.ones_like(action_probs) / action_probs.size(1)

        # 探索策略：ε-贪婪
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.multinomial(action_probs, num_samples=1).item()

        next_state, reward, done, truncated, info = env.step(action)

        next_state = np.array(next_state, dtype=np.float32) / 255.0  # 归一化输入
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value_network(state).item())
        masks.append(1 - done)

        state = next_state
        total_reward += reward
        if done or truncated:
            state, info_dict = env.reset()
            state = np.array(state, dtype=np.float32) / 255.0  # 归一化输入
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    next_values = value_network(state).item()

    # 计算收益和优势
    returns = []
    advantages = []
    G = next_values
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G * masks[i]
        returns.insert(0, G)
        advantages.insert(0, G - values[i])

    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.int64)

    # 更新策略网络
    for _ in range(10):  # 多次更新以保证稳定性
        action_probs = policy_network(states)
        log_action_probs = torch.log(action_probs)
        selected_log_probs = torch.gather(log_action_probs, 1, actions.unsqueeze(1)).squeeze(1)

        ratio = torch.exp(selected_log_probs - selected_log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        entropy_loss = -torch.sum(action_probs * log_action_probs, dim=1).mean()
        total_policy_loss = policy_loss + entropy_coef * entropy_loss

        optimizer_policy.zero_grad()
        total_policy_loss.backward()
        nn.utils.clip_grad_norm_(policy_network.parameters(), max_grad_norm)
        optimizer_policy.step()

    # 更新价值网络
    value_loss = value_loss_coef * nn.MSELoss()(value_network(states).squeeze(), returns)
    optimizer_value.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(value_network.parameters(), max_grad_norm)
    optimizer_value.step()

    # 打印训练过程中的信息
    print(f"Epoch [{epoch}/{epochs}], Reward: {total_reward}")

# 在完成训练后可以保存模型等进一步操作
