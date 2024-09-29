import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import cv2
from gymnasium.wrappers import AtariPreprocessing, FrameStack

device = torch.device("cpu")
print(device)

transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平卷积层的输出
        x = F.relu(self.fc4(x))
        out = self.head(x)
        return out


class DQNAgent():
    def __init__(self, in_channels=4, action_space=[], trained_model_path=''):
        self.in_channels = in_channels
        self.action_space = action_space
        self.action_dim = self.action_space.n
        self.dqn = DQN(self.in_channels, self.action_dim).to(device)

        if trained_model_path != '':
            self.dqn.load_state_dict(torch.load(trained_model_path, map_location=device))

    def select_action(self, state):
        with torch.no_grad():
            state = state.to(device)
            action = self.dqn(state).max(1)[1].view(1, 1)
        return action


def main():
    env = gym.make('BreakoutNoFrameskip-v4')
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
    env = FrameStack(env, num_stack=4)

    trained_model_path = '/Users/gushuai/Desktop/bigdata/上的各种课/大三下/数据分析2/dqn_breakout_episode2220.pt'
    agent = DQNAgent(in_channels=4, action_space=env.action_space, trained_model_path=trained_model_path)

    obs = env.reset()
    state = np.concatenate([obs] * 4, axis=2)
    state = torch.tensor(state.transpose((2, 0, 1)), dtype=torch.float).unsqueeze(0)  # 转换为四维张量

    for _ in range(1000):  # 玩 10000 个步骤
        env.render()
        action = agent.select_action(state)
        obs, reward, done, info = env.step(action.item())

        next_state = np.concatenate([obs] * 4, axis=2)
        next_state = torch.tensor(next_state.transpose((2, 0, 1)), dtype=torch.float).unsqueeze(0)

        if done:
            break

        state = next_state

    env.close()


if __name__ == '__main__':
    main()
