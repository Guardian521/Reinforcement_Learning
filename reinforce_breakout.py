import numpy as np
import gym
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import cv2
from itertools import count
import random, math

device = torch.device("cpu")
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))

## 超参数
batch_size = 32
Gamma = 0.99 #折扣因子
eps_start = 1
eps_end = 0.02
eps_decay = 1000000 #epsilon_greedy策略
eps_random_count = 50000  # 前50000步纯随机用于探索
target_update = 30000000000  # steps
render = False
lr = 1e-4 #学习率
initial_memory = 10000
memory_size = 10 * initial_memory
n_episode = 100  # 10000000

model_store_path = '/Users/gushuai/Desktop/bigdata/上的各种课/大三下/数据分析2/models'  # +'DQN_Network_pytorch_pong'
modelname = 'DQN_Network_Breakout'
madel_path = model_store_path + 'DQN_Network_Breakout_episode60.pt'  # + '/' + 'model/'

#记忆池
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # 移动指针，经验池满了之后从最开始的位置开始将最近的经验存进经验池

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # 从经验池中随机采样

    def __len__(self):
        return len(self.memory)


class DQN_Network(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        super(DQN_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):

        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 将卷积层的输出展平
        x = F.relu(self.fc4(x))
        out = self.head(x)
        return out


class DQN_Network_agent():
    def __init__(self, in_channels=4, action_space=[], learning_rate=1e-4, memory_size=10000, epsilon=1,
                 trained_model_path=''):

        self.in_channels = in_channels
        self.action_space = action_space
        self.action_dim = self.action_space.n

        self.memory_buffer = ReplayMemory(memory_size)
        self.stepdone = 0
        self.DQN_Network = DQN_Network(self.in_channels, self.action_dim).to(device)
        self.target_DQN_Network = DQN_Network(self.in_channels, self.action_dim).to(device)
        if (trained_model_path != ''):
            self.DQN_Network.load_state_dict(torch.load(trained_model_path))

        self.target_DQN_Network.load_state_dict(self.DQN_Network.state_dict())
        self.optimizer = optim.RMSprop(self.DQN_Network.parameters(), lr=learning_rate, eps=0.001, alpha=0.95)

    def select_action(self, state):

        self.stepdone += 1
        state = state.to(device)
        epsilon = eps_end + (eps_start - eps_end) * \
                  math.exp(-1. * self.stepdone / eps_decay)  # 随机选择动作系数epsilon 衰减，也可以使用固定的epsilon
        # epsilon-greedy策略选择动作
        if self.stepdone < eps_random_count or random.random() < epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        else:
            action = self.DQN_Network(state).detach().max(1)[1].view(1, 1)  # 选择Q值最大的动作并view

        return action

    def learn(self):
        # 经验池小于batch_size则直接返回
        if self.memory_buffer.__len__() < batch_size:
            return
        # 从经验池中采样

        transitions = self.memory_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

        # 判断是不是在最后一个状态，最后一个状态的next设置为None
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8).bool()

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(device)

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        # 计算当前状态的Q值
        state_action_values = self.DQN_Network(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = self.target_DQN_Network(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * Gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.DQN_Network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


class Trainer():
    def __init__(self, env, agent, n_episode):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        # self.losslist = []
        self.rewardlist = []
        self.avg_rewardlist = []

    # 获取当前状态，将env返回的状态通过transpose调换轴后作为状态
    def get_state(self, obs):
        # print(obs.shape)
        state = np.array(obs)
        # state = state.transpose((1, 2, 0)) #将2轴放在0轴之前
        state = torch.from_numpy(state)
        return state.unsqueeze(0)  # 转化为四维的数据结构

    # 训练智能体
    def train(self):
        for episode in range(self.n_episode):

            obs = self.env.reset()
            state = np.stack((obs[0], obs[1], obs[2], obs[3]))
            state = self.get_state(state)
            episode_reward = 0.0

            for t in count():
                action = self.agent.select_action(state)
                if render:
                    self.env.render()

                obs, reward, done, _, _ = self.env.step(action)
                episode_reward += reward

                if not done:

                    next_state = np.stack((obs[0], obs[1], obs[2], obs[3]))
                    next_state = self.get_state(next_state)
                else:
                    next_state = None
                reward = torch.tensor([reward], device=device)

                # 将四元组存到memory中

                self.agent.memory_buffer.push(state, action.to('cpu'), next_state, reward.to('cpu'))  # 里面的数据都是Tensor


                state = next_state
                # 经验池满了之后开始学习
                if self.agent.stepdone > initial_memory:
                    self.agent.learn()
                    if self.agent.stepdone % target_update == 0:
                        print('======== target DQN_Network updated =========')
                        self.agent.target_DQN_Network.load_state_dict(self.agent.DQN_Network.state_dict())

                if done:
                    break
            agent_epsilon = eps_end + (eps_start - eps_end) * math.exp(-1. * self.agent.stepdone / eps_decay)

            print('Total steps: {} \t Episode/steps: {}/{} \t Total reward: {} \t Avg reward: {} \t epsilon: {}'.format(
                self.agent.stepdone, episode, t, episode_reward, episode_reward / t, agent_epsilon))

            if episode % 20 == 0:
                torch.save(self.agent.DQN_Network.state_dict(),
                           model_store_path + '/' + "{}_episode{}.pt".format(modelname, episode))

            self.rewardlist.append(episode_reward)
            self.avg_rewardlist.append(episode_reward / t)

            self.env.close()
        return

    # 绘制单幕总奖励曲线
    def plot_total_reward(self):
        plt.plot(self.rewardlist)
        plt.xlabel("Training epochs")
        plt.ylabel("Total reward per episode")
        plt.title('Total reward curve of DQN_Network on Skiing')
        plt.savefig('DQN_Network_train_total_reward.png')
        plt.show()

    # 绘制单幕平均奖励曲线
    def plot_avg_reward(self):
        plt.plot(self.avg_rewardlist)
        plt.xlabel("Training epochs")
        plt.ylabel("Average reward per episode")
        plt.title('Average reward curve of DQN_Network on Skiing')
        plt.savefig('DQN_Network_train_avg_reward.png')
        plt.show()


# 奖励裁剪
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)


# 对环境的观测数据进行预处理，将帧图像转换为指定的尺寸（默认为 84x84），并将其转换为灰度图像。
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


# Frame Stacking
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """
        Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        if isinstance(ob, tuple):  # obs is tuple in newer version of gym
            ob = ob[0]
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        if isinstance(ob, tuple):
            ob = ob[0]
        self.frames.append(ob)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN_Network and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN_Network's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]


def env_wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True):
    """
    Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)  # scale the frame image
    if clip_rewards:
        env = ClipRewardEnv(env)  # clip the reward into [-1,1]
    if frame_stack:
        env = FrameStack(env, 4)  # stack 4 frame to replace the RGB 3 chanels image
    return env


# create environment and warp it into DeepMind style
env = env_wrap_deepmind(gym.make("ALE/Breakout-v5"), episode_life=True, clip_rewards=False, frame_stack=True,
                        scale=False)
action_space = env.action_space

# use 4 stacked chanel
agent = DQN_Network_agent(in_channels=4, action_space=action_space, learning_rate=lr, memory_size=memory_size)

trainer = Trainer(env, agent, n_episode)  # 这里应该用超参数里的n_episode
trainer.train()
trainer.plot_total_reward()
# trainer.plot_avg_reward()

# save total reward list
np.save('total_reward_list_breakout_1e5.npy', np.array(trainer.rewardlist))
np.save('avg_reward_list_breakout_1e5.npy', np.array(trainer.avg_rewardlist))

print('The training costs {} episodes'.format(len(trainer.rewardlist)))

print('The max episode reward is {}, at episode {}'.format(
    max(trainer.rewardlist),
    trainer.rewardlist.index(max(trainer.rewardlist))
))

# 合并5 episodes为1episode
assert (len(trainer.rewardlist) % 5 == 0)
reshaped_reward_array = np.array(trainer.rewardlist).reshape((int(len(trainer.rewardlist) / 5), 5))
# 沿着第二个维度求和
summed_rewawrd_array = reshaped_reward_array.sum(axis=1)

print('Now takes 5 episodes as 1, the training cost {} complete episodes'.format(len(summed_rewawrd_array)))

print('The max episode return is {}, at episode {}'.format(
    max(summed_rewawrd_array),
    np.where(summed_rewawrd_array == max(summed_rewawrd_array))
))

# 合并200 episodes为1episode
assert (len(summed_rewawrd_array) % 200 == 0)
reshaped_reward_array_200 = summed_rewawrd_array.reshape((int(len(summed_rewawrd_array) / 200), 200))
# 沿着第二个维度求和
summed_rewawrd_array_200 = reshaped_reward_array_200.sum(axis=1)
avg_rewawrd_array_200 = summed_rewawrd_array_200 / 200.0

np.save('avg_rewawrd_array_200_breakout_1e5.npy', avg_rewawrd_array_200)

print('The following graph takes 1000 games as 1 epoch where 5 games equals to 1 episode as stated before')

max_idx = np.argmax(avg_rewawrd_array_200)
max_y = max(avg_rewawrd_array_200)
print('The best average return per epoch is {}, at epoch {}'.format(max_idx, max_y))

plt.figure(figsize=(10, 6))
plt.plot(avg_rewawrd_array_200, marker='o', markersize=4)
plt.xlabel("Training Epochs", fontsize=12)
plt.ylabel("Average Reward per Episode", fontsize=12)

plt.scatter(max_idx, max_y, color='red', s=60)
plt.annotate(f'max avg return: ({max_idx}, {max_y:.2f})', xy=(max_idx, max_y), xytext=(max_idx - 40, max_y - 1),
             arrowprops=dict(facecolor='red', shrink=0.05))
# plt.title('Average Reward of DQN_Network on Breakout')
plt.savefig('DQN_Network_train_total_reward_Breakout.svg')
plt.show()
