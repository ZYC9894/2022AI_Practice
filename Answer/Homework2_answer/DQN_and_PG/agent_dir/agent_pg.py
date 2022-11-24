import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from agent_dir.agent import Agent


class PGNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PGNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        self.input_size = input_size[0]
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        inputs = torch.Tensor(inputs)
        inputs = inputs.view(-1, self.input_size)
        inputs = F.relu(self.fc1(inputs))
        inputs = self.fc2(inputs)
        return F.softmax(inputs,dim=1)


class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Trajectory buffer. It will clear the buffer after updating.
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer_rewards = []
        self.buffer_probs = []
        self.buffer_size = buffer_size

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        ##################
        return len(self.buffer_rewards)

    def push(self, reward, pro):
        ##################
        # YOUR CODE HERE #
        ##################
        if len(self.buffer_rewards) == self.buffer_size:
            self.buffer_rewards.pop(0)
            self.buffer_probs.pop(0)
        self.buffer_rewards.append(reward)
        self.buffer_probs.append(pro)

    def sample(self, batch_size):
        """
        Sample all the data stored in the buffer
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.buffer_rewards, self.buffer_probs

    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer_rewards.clear()
        self.buffer_probs.clear()


class AgentPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentPG, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        self.hidden_size = args.hidden_size
        self.lr = args.lr
        self.gamma = args.gamma
        self.grad_norm_clip = args.grad_norm_clip
        self.test = args.test
        self.n_frames = args.n_frames
        self.seed = args.seed
        self.env = env
        self.input_size = env.observation_space.shape
        self.output_size = env.action_space.n
        self.network = PGNetwork(self.input_size, self.hidden_size, self.output_size)
        self.optim = optim.Adam(self.network.parameters(), lr=args.lr)
        self.buffer_size = 1000
        self.buffer = ReplayBuffer(self.buffer_size)
        self.batch_size = 0

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.network.load_state_dict(torch.load('pgnetwork_params.pth'))

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        rewards, probs = self.buffer.sample(self.batch_size)
        R = 0
        policy_loss = []
        rs = []
        for r in rewards:
            R = r + self.gamma * R
            rs.insert(0,R)
        rs = torch.tensor(rs)
        # 归一化
        rs = (rs - rs.mean()) / (rs.std() + np.finfo(np.float32).eps.item())
        for i in range(len(probs)):
            # 交叉熵
            policy_loss.append(-probs[i] * rs[i])
        self.optim.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.grad_norm_clip)
        self.optim.step()
        self.buffer.clean()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        probability = self.network(observation)
        m = Categorical(probability)
        action = m.sample()
        if test:
            return action.item()
        else:
            r = []
            r.append(action.item())
            r.append(m.log_prob(action))
            return r


    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        torch.manual_seed(self.seed)
        writer = SummaryWriter('./log')
        running_reward = 10
        for i_episode in range(self.n_frames):
            obs,  episode_reward = self.env.reset(),0
            self.batch_size = 0  # 重置
            for step in range(self.buffer_size):
                r = self.make_action(obs, self.test)
                action = r[0]
                pro = r[1]
                next_obs, reward, done, info = self.env.step(action)
                self.buffer.push(reward, pro)
                episode_reward += reward
                obs = next_obs
                self.batch_size += 1
                if done:
                    break
            running_reward = 0.05*episode_reward + (1-0.05) * running_reward
            print("episode" + str(i_episode) + ";reward:" + str(episode_reward))
            writer.add_scalar('reward', episode_reward, i_episode)
            self.train()
            torch.save(self.network.state_dict(), 'pgnetwork_params.pth')  # 只保存网络中的参数