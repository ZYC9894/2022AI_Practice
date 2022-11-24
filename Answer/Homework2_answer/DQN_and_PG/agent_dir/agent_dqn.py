import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from agent_dir.agent import Agent
import math
import torch.autograd as autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, output_size)

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        inputs = F.relu(self.conv1(inputs))
        inputs = F.relu(self.conv2(inputs))
        inputs = F.relu(self.conv3(inputs))
        inputs = F.relu(self.fc4(inputs.reshape(inputs.size(0), -1)))
        return self.fc5(inputs)


class ReplayBuffer:
    def __init__(self, buffer_size):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer = []
        self.buffer_size = buffer_size

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        ##################
        return len(self.buffer)

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        ##################
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        ##################
        index = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)

    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.hidden_size = args.hidden_size
        # 图像转向量
        print(env.observation_space.shape)
        self.input_size = env.observation_space.shape  # 4*84*84
        self.output_size = env.action_space.n
        self.eval_network = QNetwork(self.input_size, self.output_size).to(device)
        self.target_network = QNetwork(self.input_size, self.output_size).to(device)
        self.target_network.load_state_dict(self.eval_network.state_dict())
        self.optim = optim.Adam(self.eval_network.parameters(), lr=args.lr)
        self.gamma = args.gamma  # 学习率
        self.buffer = ReplayBuffer(args.buffer_size)
        self.buffer_size = args.buffer_size  # buffer大小
        self.loss_func = nn.MSELoss()
        self.learn_step = 0  # 计步器
        self.eps = 1  # 贪婪决策概率
        self.eps_start = 1  #初始贪婪决策概率
        self.eps_end = 0.01  # 最终贪婪决策概率
        self.grad_norm_clip = args.grad_norm_clip  # 梯度剪裁
        self.target_update_freq = args.target_update_freq  # target更新频率
        self.batch_size = args.batch_size  # 批量大小
        self.n_frames = args.n_frames
        self.test = args.test
        # self.target_network.load_state_dict(torch.load('network_params.pth'))
        # self.eval_network.load_state_dict(torch.load('network_params.pth'))

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        # 导入网络
        # self.target_network.load_state_dict(torch.load('PongNoFrameskip-v4_dqn.pth',map_location='cpu'))
        #self.target_network = torch.load('PongNoFrameskip-v4_dqn.pth',map_location='cpu')
        self.target_network.load_state_dict(torch.load('network_params.pth',map_location='cpu'))

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        actions = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(device)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(device)

        q_eval = self.eval_network(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.target_network(next_obs).detach()
        q_target = rewards + self.gamma * (1 - dones) * torch.max(q_next, dim=-1)[0]
        Loss = self.loss_func(q_eval, q_target)
        self.optim.zero_grad()
        Loss.backward()
        self.optim.step()
        return Loss.item()
        # print("loss:"+str(loss))

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if test:
            self.eps = 0.01
        else:
            self.eps = self.eps - (self.eps_start - self.eps_end)/100000
            self.eps = max(self.eps, self.eps_end)
        if np.random.uniform() <= self.eps:
            action = np.random.randint(0, self.output_size)
        else:
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
            action_value = self.target_network(observation)
            # print(action_value)
            action = torch.max(action_value, dim=-1)[1].cpu().numpy()
            # action = np.random.choice(action, 1)
        return int(action)

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        writer = SummaryWriter('./log')
        print(self.eps_end)
        step = 0
        for i_episode in range(self.n_frames):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            loss = 0
            while not done:
                loss_ = []
                action = self.make_action(obs, self.test)
                next_obs, reward, done, info = self.env.step(action)
                self.buffer.push(obs, action, reward, next_obs, done)
                episode_reward += reward
                obs = next_obs
                if step >= self.batch_size*300:
                    loss_.append(self.train())
                if step % 1000 == 0:
                    self.target_network.load_state_dict(self.eval_network.state_dict())
                step += 1
                if done:
                    if len(loss_):
                        loss = sum(loss_)/len(loss_)
                    break

            print("episode" + str(i_episode) + ";reward:" + str(episode_reward))
            writer.add_scalar('dqn_reward7', episode_reward, i_episode)
            # print("len:"+str(self.buffer.__len__()))
            print("eps:" + str(self.eps),"loss:",loss)
            # print(self.test)
        # 保存网络
        torch.save(self.target_network.state_dict(), 'network_params.pth')  # 只保存网络中的参数