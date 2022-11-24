import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from agent_dir.agent import Agent


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        self.input_size = input_size[0]
        # ACTOR
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
        return F.softmax(inputs, dim=1)

class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        self.input_size = input_size[0]
        # Critic
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
        return inputs


class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Trajectory buffer. It will clear the buffer after updating.
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer_rewards = []
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_size = buffer_size

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        ##################
        return len(self.buffer_rewards)

    def push(self, reward, obs, action):
        ##################
        # YOUR CODE HERE #
        ##################
        if len(self.buffer_rewards) == self.buffer_size:
            self.buffer_rewards.pop(0)
            self.buffer_obs.pop(0)
            self.buffer_actions.pop(0)
        self.buffer_rewards.append(reward)
        self.buffer_obs.append(obs)
        self.buffer_actions.append(action)

    def sample(self, batch_size):
        """
        Sample all the data stored in the buffer
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.buffer_rewards, self.buffer_obs, self.buffer_actions

    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer_rewards.clear()
        self.buffer_obs.clear()
        self.buffer_actions.clear()


class AgentA2C(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentA2C, self).__init__(env)
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
        self.actor = ActorNetwork(self.input_size, self.hidden_size, self.output_size)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic = CriticNetwork(self.input_size, self.hidden_size, 1)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)
        self.max_steps = 300  # 最大步数
        self.buffer_size = self.max_steps
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
        self.actor.load_state_dict(torch.load('a2cnetwork_params.pth'))

    def actorTrain(self, final_reward):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        rewards, obs, actions = self.buffer.sample(self.batch_size)
        pro_actions = self.actor(obs)
        vs = self.critic(obs).detach()
        discounted_r = np.zeros_like(rewards)
        for i in reversed(range(0, len(rewards))):
            final_reward = final_reward * self.gamma + rewards[i]
            discounted_r[i] = final_reward
        qs = torch.Tensor(discounted_r)

        adv_fun = qs - vs
        self.actor_optim.zero_grad()
        # print(actions)
        # torch.log(pro_actions) * torch.tensor(actions)
        loss = - torch.mean(torch.sum(torch.log(pro_actions) * torch.tensor(actions), 1)*adv_fun)
        loss.backward()
        self.actor_optim.step()
        return qs

    def criticTrain(self, qs):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        rewards, obs, actions = self.buffer.sample(self.batch_size)
        values = self.critic(obs)
        # print(qs)
        qslist = []
        for i in range(len(qs)):
            qslist.append([qs[i]])
        qs = torch.tensor(qslist)
        # print(qs)
        loss = F.mse_loss(values, qs)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        #
        probability = self.actor(observation)
        m = Categorical(probability)
        action = m.sample()
        return action.item()

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        torch.manual_seed(self.seed)
        writer = SummaryWriter('./log')
        epi_array = []
        step = 0
        for i_episode in range(self.n_frames):
            final_reward = 0
            obs,  episode_reward = self.env.reset(), 0
            for step in range(self.max_steps):
                step+=1
                action = self.make_action(obs, self.test)
                next_obs, reward, done, info = self.env.step(action)
                one_hot_action = [int(k == action) for k in range(self.output_size)]
                self.buffer.push(reward, obs, one_hot_action)
                episode_reward += reward
                obs = next_obs
                self.batch_size += 1
                if done:
                    break
            if not done:
                final_reward = self.critic(obs)
            qs = self.actorTrain(final_reward)
            self.criticTrain(qs)
            self.buffer.clean()
            print("episode" + str(i_episode) + ";reward:" + str(episode_reward))
            writer.add_scalar('a2c_reward2', episode_reward, i_episode)
            torch.save(self.actor.state_dict(), 'a2cnetwork_params2.pth')  # 只保存网络中的参数

