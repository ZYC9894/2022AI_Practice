import os
import gym
import wandb
import argparse
import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn, optim
from tensorboardX import SummaryWriter
from pathlib import Path
import copy

device = th.device("cuda:1" if th.cuda.is_available() else "cpu")


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.action = nn.Linear(fc2_dim, action_dim)

        self.apply(weight_init)

    def forward(self, state):
        x = th.relu(self.ln1(self.fc1(state)))
        x = th.relu(self.ln2(self.fc2(x)))
        action = th.tanh(self.action(x)) # 0-1
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(action_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)
        self.apply(weight_init)

    def forward(self, state, action):
        x_s = th.relu(self.ln1(self.fc1(state)))
        x_s = self.ln2(self.fc2(x_s))
        x_a = self.fc3(action)
        x = th.relu(x_s + x_a)
        q = self.q(x)
        return q


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_index = 0

        self.state = np.zeros((self.buffer_size, state_dim))
        self.action = np.zeros((self.buffer_size, action_dim))
        self.reward = np.zeros((self.buffer_size, ))
        self.next_state = np.zeros((self.buffer_size, state_dim))
        self.terminal = np.zeros((self.buffer_size, ), dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        current_idx = self.buffer_index % self.buffer_size

        self.state[current_idx] = state
        self.action[current_idx] = action
        self.reward[current_idx] = reward
        self.next_state[current_idx] = next_state
        self.terminal[current_idx] = done

        self.buffer_index += 1

    def sample(self):
        buffer_len = min(self.buffer_size, self.buffer_index)
        batch_index = np.random.choice(buffer_len, self.batch_size, replace=False)

        states = self.state[batch_index]
        actions = self.action[batch_index]
        rewards = self.reward[batch_index]
        next_states = self.next_state[batch_index]
        terminals = self.terminal[batch_index]

        return states, actions, rewards, next_states, terminals

    def ready(self):
        return self.buffer_index >= self.batch_size



class DDPG:
    def __init__(self, env, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim, critic_fc1_dim, critic_fc2_dim, args):
        self.env = env
        
        self.actor = Actor(state_dim, action_dim, actor_fc1_dim, actor_fc2_dim).to(device)
        self.critic = Critic(state_dim, action_dim, critic_fc1_dim, critic_fc2_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr)
        
        self.tau = args.tau
        self.gamma = args.gamma
        self.action_noise = args.action_noise
        self.buffer = ReplayBuffer(args.buffer_size, args.batch_size, state_dim, action_dim)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
        self.device = device
        self.update_network_parameters(tau=1.0)
    
    def choose_action(self, state):
        # self.actor.eval()
        
        state = th.tensor(state, dtype=th.float32).unsqueeze(dim=0).to(device)
        
        action = self.actor.forward(state).squeeze()
        noise = th.tensor(np.random.normal(loc=0.0, scale=self.action_noise), dtype=th.float).to(device)
        action = th.clamp(action+noise, -1, 1) 
        # self.actor.train()
        return action.detach().cpu().numpy()

    def store_transition(self, transition):
        self.buffer.store_transition(*transition)
        
    def learn(self):

        self.learn_step += 1 
        
        state, actions, rewards, next_state, dones = self.buffer.sample()
        
        state = th.tensor(state, dtype=th.float32).to(device)
        actions = th.tensor(actions, dtype=th.float32).to(device)
        dones = th.tensor(dones, dtype=th.int).to(device)
        rewards = th.tensor(rewards, dtype=th.float32).to(device)
        next_state = th.tensor(next_state, dtype=th.float32).to(device)

        with th.no_grad():
            next_actions = self.target_actor(next_state)
            target_q = self.target_critic(next_state, next_actions).view(-1)
            target = rewards + self.gamma * (1 - dones) *  target_q
        '''s,a -> reward'''
        q = self.critic(state, actions).view(-1)  # s,a->reward
        critic_loss = F.mse_loss(q, target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''old actor -> actions, objective: current actor, current actor -> current actions'''
        new_actions = self.actor(state)  # actions, new_actions
        actor_loss = -th.mean(self.critic(state, new_actions))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)

    def save(self, path, num):
        th.save(self.actor.state_dict(), "{}/actor_{}.th".format(path, num))
        th.save(self.target_actor.state_dict(), "{}/target_actor_{}.th".format(path, num))
        th.save(self.critic.state_dict(), "{}/critic_{}.th".format(path, num))
        th.save(self.target_critic.state_dict(), "{}/target_critic_{}.th".format(path, num))
        th.save(self.actor_optimizer.state_dict(), "{}/actor_optimizer_{}.th".format(path, num))
        th.save(self.critic_optimizer.state_dict(), "{}/critic_optimizer_{}.th".format(path, num))


def scale_action(action, high, low):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias
    return action_


def main(args, logger=None):
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(env, state_dim=state_dim, action_dim=action_dim, actor_fc1_dim=400, actor_fc2_dim=300, critic_fc1_dim=400, critic_fc2_dim=300, args=args)
   
    for i_episode in range(args.n_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            action_ = scale_action(action.copy(), env.action_space.high, env.action_space.low)
            next_state, reward, terminated, truncated, info = env.step(action_)
            done = terminated or truncated
            agent.store_transition([state, action, reward, next_state, done])
            episode_reward += reward
            state = next_state
            if agent.buffer.ready():
                agent.learn()
        if args.use_wandb:
            wandb.log({"Reward": episode_reward}, step=i_episode)
        
        logger.add_scalar("Reward", episode_reward, i_episode)
        
        print(f"Episode: {i_episode}, Reward: {episode_reward}")

        if i_episode > 0 and i_episode / args.log_freq == 0:
            os.makedirs(str(args.run_dir / 'incremental'), exist_ok=True)
            agent.save(str(args.run_dir / 'incremental'), i_episode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="DDPG", type=str)
    parser.add_argument("--env", default="LunarLanderContinuous-v2", type=str)  # BOX2D environment name
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--n_episodes", default=1000, type=int)
    parser.add_argument("--gamma",default=0.99, type=float)
    parser.add_argument("--log_freq", default=500, type=int)
    parser.add_argument("--buffer_size", default=1000000, type=int)

    parser.add_argument("--tau", default=0.005, type=float)
    
    parser.add_argument("--action_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--use_wandb", default=False, type=bool)

    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(project="DDPG", config=args, name="DDPG")
        args = wandb.config
    if not args.use_cuda:
        device = "cpu"

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    model_dir = Path('./models') / args.algorithm / args.env
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
						 if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    results_dir = run_dir / 'results'
    os.makedirs(str(log_dir))
    os.makedirs(str(results_dir))
    logger = SummaryWriter(str(log_dir))
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.run_dir = run_dir
    
    main(args, logger)
