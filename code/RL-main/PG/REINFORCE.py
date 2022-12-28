"""
REINFORCE (Policy Gradient) algorithm
"""
import os
import gym
import argparse
import numpy as np
import torch as th
import wandb
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from pathlib import Path

device = th.device("cuda:1" if th.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.fc2(x), dim=-1)
        return prob


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((buffer_size, state_dim))
        self.action = np.zeros((buffer_size, 1))
        self.reward = np.zeros((buffer_size, 1))
        self.next_state = np.zeros((buffer_size, state_dim))
        self.done = np.zeros((buffer_size, 1))

        self.device = device

    def store_transition(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        return self.state[ind], self.action[ind], self.reward[ind], self.next_state[ind], self.done[ind]
        
    def clean(self):
        self.ptr = 0
        self.size = 0


class PG:
    def __init__(self, args, state_dim, hidden_size, action_dim):
        self.args = args
        self.net = Net(state_dim, hidden_size, action_dim).to(device)
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer(state_dim, action_dim, args.buffer_size)
    
    def choose_action(self, state):
        state = th.tensor(state, dtype=th.float32).to(device)
        prob = self.net(state) # distribution
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)  # log pi * Gt
        return int(action), log_prob

    def store_transition(self, *transition):
        self.buffer.store_transition(*transition)

    def train(self):
        state, actions, rewards, next_state, dones = self.buffer.sample(self.buffer.size)
        state = th.tensor(state, dtype=th.float32).to(device)
        actions = th.tensor(actions, dtype=th.long).to(device)
        dones = th.tensor(dones, dtype=th.int).to(device)
        rewards = th.tensor(rewards, dtype=th.float32).to(device)
        next_state = th.tensor(next_state, dtype=th.float32).to(device)
        
        # log_probs = th.log(self.net(state)).gather(-1, actions) # probs, th.log(probs)->log_prob
        prob = self.net(state) # distribution
        dist = Categorical(prob)
        log_probs = dist.log_prob(actions.squeeze(-1)) 
        
        rets = th.zeros_like(rewards, dtype=th.float32).to(device)
        future_rets = 0.0
        for t in reversed(range(rewards.shape[0])):
            future_rets = rewards[t] + self.args.gamma * future_rets
            rets[t] = future_rets
        loss = -(rets.squeeze(-1) * log_probs).sum()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    def save(self, path, num):
        th.save(self.net.state_dict(), "{}/net_{}.th".format(path, num))
        th.save(self.optim.state_dict(), "{}/optim_{}.th".format(path, num))

def main(args, logger=None):
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PG(args, state_dim, args.hidden, action_dim)
    for i_episode in range(args.n_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, log_prob = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
        
        agent.train()
        agent.buffer.clean()  # on policy

        if args.use_wandb:
            wandb.log({"Reward": episode_reward}, step=i_episode)
        logger.add_scalar("Reward", episode_reward, i_episode)
        if i_episode % args.log_freq == 0:
            print(f"Episode: {i_episode}, Reward: {episode_reward}")
        if i_episode > 0 and i_episode / args.log_freq == 20:
            os.makedirs(str(args.run_dir / 'incremental'), exist_ok=True)
            agent.save(str(args.run_dir / 'incremental'), i_episode)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="PG_v2", type=str)
    parser.add_argument("--env", default="CartPole-v1", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--n_episodes", default=3000, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--log_freq", default=20, type=int)
    parser.add_argument("--buffer_size", default=10000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--use_wandb", default=False, type=bool)
    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(project="PG", config=args, name="PG")
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
