import os
import argparse
import gym
import torch as th
import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from pathlib import Path

device = th.device("cuda:1" if th.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.fc2(x), dim=-1)
        return prob


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def push(self, *transition):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self):
        return zip(*self.buffer)

    def clean(self):
        return self.buffer.clear()


class PPO:
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = ReplayBuffer(args.buffer_size)
        self.actor = Actor(self.state_dim, args.hidden, self.action_dim).to(device)
        self.critic = Critic(self.state_dim, args.hidden).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.a_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.c_lr)
        self.loss_fn = nn.MSELoss()
        

    def choose_action(self, state):
        state = th.tensor(state, dtype=th.float32).to(device)
        prob = self.actor(state)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action), log_prob

    def store_transition(self, *transition):
        self.buffer.push(*transition)

    def train(self):
        state, action, reward, next_state, done, old_log_prob = self.buffer.sample()
        state = th.tensor(state, dtype=th.float32).to(device)
        reward = th.tensor(reward, dtype=th.float32).view(-1, 1).to(device)
        action = th.tensor(action, dtype=th.int).to(device)
        done = th.tensor(done, dtype=th.int).view(-1, 1).to(device)
        old_log_prob = th.stack(old_log_prob).detach()
        next_state = th.tensor(next_state, dtype=th.float32).to(device)

        # cal target using old critic
        with th.no_grad():
            target = reward + args.gamma * self.critic(next_state) * (1 - done)
            delta = target - self.critic(state)

            delta = delta.cpu().numpy()
            gae_list = []
            gae = 0.0
            for delta_t in delta[::-1]:
                gae = args.gamma * args.lam * gae + delta_t
                gae_list.append(gae)
            gae_list.reverse()
            adv = th.tensor(gae_list, dtype=th.float32).to(device)

        for _ in range(args.n_update):
            mini_batch = state.shape[0]
            random_index = np.random.choice(0, state.shape[0], mini_batch)
            self.update_actor(state[random_index], action[random_index], adv[random_index], old_log_prob[random_index])
            self.update_critic(state[random_index], target[random_index])
            
    def update_actor(self, state, action, adv, old_log_prob):
        # cal actor loss
        prob = self.actor(state)
        dist = Categorical(prob)
        log_prob = dist.log_prob(action)

        ratio = th.exp(log_prob - old_log_prob)
        surr1 = ratio * adv
        surr2 = th.clamp(ratio, 1-args.epsilon, 1+args.epsilon) * adv
        actor_loss = -th.mean(th.min(surr1, surr2))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
    def update_critic(self, state, target):
        v = self.critic(state)
        critic_loss = self.loss_fn(v, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
    
    def save(self, path, num):
        th.save(self.actor.state_dict(), "{}/actor_{}.th".format(path, num))
        th.save(self.critic.state_dict(), "{}/critic_{}.th".format(path, num))
        th.save(self.actor_optim.state_dict(), "{}/actor_optim_{}.th".format(path, num))
        th.save(self.critic_optim.state_dict(), "{}/critic_optim_{}.th".format(path, num))
        

def main(args, logger=None):
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim, args=args)

    for i_episode in range(args.n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, prob = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done, prob)
            state = next_state
            episode_reward += reward
        if args.use_wandb:
            wandb.log({"Reward": episode_reward})
        logger.add_scalar("Reward", episode_reward, i_episode)
        if i_episode % 10 == 0:
            print(f"Episode: {i_episode}, Reward: {episode_reward}")

        agent.train()
        agent.buffer.clean()

        if i_episode > 0 and i_episode / 100 == 5:
            os.makedirs(str(args.run_dir / 'incremental'), exist_ok=True)
            agent.save(str(args.run_dir / 'incremental'), i_episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="PPO", type=str)
    parser.add_argument("--env", default="CartPole-v0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_episodes", default=3000, type=int)
    parser.add_argument("--a_lr", default=1e-3, type=float)
    parser.add_argument("--c_lr", default=1e-3, type=float)
    parser.add_argument("--hidden", default=64, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--buffer_size", default=10000, type=int)
    parser.add_argument("--n_update", default=10, type=int)

    parser.add_argument("--lam", default=0.8, type=float)

    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--use_wandb", default=False, type=bool)
    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(project="PPO", config=args, name="PPO")
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
