"""
naive A2C
"""
import os
import gym
import argparse
import numpy as np
import wandb
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from pathlib import Path

device = th.device("cuda:1" if th.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input_dim, hidden, out_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = F.log_softmax(self.fc2(x), dim=-1)  # log_prob
        return out


class Critic(nn.Module):
    def __init__(self, input_dim, hidden, out_dim=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)  # evaluate state-value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


def choose_action(prob, action_dim):
    action = np.random.choice(a=action_dim, p=prob[0].detach().cpu().numpy())
    return action


def train_critic(critic_optim, critic, sample):
    obs, one_hot_action, log_probs, reward, next_obs = sample
    obs_v = critic(obs).squeeze(-1)
    next_obs_v = critic(next_obs).squeeze(-1)
    # (Q-V)^2
    critic_loss = (reward + args.gamma * next_obs_v.item() - obs_v)**2 # mse
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()


def train_actor(actor_optim, critic, sample):
    obs, one_hot_action, log_probs, reward, next_obs = sample

    obs_v = critic(obs).squeeze(-1)
    next_obs_v = critic(next_obs).squeeze(-1)

    advantage = reward + next_obs_v.detach() - obs_v.detach()
    
    one_hot_action = one_hot_action.clone().unsqueeze(dim=0)

    
    # -A*log(pi(a|s))
    actor_loss = -advantage * th.sum(log_probs * one_hot_action, dim=-1)
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()


def main(args, logger=None):
    # init env
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # init network
    actor = Actor(input_dim=obs_dim, hidden=args.hidden, out_dim=action_dim).to(device)
    actor_optim = th.optim.Adam(actor.parameters(), lr=args.a_lr)
    critic = Critic(input_dim=obs_dim, hidden=args.hidden, out_dim=1).to(device)
    critic_optim = th.optim.Adam(critic.parameters(), lr=args.c_lr)
    reward_list = []

    for i_episode in range(args.episode):
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            step += 1
            # env.render()
            obs = th.tensor(obs, dtype=th.float32).unsqueeze(dim=0).to(device) # (1,4)
            
            log_probs = actor(obs) # (1,2)
            probs = th.exp(log_probs)
            action = choose_action(probs, action_dim) # int
            one_hot_action = th.eye(action_dim)[action].unsqueeze(dim=0).to(device) # (1, 2)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                reward = -10.0
            episode_reward += reward
            # training
            sample = obs, one_hot_action, log_probs, th.tensor(reward, dtype=th.float32).to(device), th.tensor(next_obs, dtype=th.float32).unsqueeze(dim=0).to(device)
            train_critic(critic_optim, critic, sample)
            train_actor(actor_optim, critic, sample)
            obs = next_obs
        if args.use_wandb:
            wandb.log({"reward": episode_reward})
        reward_list.append(episode_reward)
        logger.add_scalar("Reward", episode_reward, i_episode)
        if i_episode % args.log_freq == 0:
            print("Episode:%d , reward: %f" % (i_episode, episode_reward))
        if i_episode > 0 and step / args.log_freq == 10:
            save_path = str(args.run_dir / 'incremental')
            os.makedirs(save_path, exist_ok=True)
            th.save(actor.state_dict(), "{}/actor_{}.th".format(save_path, step))
            th.save(critic.state_dict(), "{}/critic_{}.th".format(save_path, step))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="naive_A2C", type=str)
    parser.add_argument("--env", default="CartPole-v1", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--a_lr", default=1e-3, type=float)
    parser.add_argument("--c_lr", default=1e-3, type=float)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--episode", default=2000, type=int)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--log_freq", default=50, type=int)
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--use_wandb", default=False, type=bool)
    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(project="A2C", config=args, name="A2C")
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
