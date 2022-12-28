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

device = th.device("cuda:1" if th.cuda.is_available() else "cpu")

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q


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
        ind = np.random.randint(0, self.size, size=batch_size)
        return self.state[ind], self.action[ind], self.reward[ind], self.next_state[ind], self.done[ind]
        
    def clean(self):
        self.ptr = 0
        self.size = 0


class DQN:
    def __init__(self, args, state_dim, hidden_size, action_dim):
        self.action_dim = action_dim
        self.args = args
        self.q_net = QNet(state_dim, hidden_size, action_dim).to(device)
        self.target_net = QNet(state_dim, hidden_size, action_dim).to(device)
        self.optim = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.eps = args.eps
        self.buffer = ReplayBuffer(state_dim, action_dim, args.buffer_size)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
        self.device = device
    
    def choose_action(self, state):
        state = th.tensor(state, dtype=th.float32).to(device)
        if np.random.uniform() <= self.eps:
            action = np.random.randint(0, self.action_dim)
        else:
            action_value = self.q_net(state)
            action = th.max(action_value, dim=-1)[1].cpu().numpy()
        return int(action)

    def store_transition(self, *transition):
        self.buffer.store_transition(*transition)
        
    def learn(self):
        if self.eps > self.args.eps_min:
            self.eps *= self.args.eps_decay

        self.learn_step += 1 
        
        state, actions, rewards, next_state, dones = self.buffer.sample(self.args.batch_size)
        state = th.tensor(state, dtype=th.float32).to(device)
        actions = th.tensor(actions, dtype=th.long).to(device)
        dones = th.tensor(dones, dtype=th.int).to(device)
        rewards = th.tensor(rewards, dtype=th.float32).to(device)
        next_state = th.tensor(next_state, dtype=th.float32).to(device)

        q_eval = self.q_net(state).gather(-1, actions)
        
        q_eval_next = self.q_net(next_state)
        next_actions = th.max(q_eval_next, dim=-1, keepdim=True)[1]
        q_next = self.target_net(next_state).gather(-1, next_actions).detach()
        
        q_target = rewards + self.args.gamma * (1 - dones) * q_next
        
        loss = self.loss_fn(q_eval, q_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.learn_step % self.args.update_target == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
    
    def save(self, path, num):
        th.save(self.q_net.state_dict(), "{}/q_net_{}.th".format(path, num))
        th.save(self.target_net.state_dict(), "{}/target_net_{}.th".format(path, num))
        th.save(self.optim.state_dict(), "{}/optim_{}.th".format(path, num))


def main(args, logger=None):
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(args, state_dim, args.hidden, action_dim) 
    for i_episode in range(args.n_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            if agent.buffer.size >= args.buffer_size:
                agent.learn()
        if args.use_wandb:
            wandb.log({"Reward": episode_reward}, step=i_episode)
        
        logger.add_scalar("Reward", episode_reward, i_episode)
        
        print(f"Episode: {i_episode}, Reward: {episode_reward}")

        if i_episode > 0 and i_episode / args.update_target == 5:
            os.makedirs(str(args.run_dir / 'incremental'), exist_ok=True)
            agent.save(str(args.run_dir / 'incremental'), i_episode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="Double_DQN", type=str)
    parser.add_argument("--env", default="CartPole-v1", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--hidden", default=64, type=int)
    parser.add_argument("--n_episodes", default=1000, type=int)
    parser.add_argument("--gamma",default=0.99, type=float)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--buffer_size", default=10000, type=int)
    parser.add_argument("--eps", default=1.0, type=float)
    parser.add_argument("--eps_min", default=0.05, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--eps_decay", default=0.999, type=float)
    parser.add_argument("--update_target", default=100, type=int)
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--use_wandb", default=False, type=bool)

    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(project="DQN_CartPole", config=args, name="DQN")
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
