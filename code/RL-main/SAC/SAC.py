import os
import gym
# import wandb
import argparse
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, Normal
from torch import nn, optim
from tensorboardX import SummaryWriter
from pathlib import Path
import copy

device = th.device("cuda:1" if th.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        
        self.l3 = nn.Linear(256, action_dim) # mu
        self.l4 = nn.Linear(256, action_dim) # sigma
        

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        mu = self.l3(x)
        log_std = F.relu(self.l4(x))
        log_std = th.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        std = log_std.exp()
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = th.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = th.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            th.FloatTensor(self.state[ind]).to(self.device),
            th.FloatTensor(self.action[ind]).to(self.device),
            th.FloatTensor(self.next_state[ind]).to(self.device),
            th.FloatTensor(self.reward[ind]).to(self.device),
            th.FloatTensor(self.not_done[ind]).to(self.device)
        )



class SAC(object):
    def __init__(self, state_dim, action_dim, discount=0.99, tau=0.005, policy_freq=2, ent_coef=0.01):

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        self.ent_coef = ent_coef

        self.total_it = 0

    def select_action(self, state):
        state = th.tensor(state, dtype=th.float32).to(device)
        action_mu, action_std = self.actor(state)
        dist = Normal(action_mu, action_std)
        action = th.tanh(dist.sample())
        return action.detach().cpu().numpy()

    def evaluate(self, state):
        action_mean, action_std = self.actor(state)
        dist = Normal(action_mean, action_std)
        noise = Normal(0, 1)
        z = noise.sample()
        final_action = th.tanh(action_mean+action_std*z.to(device))
        log_prob = dist.log_prob(action_mean+action_std*z.to(device)) - th.log(1 - final_action.pow(2) + 1e-6)
        return final_action, log_prob

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with th.no_grad():
            # next_mu, next_std = self.actor(next_state)
            # next_dist = Normal(next_mu, next_std)
            # next_action = th.tanh(next_dist.sample())
            next_action, _ = self.evaluate(next_state)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = th.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            # action_mean, action_std = self.actor(state)
            # dist = Normal(action_mean, action_std)
            # curr_action = dist.sample()
            # final_action = th.tanh(curr_action)
            # log_prob = dist.log_prob(curr_action) - th.log(1 - final_action.pow(2) + 1e-6)
            final_action, log_prob = self.evaluate(state)

            entropies = -log_prob.sum(dim=1, keepdim=True)
            curr_Q1, curr_Q2 = self.critic(state, final_action)
            curr_Q = th.min(curr_Q1, curr_Q2)
            actor_loss = -(self.ent_coef * entropies + curr_Q).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, path, num):
        th.save(self.actor.state_dict(), "{}/actor_{}.th".format(path, num))
        th.save(self.critic.state_dict(), "{}/critic_{}.th".format(path, num))
        th.save(self.actor_optim.state_dict(), "{}/actor_optim_{}.th".format(path, num))
        th.save(self.critic_optim.state_dict(), "{}/critic_optim_{}.th".format(path, num))


    def load(self, filename, num):
        self.critic.load_state_dict(th.load(filename + "/critic_{}.th".format(num)))
        self.critic_optimizer.load_state_dict(th.load(filename + "/critic_optim_{}.th".format(num)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(th.load(filename + "/actor_{}.th".format(num)))
        self.actor_optimizer.load_state_dict(th.load(filename + "/actor_optim_{}.th".format(num)))


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def main(args, logger=None):
    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discount": args.discount,
        "tau": args.tau,
        "policy_freq": args.policy_freq,
        "ent_coef": args.ent_coef
    }

    # Initialize policy
    policy = SAC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    action_range = [env.action_space.low, env.action_space.high]

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state))
        action_in =  action * (action_range[1] - action_range[0]) / 2.0 +  (action_range[1] + action_range[0]) / 2.0
        # Perform action
        next_state, reward, done, _ = env.step(action_in) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            logger.add_scalar("Episode_Reward", episode_reward, t+1)
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            test_reward = eval_policy(policy, args.env, args.seed)
            # np.save(f"./results/{file_name}", evaluations)
            logger.add_scalar("Test_Reward", test_reward, t+1)
            if args.save_model: 
                os.makedirs(str(args.run_dir / 'incremental'), exist_ok=True)
                policy.save(str(args.run_dir / 'incremental'), episode_num)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")                  # Policy name
    parser.add_argument("--env", default="HalfCheetah-v2")          # Mujoco environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=2e4, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--lr", default=3e-4)                       # Learning rate
    parser.add_argument("--ent_coef", default=0.01)                 # Entropy coefficient
    parser.add_argument("--policy_freq", default=1, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--use_cuda", default=True, type=bool)
    args = parser.parse_args()
    

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not args.use_cuda:
        device = "cpu"

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    model_dir = Path('./models') / args.policy / args.env
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
