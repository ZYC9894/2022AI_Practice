import argparse
import numpy as np
import gym
from wrappers import make_env
from argument import dqn_arguments, pg_arguments
from torch.utils.tensorboard import SummaryWriter

seed = 11037


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import dqn_arguments
        parser = dqn_arguments(parser)
        # parser = pg_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    writer = SummaryWriter('./log')
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        writer.add_scalar('duelingdqn_episode_reward', episode_reward, i)
        print('reward'+str(episode_reward))
    print('Run %d episodes'%(total_episodes))
    # writer.add_scalar('baseline_mean_reward', np.mean(rewards), 0)
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        # from agent_dir.agent_pg import AgentPG
        # from agent_dir.agent_pg_baseline import AgentPG
        # agent = AgentPG(env, args)
        from agent_dir.agent_A2C import AgentA2C
        agent = AgentA2C(env, args)
        test(agent, env, total_episodes=100)

    if args.test_dqn:
        env_name = args.env_name
        print(args.env_name)
        env = make_env(env_name)
        print(env.observation_space.shape)
        # from agent_dir.agent_dqn import AgentDQN
        from agent_dir.agent_duelingdqn import AgentDQN
        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
