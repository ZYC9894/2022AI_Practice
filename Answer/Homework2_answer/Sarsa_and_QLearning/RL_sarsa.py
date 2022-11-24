
import numpy as np
import pandas as pd


class Sarsa:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        ''' build q table'''
        ############################

        # YOUR IMPLEMENTATION HERE #

        ############################
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        ''' choose action from q table '''
        ############################

        # YOUR IMPLEMENTATION HERE #

        ############################
        self.check_state_exist(observation)
        # best_action = self.q_table.loc[observation, :].idxmax()
        best_action_value = self.q_table.loc[observation, :].max()
        best_action = np.random.choice(self.q_table.loc[observation, :][self.q_table.loc[observation, :] == best_action_value].index)
        random_action = np.random.choice(self.actions)
        if np.random.uniform(0, 1) <= self.epsilon:
            return best_action
        else:
            return random_action

    def learn(self, s, a, r, s_):
        ''' update q table '''
        ############################

        # YOUR IMPLEMENTATION HERE #

        ############################
        a_ = self.choose_action(s_)
        if s_ == 'terminal':
            self.q_table.loc[s, a] += self.lr * (r - self.q_table.loc[s, a])
        else:
            self.q_table.loc[s, a] += self.lr*(r + self.gamma*self.q_table.loc[s_, a_] - self.q_table.loc[s, a])

    def check_state_exist(self, state):
        ''' check state '''
        ############################

        # YOUR IMPLEMENTATION HERE #

        ############################
        if state not in self.q_table.index:
            series_state = pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(series_state)