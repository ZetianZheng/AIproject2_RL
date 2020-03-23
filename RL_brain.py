# RL
import numpy as np
import random as rd


class QLearningTable:

    def __init__(self, size, alpha=0.01, gamma=0.9, epsilon=0.5, q_table=None, agent = True):

        if q_table is None:
            q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = q_table
        self.size = size**2
        self.agent = agent

    def get_action(self, state, value):
        try:
            for k,v in self.q_table[state].items():
                if v == value:
                    return k
        except:
            print('get_action wrong')
            print(self.q_table[state])

    def choose_action(self, state):
        self.check_state_exist(state)
        # action selection
        if np.random.uniform() <= self.epsilon:
            # exploitation, choose best action
            value = max(self.q_table[state].values())
            action = self.get_action(state, value)

        else:
            # exploration, choose random action
            action = rd.sample(self.q_table[state].keys(), 1)[0]

        return action

    def learn(self, s, a, r, s_, done):  # state, action, reward, next_state
        # update
        if not done:
            q_predict = self.q_table[s][a]

            self.check_state_exist(s_)
            value = max(self.q_table[s_].values())
            action = self.get_action(s_, value)
            q_target = r - self.gamma * self.q_table[s_][action]

            self.q_table[s][a] += self.alpha * (q_target - q_predict)
            self.q_table[s][a] = round(self.q_table[s][a], 2)

        else:  # next state is terminal
            self.q_table[s][a] = r

    def check_state_exist(self, state):
        if state not in self.q_table:
            # append new state to q table
            self.q_table[state] = {}
            #print(state)
            for i in range(self.size):
                if state[i] == '-':
                    self.q_table[state][i] = 0

    def output_q_table(self):
        return self.q_table