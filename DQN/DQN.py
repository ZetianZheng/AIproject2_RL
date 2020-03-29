
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper Parameters
BATCH_SIZE = 64
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 4000

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES,20)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(20, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# Deep Q Network off-policy
class DeepQNetwork(object):
    def __init__(self, N_STATES, N_ACTIONS, load = False):
        if load:
            self.load_net()
        else:
            self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, 2 * N_STATES+2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def state_to_array(self,state):
        a = []
        for i in range(len(state)):
            if state[i] == 'O':
                a.append(1)
            elif state[i] == 'X':
                a.append(-1)
            else:
                a.append(0)
        return a

    def choose_action(self, x):
        x = self.state_to_array(x)
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
            if type(action) == list:
                action = action[0]
        return action

    def store_transition(self, s, a, r, s_):
        s = self.state_to_array(s)
        s_ = self.state_to_array(s_)
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_net(self):
        self.eval_net = torch.load('eval_net.pkl')
        self.target_net = torch.load('target_net.pkl')

    def save_net(self):
        torch.save(self.eval_net, 'eval_net.pkl')
        torch.save(self.target_net, 'target_net.pkl')