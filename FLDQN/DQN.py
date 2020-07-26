import copy

import torch
import numpy as np
import torch.nn as nn
from FLDQN.Net import Net

# 设定 memory size 和 action space
MEMORY_SIZE = 45
# FEATURE = 540
N_FEATURES = 61
# 动作空间为 [0, 1, 2, 3, 4, 5]，实际动作为 [20, 21, 22, 23, 24, 25] 加 20
N_ACTIONS = 6
BATCH_SIZE = 20
# learning rate
LR = 0.01
# greedy policy
EPSILON = 0.9
# reward discount
GAMMA = 0.9
# target update frequenc
TARGET_REPLACE_ITER = 20

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_SIZE, N_FEATURES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.cost_his = []

#动作选择
    def choose_action(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value,1)[1].data.numpy()[0,0]  #return the argmax
        else: #随机选动作
            action = np.random.randint(0,N_ACTIONS)
        return action


    def store_transition(self,s,a,r,s_):
        transition = np.hastack((s,[a,r],s_))
        #如果记忆库满了，就覆盖老数据
        index = self.memory_counter%MEMORY_SIZE
        self.memory[index,:]=transition
        self.memory_counter += 1



    def learn(self,memory):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_SIZE, BATCH_SIZE)
        b_memory = memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :N_FEATURES])
        b_a = torch.LongTensor(b_memory[:, N_FEATURES:N_FEATURES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_FEATURES+1:N_FEATURES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_FEATURES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.cost_his.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self.cost_his[-1]





