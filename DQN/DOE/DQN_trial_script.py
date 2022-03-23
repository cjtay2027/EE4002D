import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
from collections import deque
import random
from collections import namedtuple
import numpy as np
import pandas as pd

import sys

from machine import Machine
from GymMachEnv import MachineEnv


Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, next_state, action, reward, mask):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, next_state, action, reward, mask))
        self.memory[self.position] = Transition(state, next_state, action, reward, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)

class DoubleDQNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DoubleDQNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        qvalue = self.fc3(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        pred = online_net(states).squeeze(1)
        _, action_from_online_net = online_net(next_states).squeeze(1).max(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.gather(1, action_from_online_net.unsqueeze(1)).squeeze(1)


        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]

def get_action(state, target_net, epsilon, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)


def update_target_model(online_net, target_net):
    # Target -> Net
    target_net.load_state_dict(online_net.state_dict())
    
def compute_avg_return(environment, policy, num_episodes):
    total_return = 0.0
    for _ in range(num_episodes):
        state = torch.Tensor(environment.reset())
        state = state.unsqueeze(0)
        episode_return = 0.0 
        while not environment.done:
            action = get_action(state, policy, 0.1, environment)
            next_state, reward, done, _ = environment.step(action)
            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)
            state = next_state
            episode_return += reward
        total_return += episode_return   
    avg_return = total_return / num_episodes
    return avg_return# Evaluate the agent's policy once before training.

def gen_param_array(l16_array):
    full = []
    for row in range(len(l16_array)):
        x = l16_array[row]
        arr_params = []
        
        for col in range(len(x)):
            list_params = list(params.values())[col]
            value = list_params[x[col]]
            arr_params.append(value)
        print(arr_params)
        full.append(arr_params)
    return full

def evaluate_and_test(params,trial_number,folder): #trial function that trains and stores results into array
    
    max_episodes = 20000
    
    #Train Env
    machine = Machine()
    machine.curr_state = 0
    env = MachineEnv(machine)

    #Eval Env
    machine2 = Machine()
    machine2.curr_state = 0
    env2 = MachineEnv(machine2)

    online_net = DoubleDQNet(4, 2)
    target_net = DoubleDQNet(4, 2)
    update_target_model(online_net, target_net)
    
    optimizer = optim.Adam(online_net.parameters(), lr=params[0])
    online_net.train()
    target_net.train()
    memory = Memory(10000)
    
    #start_time = datetime.now().replace(microsecond=0)

    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0
    initial_exploration = 1000
    log_interval = 10
    
    tracker = [[0,0]]
    
    for e in range(max_episodes):
        done = False

        score = 0
        state = env.reset()
        state = torch.Tensor(state)
        state = state.unsqueeze(0)
        while not done:
            steps += 1
            action = get_action(state, target_net, epsilon, env)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            #reward = reward if not done or score == 499 else -1
            action_one_hot = np.zeros(2)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

            if steps > initial_exploration:
                epsilon -= params[2]
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(64)
                loss = DoubleDQNet.train_model(online_net, target_net, optimizer, batch,params[1])

                if steps % params[3] == 0:
                    update_target_model(online_net, target_net)

        if e % log_interval == 0:
            if e == 0: continue
            eval_score = compute_avg_return(env2,online_net,20)
            tracker.append([eval_score,e])
    
    #end_time = datetime.now().replace(microsecond=0)
    #print("Total training time  : ", end_time - start_time)
    np.savetxt(f'{folder}/trial_{trial_number}.txt', tracker, delimiter=',',fmt='%s')
    torch.save(online_net,f'{folder}/trial_{trial_number}_dqn_agent_.pt')
    return np.array(tracker)

def train(params_array,folder_name):
    trial = 0
    for sets in params_array:
        print("trial ",trial)
#         if trial in [0,1,2]: 
#             trial+=1
#             continue
        evaluate_and_test(sets,trial,folder_name)
        trial+=1
    return

def main(args):
    folder_name = args[0]

    params = {
     'lr' : [1e-04,1e-03,1e-02,1e-01],
     'gamma' : [0.80,0.90,0.95,0.99],
     'epsilon_decay': [1e-05,1e-04,1e-03,1e-02],
     'update_freq' : [50,100,150,200]
    }

    config_array = np.array(pd.read_csv('L16.txt',header=None, sep='\t'))
    params_array = gen_param_array(config_array)
    train(params_array,folder_name)

    return

if __name__ == "__main__":
    '''
    Arguments:
    1. Folder name eg. run_1
    
    '''
    
    main(sys.argv[1:])
