from gym.utils import seeding
import random
import json
from sklearn import mixture
import gym
from gym import spaces
import pandas as pd
import numpy as np
from machine import Machine
import numpy as np


lmd0 = 0.013364
lmd1 = 0.333442
lmdM = 1 - lmd0 - lmd1 #0.6531...
mu0 = 0.125
mu1 = 0.25
muM = 0.5
maintenance_cost = 500


#transition matrices

#transition matrix for a = 0 (no maintenance)
a0_tm = np.array([[lmdM, lmd1, 0, 0, 0, 0, 0, 0, 0, lmd0], #current state 0 to next state
                  [0, lmdM, lmd1, 0, 0, 0, 0, 0, 0, lmd0], #current state 1 to next state
                  [0, 0, lmdM, lmd1, 0, 0, 0, 0, 0, lmd0], #current state 2 to next state
                  [0, 0, 0, lmdM, 0, 0, 0, 0, lmd1, lmd0], #current state 3 to next state
                  [muM, 0, 0, 0, 1-muM, 0, 0, 0, 0, 0], #current state 4 to next state
                  [muM, 0, 0, 0, 0, 1-muM, 0, 0, 0, 0], #current state 5 to next state
                  [0, muM, 0, 0, 0, 0, 1-muM, 0, 0, 0], #current state 6 to next state
                  [0, 0, muM, 0, 0, 0, 0, 1-muM, 0, 0], #current state 7 to next state
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #current state 8 to next state
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]) #current state 9 to next state


#transition matrix for a = 1 (maintenance steps)
a1_tm = np.array([[0, 0, 0, 0, 1-lmd0, 0, 0, 0, 0, lmd0], #current state 0 to next state
                  [0, 0, 0, 0, 0, 1-lmd0, 0, 0, 0, lmd0], #current state 1 to next state
                  [0, 0, 0, 0, 0, 0, 1-lmd0, 0, 0, lmd0], #current state 2 to next state
                  [0, 0, 0, 0, 0, 0, 0, 1-lmd0, 0, lmd0], #current state 3 to next state
                  [muM, 0, 0, 0, 1-muM, 0, 0, 0, 0, 0], #current state 4 to next state
                  [muM, 0, 0, 0, 0, 1-muM, 0, 0, 0, 0], #current state 5 to next state
                  [0, muM, 0, 0, 0, 0, 1-muM, 0, 0, 0], #current state 6 to next state
                  [0, 0, muM, 0, 0, 0, 0, 1-muM, 0, 0], #current state 7 to next state
                  [mu1, 0, 0, 0, 0, 0, 0, 0, 1-mu1, 0], #current state 8 to next state
                  [mu0, 0, 0, 0, 0, 0, 0, 0, 0, 1-mu0]]) #current state 9 to next state

tm = [a0_tm,a1_tm]
reward = {0:1000,1:900,2:800,3:500,4:-500,5:-500,6:-500,7:-500,8:-3000,9:-1000}

class MachineEnv(gym.Env):
    
    def __init__(self,machine):
        self.action_space = spaces.Discrete(2) 
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (4,), dtype=np.float32)
        
        self.simulator = machine #assumes that initialise machine state as 0 alr
        self.state = machine.curr_state
        self.trans = tm
        self.steps = 0
        self.done = False
        self.reward_func = reward
    
    def sensor(self): # generate observation at state
        sensor_reading = self.simulator.readSensors()
        return sensor_reading.astype(np.float32)
    
    def reset(self):
        self.state = 0
        self.simulator.curr_state = self.state
        self.done = False
        self.steps = 0
        self.state_seq = []
        return self.sensor()
    
   
    def step(self, action):
        
        transition_mat_action = self.trans[action]
        #print(f"Transition Prob: {transition_mat_action[self.state]}")
        nxt_state = np.random.choice([i for i in range(10)],1,p=transition_mat_action[self.state])[0] #select nxt state based
        reward = self.reward_func[nxt_state] #reward for going to next state
        self.state = nxt_state #update state
        self.simulator.curr_state = self.state #update GMM state
        
        while(self.state in [4,5,6,7]): #cumulative reward during maintenance
            transition_mat_action = self.trans[0] #default action 0 in maintenance state
            nxt_state = np.random.choice([i for i in range(10)],1,p=transition_mat_action[self.state])[0] #select nxt state based on prob
            self.state = nxt_state #update state
            self.simulator.curr_state = self.state #update GMM state
            reward+=self.reward_func[nxt_state]
            
        
        self.steps += 1
        
#         if(self.steps == 50):#condition for end of episode
#             self.done = True
        if self.state in [8,9] or self.steps >= 100:
            self.done = True
        
        return self.sensor(),reward,self.done,{}
    
    def render(self):
        print("\r render")
