# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:02:58 2021

@author: tay chao jie
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
from tensorboardX import SummaryWriter
from torch.distributions import Categorical




#Global Variables
sensor_obs = '[0 0 0 0]'
r = 0
maintenance_cost = 500
machineID = ''
curr_state = 0


##############################Model########################################
def layering_32_32(pos):
    layer_1 = np.reshape(pos[0:160],(5,32))
    layer_2 = np.reshape(pos[160:1216], (33,32))
    layer_3 = np.reshape(pos[1216:], (33,2))
    
    return [layer_1,layer_2,layer_3]

def relu(X):
    return np.maximum(0,X)

class PSO_policy(object):

    def __init__(self):
        self.bestGlobalPos = []

    def get_action(self, state):
        layers = layering_32_32(self.bestGlobalPos)
        z = state
        for index in range(len(layers)):
            z = np.append(z, 1) #need add bias
            z = z.dot(layers[index])
            if index != (len(layers)-1):  #do not include relu in the final output layer
                z = relu(z)
            
        exp = np.exp(z)
        prob = exp/np.sum(exp)

        return np.random.choice(2,p=prob)

####################MQTT functions##############################################################
import paho.mqtt.client as mqtt
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print('Connected OK')
    else:
        print('Bad connection Returned code = ' + str(rc))

def on_disconnect(client, userdata, flags, rc = 0):
    print('Disconnected result code: ' + str(rc))

def on_message(client, userdata, message):  
    message_topic = message.topic
    message_payload = message.payload.decode('utf-8')
    #print('Message topic: ' + message.topic)
    #print('Message received: ' + message_payload)
    #print('\n')
    
    global r, sensor_obs, machineID, curr_state
    
    
    if message_topic == f'{machineID} reward_value':
        r = message_payload
        
    if message_topic == f'{machineID} sensor_obs_values':
        sensor_obs = message_payload

    if message_topic == f'{machineID} curr_state':
        curr_state = message_payload

########################Utility Func####################################
def update_tensorboard(writer,timestep,value):
    writer.add_scalar('Cumulative_reward',float(value),timestep)
    return

def run_agent(client,model,machine,final_time,writer=None): 
    #initiate global variables
    #client.loop_start()
    global r
    rewards = []
    actions = []
    total_rewards = []
    timestep = 0
    
    while(timestep <= final_time):
        client.loop_start()
        timestep+=1  #increment timestep

        #Receive State/Flag about machine at state 8,9
        topic_fail = f'{machineID} curr_state'
        client.subscribe(topic_fail)
        client.on_message = on_message

        ###If in failure default is maintenance####
        if int(curr_state) in [8,9]:
            topic_action = f'{machine} tm choice'
            payload_action = 'a'+ str(1) #at failure default send maintenance
            client.publish(topic_action, payload_action)

            #Receive reward given action using mqtt
            topic_r = f'{machine} reward_value'
            client.subscribe(topic_r)
            client.on_message = on_message

            rewards.append(int(r))
            total_rewards.append(sum(rewards))
            r = 0
            
            #update_tensorboard(writer,timestep,total_rewards[-1])
            print(f"Timestep {timestep}:Failure")
            print(f"Cumulative Reward for timestep: {timestep} = {total_rewards[-1]}")
            print()
            time.sleep(4)
            continue #carry on loop
        
        
        #Receive sensor reading from mqtt
        topic_sensor_obs = f'{machine} sensor_obs_values'

        client.subscribe(topic_sensor_obs)
        client.on_message = on_message  #update variable
        
        obs = sensor_obs[1:-1].split(" ")
        #print("OBS: ",obs)
        obs = list(filter(None,obs))
        obs = list(map(lambda x: float(x),obs))
        #print(f"{machine} sensor_obs",obs)
        
        obs = np.array(obs)
        obs = np.reshape(obs,(1,4))
        #obs = torch.Tensor(obs)
        
        #Model output action
        action = model.get_action(obs)
        
        if obs.tolist() == [[-1,-1,-1,-1]]: #default action 0 in maintenance state
            print("Maintenance")
            action = 0
            
        elif obs.tolist() == [0,0,0,0] or obs.tolist() == [[0,0,0,0]]:
            timestep-=1
            action = 0
            print(f"Timestep {timestep}:Physical Layer not started")
            print()
            time.sleep(4)
            continue
            
        print(f"{machine} Sensor: {obs}, Action: {action}, state: {curr_state}")

        #Send action to mqtt broker
        topic_action = f'{machine} tm choice'
        payload_action = 'a'+ str(action)
        client.publish(topic_action, payload_action)
        

        #Receive reward given action using mqtt
        topic_r = f'{machine} reward_value'
        client.subscribe(topic_r)
        client.on_message = on_message

        
        rewards.append(int(r))
        actions.append(action)
        r = 0
            
        total_rewards.append(sum(rewards))

        #update_tensorboard(writer,timestep,total_rewards[-1])
        print(f"Cumulative Reward for timestep: {timestep} = {total_rewards[-1]}")
        print()
        
        time.sleep(4)

    np.savetxt('records/PSO/PSO_cumulative_reward_3.txt',np.array(total_rewards), delimiter=',',fmt='%s')
    
    return (total_rewards)




def main(argv):
    global machineID
    machineID = str(argv[0])  #first agument is ID
    final_time = 3000
    ########################MQTT initialisation###############################
    name = f'Charles_Model_{machineID}'
    client = mqtt.Client(name)
    #broker = "mqtt.eclipseprojects.io"
    #broker = "test.mosquitto.org"
    #broker = "broker-cn.emqx.io"
    broker = "127.0.0.1"
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.connect(broker)

    ############################Upload Pretrained Model##########################
    print("Uploading trained model")
    model = PSO_policy()
    model.bestGlobalPos = np.loadtxt('32_32_weights.txt',delimiter = ',')
    print(f"Starting agent for {machineID} using broker: {broker}")
    print()

    ##############################Initiate Tensorboard Writer##############
    #writer = SummaryWriter(f'logs/DQN/{machineID}')
    writer = None
    
    #################Run agent###################################
    results = run_agent(client,model,machineID,final_time,writer)

    return


if __name__ == "__main__":  #Eg. Machine ID: F1M1 (factory 1 machine 1)
   main(sys.argv[1:])


