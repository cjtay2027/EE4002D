# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:02:58 2021

@author: tay chao jie
"""

import numpy as np
import random
import torch
from torch import nn
import statistics
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import sys
from tensorboardX import SummaryWriter

#Global Variables
sensor_obs = '[0 0 0 0]'
r = 0
maintenance_cost = 500
machineID = ''
curr_state = 0

######################################Actor Model Skeleton###################################
class TRPO(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(TRPO, self).__init__()
        self.t = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = torch.relu(self.fc_1(input))
        x = torch.relu(self.fc_2(x))
        policy = F.softmax(self.fc_3(x),dim=-1)

        return Categorical(policy)


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

    #if 'curr_state' not in message.topic:
     #   print('Message topic: ' + message.topic)
      #  print('Message received: ' + message_payload)
       # print('\n')
    
    global r, sensor_obs, machineID,curr_state
    
    
    if message_topic == f'{machineID} reward_value':
        r = message_payload
        
    if message_topic == f'{machineID} sensor_obs_values':
        sensor_obs = message_payload

    if message_topic == f'{machineID} curr_state':
        curr_state = message_payload

########################Utility Func####################################
def update_tensorboard(writer,timestep,value):
    writer.add_scalar('Cumulative_reward',value,timestep)
    return

def run_agent(client,model,machine,writer): 
    #client.loop_start()
    rewards = []
    actions = []
    total_rewards = []
    timestep = 0
    
    while(True):
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
            
            update_tensorboard(writer,timestep,total_rewards[-1])
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
        
        obs = torch.FloatTensor(obs)

        #Model output action
        action = model(obs).sample().tolist()

        if obs.tolist() == [-1,-1,-1,-1]: #default action 0 in maintenance state
            action = 0
            
        elif obs.tolist() == [0,0,0,0]:
            timestep-=1
            action = 0
            print(f"Timestep {timestep}:Physical Layer not started")
            print()
            time.sleep(4)
            
            
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
        
            
        total_rewards.append(sum(rewards))
        update_tensorboard(writer,timestep,total_rewards[-1])

        print(f"Cumulative Reward for timestep: {timestep} = {total_rewards[-1]}")
        print()
        
        time.sleep(4)
    
    return (total_rewards,batch_actions,state_seq)




def main(argv):
    global machineID
    machineID = argv[0]  #first agument is ID
    ########################MQTT initialisation###############################
    name = f'Charles_Model_{machineID}'
    client = mqtt.Client(name)
    #broker = "mqtt.eclipseprojects.io"
    broker = "test.mosquitto.org"
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.connect(broker)

    ############################Upload Pretrained Model##########################
    print("Uploading TRPO model")
    model = torch.load('TRPO_agent_v2.pt')
    model.eval()  #set agent to evaluation mode
    print(f"Starting agent for {machineID} using broker:{broker}")
    print()


    ##############################Initiate Tensorboard Writer##############
    writer = SummaryWriter(f'logs/TRPO/{machineID}')
    #writer = SummaryWriter(f'logs/baseline')

    #################Run agent###################################
    run_agent(client,model,machineID,writer)


if __name__ == "__main__":
   main(sys.argv[1:])



