from typing import OrderedDict
import numpy as np
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F





#TODO - need a real good hookig and saving process. be organized
#TODO - check batching process. 1 for new gameplay, 12 


# just save hooks to an increasingly large list? 
# self.activations will be written over every period. 
#save output action? 

class DQN_H2(nn.Module):

    def __init__(self, n_observations, n_actions, in_channels=3):
        super(DQN_H2, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        self.flatten_conv = nn.Linear(22 * 16 * 64, 256)
        self.h_layer1 = nn.Linear(256, 256)
        self.h_layer2 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, n_actions)

        self.all_hooked_output = []
        self.hook1 = None
        self.hook2 = None
        self.activations = OrderedDict()

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            print(f"test hooking activations for layer{layer_name}, size is {output.size()}")
            self.activations[layer_name] = output
        return hook

    def turn_on_hooks(self):
        print("Registering hooks")
        self.hook1 = self.h_layer1.register_forward_hook(self.forward_hook("h1"))
        self.hook2 = self.h_layer2.register_forward_hook(self.forward_hook("h2"))

    def get_activations(self, new_state=False):
        return self.all_hooked_output

    def forward(self, x):
        # print("db. Forward Loop")
        x = x.float() / 255
        x = x.permute((0,3,1,2))
        # print(f'x shape is {x.shape} at start of forward')
        x = self.cnn(x)
        # print(f'x shape is {x.shape}')
        x = x.reshape(x.size(0), -1)
        # print(f'x shape is {x.shape} after reshape')
        x = F.relu(self.flatten_conv(x)) #check that
        # print(f'x shape is {x.shape} after convolutions')
        x = F.relu(self.h_layer1(x))
        x = F.relu(self.h_layer2(x))

        self.all_hooked_output 
        logits = self.fc(x)
        self.activations['dqn_logits'] = logits
        if self.hook1:
            # print("appending to list of hooks")
            # print(self.activations)
            self.all_hooked_output.append(self.activations)
        #this should be list of dicts w key for each h layer output and the final logits
        
        return logits


class DQN_H5(nn.Module):

    def __init__(self, n_observations, n_actions, in_channels=3):
        super(DQN_H5, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        self.flatten_conv = nn.Linear(22 * 16 * 64, 256)
        self.h_layer1 = nn.Linear(256, 256)
        self.h_layer2 = nn.Linear(256, 256)
        self.h_layer3 = nn.Linear(256, 256)
        self.h_layer4 = nn.Linear(256, 256)
        self.h_layer5 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, n_actions)

        self.all_hooked_output = []
        self.hook1 = None
        self.hook2 = None
        self.activations = OrderedDict()

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            print(f"test hooking activations for layer{layer_name}, size is {output.size()}")
            self.activations[layer_name] = output
        return hook

    def turn_on_hooks(self, new_state=True):
        print("Registering hooks")
        self.hook1 = self.h_layer1.register_forward_hook(self.forward_hook("h1"))
        self.hook2 = self.h_layer2.register_forward_hook(self.forward_hook("h2"))

    def get_activations(self, new_state=False):
        return self.all_hooked_output

    def forward(self, x):
        # print("db. Forward Loop")
        x = x.float() / 255
        x = x.permute((0,3,1,2))
        # print(f'x shape is {x.shape} at start of forward')
        x = self.cnn(x)
        # print(f'x shape is {x.shape}')
        x = x.reshape(x.size(0), -1)
        # print(f'x shape is {x.shape} after reshape')
        x = F.relu(self.flatten_conv(x)) #check that
        # print(f'x shape is {x.shape} after convolutions')
        x = F.relu(self.h_layer1(x))
        x = F.relu(self.h_layer2(x))

        self.all_hooked_output 
        logits = self.fc(x)
        self.activations['dqn_logits'] = logits
        if self.hook1:
            print("appending to list of hooks")
            print(self.activations)
            self.all_hooked_output.append(self.activations)
        #this should be list of dicts w key for each h layer output and the final logits
        
        return logits

