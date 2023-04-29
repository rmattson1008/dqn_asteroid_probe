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


# just save hooks to an increasingly large list? 
# self.embeddings will be written over every period. 
#save output action? 

class DQN_H2(nn.Module):

    def __init__(self, n_observations, n_actions, in_channels=3, track_hooks=False):
        super(DQN_H2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten_conv = nn.Linear(22 * 16 * 64, 256)
        self.h_layer1 = nn.Linear(256, 256)
        self.h_layer2 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, n_actions)
        self.track_hooks = False
        self.all_hooked_output = []

        self.embeddings = OrderedDict()

        self.hook_h1 = self.h_layer1.register_forward_hook(self.forward_hook("h1"))
        self.hook_h2 = self.h_layer2.register_forward_hook(self.forward_hook("h2"))


    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.embeddings[layer_name] = output
        return hook

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float() / 255
        x = x.permute((0,3,1,2))
        # print(f'x shape is {x.shape} at start of forward')
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        # print(f'x shape is {x.shape}')
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        # print(f'x shape is {x.shape}')
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        # print(f'x shape is {x.shape}')
        x = x.reshape(x.size(0), -1)
        # print(f'x shape is {x.shape} after reshape')
        x = F.relu(self.flatten_conv(x)) #check that
        # print(f'x shape is {x.shape} after convolutions')
        x = F.relu(self.h_layer1(x))
        x = F.relu(self.h_layer2(x))

        self.all_hooked_output 
        logits = self.fc(x)
        self.embeddings['dqn_logits'] = logits
        if self.track_hooks:
            self.all_hooked_output
        #this should be list of dicts w key for each h layer output and the final logits
        
        return logits





class DQN_H5(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_H5, self).__init__()
        self.layer0 = nn.Linear(n_observations, 128)
        # lets define these with ordered dicts. 
        self.h_layer1 = nn.Linear(128, 128)
        self.h_layer2 = nn.Linear(128, 128)
        self.h_layer3 = nn.Linear(128, 128)
        self.h_layer4 = nn.Linear(128, 128)
        self.h_layer5 = nn.Linear(128, 128)

        self.fc = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.h_layer1(x))
        x = F.relu(self.h_layer2(x))
        x = F.relu(self.h_layer3(x))
        x = F.relu(self.h_layer4(x))
        x = F.relu(self.h_layer5(x))
        return self.fc(x)




