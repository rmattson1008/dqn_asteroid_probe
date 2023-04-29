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

class DQN_H2(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_H2, self).__init__()
        self.layer0 = nn.Linear(n_observations, 128)
        # lets define these with ordered dicts. 
        self.h_layer1 = nn.Linear(128, 128)
        self.h_layer2 = nn.Linear(128, 128)
        self.fc = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer0(x))
        x = F.relu(self.h_layer1(x))
        x = F.relu(self.h_layer2(x))
        return self.fc(x)





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




