import torch.nn as nn
import numpy
import os

import torch.nn.functional as F
from torch.distributions import Categorical



class Actor(nn.Module):
    def __init__(self, input_size, action_size):
        super(Actor, self).__init__()
        self.state_size = input_size
        self.action_size = action_size

        self.l1 = nn.Linear(self.state_size, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 256)
        self.final_layer = nn.Linear(256, self.action_size)

    def forward(self,x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = self.final_layer(out)
        return Categorical(F.softmax(out, dim=-1))


class Critic(nn.Module):
    def __init__(self, input_size, action_size):
        super(Critic, self).__init__()
        self.state_size = input_size
        self.action_size = action_size

        self.l1 = nn.Linear(self.state_size, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 256)
        self.final_layer = nn.Linear(256, 1)

    def forward(self,x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = self.final_layer(out)
        return out






