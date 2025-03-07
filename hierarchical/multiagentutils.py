import torch
import torch.nn as nn
import numpy as np

class Critic(nn.Module):
    def __init__(self,state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class BufferBase():
    def __init__(self):
        super(BufferBase, self).__init__()

        #Common buffer for all implementations
        self.actions =      []
        self.states =       []
        self.logprobs =     []
        self.is_terminals = []

    def length(self):
        return len(self.actions)
    
    #Clear base & inherited buffers
    def clear(self):
        for (k, v) in self.__dict__.items():
            if isinstance(v, list):
                del v[:]

    #Print all buffers - for debugging
    def p(self):
        print("Dumping buffer:")

        for (k, v) in self.__dict__.items():
            if isinstance(v, list):
                self.plist(k,v)  

    #Print buffer values - for debugging
    def plist(self, propertyName, propertyValue):
        print(" ",propertyName)
        for element in propertyValue:
            print(element)

#Extra buffers for the worker implemetation
class WrkrRolloutBuffer(BufferBase):
    def __init__(self):        
        super().__init__()
        self.newval = 24
        self.rewards =      []
        self.state_values = []

    def reward(self, r):
        self.rewards.append(r)

    def rlength(self):
        return len(self.rewards)


#Extra buffers for the (double critic) manager implemetation
class MgrRolloutBuffer(BufferBase):
    def __init__(self):        
        super().__init__()        
        
        self.extrinsic_rewards =   []
        self.exploration_rewards = []
        self.state_values_extr =   []
        self.state_values_expl =   []

    def reward(self, r_extr, r_expl):
        self.extrinsic_rewards.append(r_extr)
        self.exploration_rewards.append(r_expl)


class EMSDNormalizer:

    def __init__(self, alpha=0.001, epsilon=1e-7):
        self.alpha = alpha      # Smoothing factor
        self.epsilon = epsilon  # Stability constant
        self.mu = 0             # Exponential moving mean
        self.v = 0              # Exponential moving variance

    def normalize(self, values):
        normalized_values = []
        for v in values:
            self.mu = self.alpha * v + (1 - self.alpha) * self.mu  # Update mean
            self.v = self.alpha * (v - self.mu) ** 2 + (1 - self.alpha) * self.v  # Update variance
            sigma = np.sqrt(self.v + self.epsilon)  # Compute std
            normalized_values.append((v - self.mu) / sigma)  # Normalize
        return torch.tensor(normalized_values, dtype=torch.float32)