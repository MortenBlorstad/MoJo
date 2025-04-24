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
        self.state_values = []

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
        self.rewards =      []
        

    def reward(self, r):
        self.rewards.append(r)

    def rlength(self):
        return len(self.rewards)
    
    #Just for debugging
    def tmplength(self):
        rl = len(self.rewards)
        sl = len(self.states)
        return "Rewardlength = " + str(rl) + " " + "State length = " + str(sl)

#Extra buffers for the manager implemetation
class MgrRolloutBuffer(BufferBase):
    def __init__(self):        
        super().__init__()        
        
        self.extrinsic_rewards =   []
        self.exploration_rewards = []

    def reward(self, r_extr, r_expl):
        self.extrinsic_rewards.append(r_extr)
        self.exploration_rewards.append(r_expl)

#Code from paper implementation found at https://github.com/danijar/director
#Edited to remove tensorflow dependency
class Normalize:

  def __init__(
      self, impl='mean_std', decay=0.99, max=1e8, vareps=0.0, stdeps=0.0):
    self._impl = impl
    self._decay = decay
    self._max = max
    self._stdeps = stdeps
    self._vareps = vareps
    self._mean = torch.tensor(0.0, dtype=torch.float64)
    self._sqrs = torch.tensor(0.0, dtype=torch.float64)
    self._step = torch.tensor(0, dtype=torch.int64)

  def __call__(self, values, update=True):
    update and self.update(values)
    return self.transform(values)

  def update(self, values):
    x = values.to(torch.float64)
    m = self._decay
    self._step += 1
    self._mean = m * self._mean + (1 - m) * x.mean()
    self._sqrs = m * self._sqrs + (1 - m) * (x ** 2).mean()

  def transform(self, values):
    correction = 1 - self._decay ** self._step
    mean = self._mean / correction
    var = (self._sqrs / correction) - mean ** 2
    if self._max > 0.0:
      scale = torch.rsqrt(         
          torch.max(var, torch.tensor(1 / self._max ** 2 + self._vareps) + self._stdeps))
    else:
      scale = torch.rsqrt(var + self._vareps) + self._stdeps
    if self._impl == 'off':
      pass
    elif self._impl == 'mean_std':
      values -= mean
      values *= scale
    elif self._impl == 'std':
      values *= scale
    else:
      raise NotImplementedError(self._impl)
    return values