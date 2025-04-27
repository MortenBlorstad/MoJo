import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Union
from collections import deque

class Critic(nn.Module):
    """
    Critic network for value estimation in reinforcement learning.
    Implements a simple feedforward neural network with tanh activations.
    
    Attributes:
        net (nn.Sequential): Neural network layers
    """

    def __init__(self, state_dim: int) -> None:
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): Dimension of the input state
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Estimated value
        """
        return self.net(x)

class BufferBase():
    """
    Base class for rollout buffers.
    Provides common functionality for storing and managing experience data.
    
    Attributes:
        actions (List[torch.Tensor]): List of actions taken
        states (List[torch.Tensor]): List of states observed
        logprobs (List[torch.Tensor]): List of action log probabilities
        is_terminals (List[bool]): List of terminal state flags
        state_values (List[torch.Tensor]): List of state values
    """

    def __init__(self) -> None:
        """Initialize the base buffer."""
        super(BufferBase, self).__init__()

        #Common buffer for all implementations
        self.actions =      []
        self.states =       []
        self.logprobs =     []
        self.is_terminals = []
        self.state_values = []

    def length(self) -> int:
        """
        Get the current length of the buffer.
        
        Returns:
            int: Number of stored experiences
        """
        return len(self.actions)
    
    #Clear base & inherited buffers
    def clear(self) -> None:
        """
        Clear all stored experiences from the buffer.
        """
        for (k, v) in self.__dict__.items():
            if isinstance(v, list):
                del v[:]

    #Print all buffers - for debugging
    def p(self) -> None:
        """
        Print the contents of the buffer for debugging.
        """
        print("Dumping buffer:")

        for (k, v) in self.__dict__.items():
            if isinstance(v, list):
                self.plist(k,v)  

    #Print buffer values - for debugging
    def plist(self, propertyName: str, propertyValue: List[Any]) -> None:
        """
        Print the contents of a specific buffer list.
        
        Args:
            propertyName (str): Name of the buffer list
            propertyValue (List[Any]): List of values to print
        """
        print(" ",propertyName)
        for element in propertyValue:
            print(element)

#Extra buffers for the worker implemetation
class WrkrRolloutBuffer(BufferBase):
    """
    Worker-specific rollout buffer.
    Extends BufferBase with worker-specific reward storage.
    
    Attributes:
        rewards (List[float]): List of rewards received
    """

    def __init__(self) -> None:        
        """Initialize the worker buffer."""       
        super().__init__()        
        self.rewards =      []
        

    def reward(self, r: float) -> None:
        """
        Add a reward to the buffer.
        
        Args:
            r (float): Reward value to add
        """
        self.rewards.append(r)

    def rlength(self) -> int:
        """
        Get the number of stored rewards.
        
        Returns:
            int: Number of stored rewards
        """
        return len(self.rewards)
    
    #Just for debugging
    def tmplength(self):
        rl = len(self.rewards)
        sl = len(self.states)
        return "Rewardlength = " + str(rl) + " " + "State length = " + str(sl)

#Extra buffers for the manager implemetation
class MgrRolloutBuffer(BufferBase):
    """
    Manager-specific rollout buffer.
    Extends BufferBase with manager-specific reward storage.
    
    Attributes:
        extrinsic_rewards (List[float]): List of extrinsic rewards
        exploration_rewards (List[float]): List of exploration rewards
    """

    def __init__(self) -> None:        
        """Initialize the manager buffer."""      
        super().__init__()        
        
        self.extrinsic_rewards =   []
        self.exploration_rewards = []

    def reward(self, r_extr: float, r_expl: float) -> None:
        """
        Add extrinsic and exploration rewards to the buffer.
        
        Args:
            r_extr (float): Extrinsic reward value
            r_expl (float): Exploration reward value
        """
        self.extrinsic_rewards.append(r_extr)
        self.exploration_rewards.append(r_expl)

#Code from papers implementation found at https://github.com/danijar/director
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