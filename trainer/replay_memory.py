import numpy as np
import random
from collections import namedtuple

# Define a named tuple for storing transitions
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        """
        Initialize the replay memory.
        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in memory.
        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode terminated.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # Expand the memory until full
        self.memory[self.position] = Transition(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size):
        """
        Sample a random batch of transitions from memory.
        Args:
            batch_size (int): Number of transitions to sample.
        Returns:
            list of Transition: A batch of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
