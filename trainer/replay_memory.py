import numpy as np
import random
<<<<<<< HEAD
from collections import namedtuple,deque
=======
<<<<<<< HEAD
from collections import namedtuple
=======
from collections import namedtuple,deque
>>>>>>> 7515b2ceab11c37e9fed4d289e351ddc4a00fcd7
>>>>>>> main

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
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> main
    

class SequenceReplayMemory:
    def __init__(self, capacity, sequence_length):
        """
        Initialize the replay memory for sequences.
        Args:
            capacity (int): Maximum number of transitions to store.
            sequence_length (int): Number of consecutive transitions per sample.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)  # Circular buffer using deque
        self.sequence_length = sequence_length

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
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of sequences from memory.
        Args:
            batch_size (int): Number of sequences to sample.
        Returns:
            list of list of Transition: A batch of sampled sequences.
        """
        if len(self.memory) < self.sequence_length:
            return []

        sequences = []
        for _ in range(batch_size):
            start_index = random.randint(0, len(self.memory) - self.sequence_length)
            sequence = list(self.memory)[start_index:start_index + self.sequence_length]
            sequences.append(sequence)

        return sequences

    def __len__(self):
        return len(self.memory)
<<<<<<< HEAD
=======
>>>>>>> 7515b2ceab11c37e9fed4d289e351ddc4a00fcd7
>>>>>>> main
