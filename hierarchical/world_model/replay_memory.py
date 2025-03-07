import numpy as np
import random
from collections import namedtuple,deque

# Define a named tuple for storing transitions
Transition = namedtuple('Transition', ('state', 'action', 'reward', "is_first", 'done'))

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
        self.sequence = deque(maxlen=sequence_length)


    def clear_sequence(self):
        self.sequence.clear()

    def push(self, state, action, reward, is_first,done):
        """
        Store a transition in memory.
        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            is_first (bool): Whether the state is the first in a sequence.
            done (bool): Whether the episode terminated.
        """
        is_first = np.array(is_first, dtype=np.bool_)
        done = np.array(done, dtype=np.bool_)
        self.sequence.append(Transition(state, action, reward, is_first, done))
        
        if len(self.sequence) == self.sequence_length:
            self.memory.append(list(self.sequence))
       
    def sample(self, batch_size):
        """
        Sample a batch of sequences from memory.
        Args:
            batch_size (int): Number of sequences to sample.
        Returns:
            list of list of Transition: A batch of sampled sequences.
        """
        if len(self.memory) < batch_size:
            return []

        sequences = []
        for _ in range(batch_size):
            index = random.randint(0, len(self.memory)-1)
            sequence = list(self.memory)[index]
            sequences.append(sequence)

        return sequences

    def __len__(self):
        return len(self.memory)
