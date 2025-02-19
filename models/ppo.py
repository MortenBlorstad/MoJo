# region imports

#  Core JAX & NumPy Imports
import jax
import jax.numpy as jnp
import numpy as np

#Flax (Neural Network Library for JAX)
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal


# Optax (Optimization Library for JAX)
import optax
#Distrax (Distribution Library for JAX)
import distrax

# Type Hinting for Cleaner Code
from typing import Sequence, NamedTuple, Any

# Gymnax (JAX-based RL Environment Library)
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
# endregion


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class ActorCritic(nn.Module):
    pass

