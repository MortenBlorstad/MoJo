"""
This is the mission control module for the agent, responsible for
managing agent decision-making processes and task execution.

This will be inpsired by the Director paper.
"""

import jax.numpy as jnp

class MissionControl():
    def __init__(self):
        pass

    def act(self, state:jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()
        
