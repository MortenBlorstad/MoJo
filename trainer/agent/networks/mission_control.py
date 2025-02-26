"""
This is the mission control module for the agent, responsible for
managing agent decision-making processes and task execution.

This will be inpsired by the Director paper.
"""

import jax.numpy as jnp


class MissionControl():
    def __init__(self):
        pass
    
    def act(self, state:jnp.ndarray, idx) -> jnp.ndarray:
        """
        Given a state, returns a mission m/g of shape (24,24).
        """
        mission = jnp.ones((24, 24), dtype=int)

        return mission
    
        
