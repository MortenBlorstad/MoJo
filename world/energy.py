from world.base_component import base_component
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple,List
from jax import jit


@jit
def round_down_to_nearest_100(step:int)->int:
    return (step // 100) * 100

@jit
def get_symmetric_coordinates(indices:jnp.array, nrows = 24, ncols = 24):
    """
    Given coordinates (i, j), returns the symmetric coordinates (j, i)
    along the main diagonal of a square grid.
    
    Args:
        indices (jnp.ndarray): A tuple or array of row and column indices.
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
    
    Returns:
        jnp.ndarray: Array with swapped and transformed (j, i) coordinates.
    """
    i, j = indices
    return ncols-j-1, nrows-i-1  # Swap i and j

class Nebula(base_component):
    def __init__(self, horizon:int):
        self.horizon = horizon
        self.nebula_tile_drift_speed:float = 0.05
        energy_node_drift_speed=[0.01, 0.02, 0.03, 0.04, 0.05]
        self.change_steps:list = [int(1/e) for e in energy_node_drift_speed]
        self.change_steps_set:set = set(self.change_steps)
        self.previous_observed_change:int = 0
        self.change_rate:int = 0
        self.prev_step:int = 0

        self.found_unique:bool = False
        self.found_unique_value:float = 0.0

        self.map = jnp.full((24,24),jnp.nan)


    def should_check(self,step:int)->bool:
        return (step - self.round_down_to_nearest_100(step)) in self.change_steps_set
    
    def get_found_unique(self, value:float)->Tuple[bool,float]:
        value = jnp.where(value<=100, value,  value - round_down_to_nearest_100(value)).item()
        if not self.should_check(value):
            return False, 0.0
        if value % 20 == 0 and value!=100:
            return True, 0.05
        elif value % 25 == 0 and (value!=100 and value!=50):
            return True, 0.04
        elif value ==34 or value == 67 != 0:
            return True, 0.03
        return False, 0.0
  
    def _set_symetry(self):
        ones_or_zeroes = jnp.where((self.map ==1 ) | (self.map ==0))
        symetry_ones_or_zeroes = get_symmetric_coordinates(ones_or_zeroes)
        self.map = self.map.at[symetry_ones_or_zeroes].set(self.map[ones_or_zeroes])


    def learn(self,current_step:int, observation: jnp.ndarray,observable:jnp.ndarray):

        if not self.should_check(current_step) or current_step ==0:
            self.prev_observation = observation
            self.prev_step = current_step
            self.map = jnp.where(observable==1,observation, self.map)
            self._set_symetry()
            return False
        
        observation_masked = jnp.where(self.prev_observable==0,0,observation)
        prev_observation_masked = self.prev_observation.at[observable==0].set(0)   


        return super().learn()
    

