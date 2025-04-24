from universe.base_component import base_component
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from typing import Tuple,List
from jax import jit


energy_node_drift_speed=jnp.array([0.01, 0.02, 0.03, 0.04, 0.05])
change_steps = jnp.array([20, 25, 34, 40, 50, 60, 67, 75, 80, 100]) + 2
#change_steps_set:set = set(change_steps)

@jit
def closest_change_rate(change_rate: float) -> float:
    """
    Selects the closest value from the list of possible change rates to the given change_rate.
    
    Args:
        change_rate (float): The estimated change rate.
    
    Returns:
        float: The closest change rate from the list.
    """
    diffs = jnp.abs(energy_node_drift_speed - change_rate)
    index = jnp.argmin(diffs)
    return energy_node_drift_speed[index]


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


@jit
def get_cardinal_units(grid):
    """
    Takes a 2D grid with units (1s) and adds a 1 in the cardinal directions
    (left, right, up, down) of each unit using JAX and jnp.where.

    Args:
        grid (jnp.ndarray): A 2D JAX array containing 1s for unit positions and 0s otherwise.

    Returns:
        jnp.ndarray: Updated grid with additional 1s in cardinal directions.
    """
    nrows, ncols = grid.shape

    # Create shifted versions of the grid
    left = jnp.roll(grid, shift=-1, axis=1)
    right = jnp.roll(grid, shift=1, axis=1)
    up = jnp.roll(grid, shift=-1, axis=0)
    down = jnp.roll(grid, shift=1, axis=0)

    # Mask edges where rolling introduced incorrect values
    left = jnp.where(jnp.arange(ncols) == ncols - 1, 0, left)
    right = jnp.where(jnp.arange(ncols) == 0, 0, right)
    up = jnp.where(jnp.arange(nrows) == nrows - 1, 0, up)
    down = jnp.where(jnp.arange(nrows) == 0, 0, down)

    # Combine the grids: Original + all shifts
    expanded_grid = jnp.where(left + right + up + down > 0, 1, 0)

    return expanded_grid

@jit
def should_check(step:int)->bool:
    return jnp.any((step - round_down_to_nearest_100(step)) == change_steps)

@jit
def get_found_unique(value: float) -> Tuple[bool, float]:
        value = jnp.where(value <= 100, value, value - round_down_to_nearest_100(value))
        
        check_result  = should_check(value)
        value-=2
        return lax.cond(
            check_result ,
            lambda: lax.cond(
                (value % 20 == 0) & (value != 100),
                lambda: (True, 0.05),
                lambda: lax.cond(
                    (value % 25 == 0) & (value != 100) & (value != 50),
                    lambda: (True, 0.04),
                    lambda: lax.cond(
                        (value == 34) | ((value == 67) & (value != 0)),
                        lambda: (True, 0.03),
                        lambda: (False, 0.0)
                    )
                )
            ),
            lambda: (False, 0.0)
        )


@jit
def update_change_rate(current_step: int, found_unique, found_unique_value, change_rate, previous_observed_change:jnp.ndarray) -> Tuple[bool,float,float]:
    found_unique, found_unique_value = jax.lax.cond(
        found_unique,
        lambda: (found_unique, found_unique_value),
        lambda: get_found_unique(current_step)
    )
    
    new_rate = jnp.where(found_unique, found_unique_value, 1 / (current_step - previous_observed_change))
    change_rate = jnp.where(
        previous_observed_change == 2,
        new_rate,
        jnp.round(0.7 * change_rate + 0.3 * new_rate, 4)
    )
    return found_unique, found_unique_value, change_rate


@jit
def reset_map(value:int, drift:int,map:jnp.ndarray)->jnp.ndarray:
    value = jnp.where(value <= 100, value, value - round_down_to_nearest_100(value))
    value= value-2
    should_reset = (value - 1) * abs(0.03) % 1 > value * abs(0.03) % 1
    return lax.cond(should_reset, lambda: jnp.full((24,24),jnp.nan),lambda: map) 
    




class Energy(base_component):
    def __init__(self, horizon:int):
        self.horizon = horizon
        self.energy_node_drift_speed:float = 0
        
        
        self.previous_observed_change:int = 2
     
        self.change_rate:int = 0
        self.prev_step:int = 0

        self.found_unique:bool = False
        self.found_unique_value:float = 0.0

        self.map = jnp.full((24,24),jnp.nan)


    
    
    # def get_found_unique(self, value:float)->Tuple[bool,float]:
    #     value = jnp.where(value<=100, value,  value - round_down_to_nearest_100(value)).item()
    #     if not self.should_check(value):
    #         return False, 0.0
    #     if value % 20 == 0 and value!=100:
    #         return True, 0.05
    #     elif value % 25 == 0 and (value!=100 and value!=50):
    #         return True, 0.04
    #     elif value ==34 or value == 67 != 0:
    #         return True, 0.03
    #     return False, 0.0
    

    def _set_symetry(self):
        ones_or_zeroes = jnp.where(~jnp.isnan(self.map))
        symetry_ones_or_zeroes = get_symmetric_coordinates(ones_or_zeroes)
        self.map = self.map.at[symetry_ones_or_zeroes].set(self.map[ones_or_zeroes])


    def learn(self,current_step:int, observation: jnp.ndarray,pos1:jnp.ndarray,pos2:jnp.ndarray,observable:jnp.ndarray):
        pos1_cardinals = get_cardinal_units(pos1)
        pos2_cardinals = get_cardinal_units(pos2)
        if current_step>1:
            effected_mask = (pos1_cardinals>0 )| (self.prev_pos1_cardinals>0) | (pos2_cardinals>0 )| (self.pos2_cardinals>0)
        else:
            effected_mask = (pos1_cardinals>0 )| (pos2_cardinals>0 )
        observation = jnp.where(observable==0, 0, observation)
        if not should_check(current_step) or current_step ==0:
            self.map = reset_map(current_step,self.energy_node_drift_speed, self.map)
            self.prev_observation = observation 
            self.prev_pos1_cardinals = pos1_cardinals
            self.pos2_cardinals = pos2_cardinals
            self.prev_observable = observable
            self.prev_step = current_step
            self.map = jnp.where((observable==1) & ~effected_mask,observation, self.map)
            self._set_symetry()
            return False
        
        observation_masked = jnp.where((self.prev_observable==0| (observation==0)), 0, observation) # only look where energy tile was visable in both timestep.
        prev_observation_masked = jnp.where((self.prev_observable==0) | (observation==0), 0, self.prev_observation) # only look where energy tile was visable in both timestep.
     
        delta = jnp.where(effected_mask,
                          0, observation_masked - prev_observation_masked) # only compare cell not affected by units. 
        changed_detected = jnp.any((delta!=0 )) 
        
        if not changed_detected:
            self.prev_observation = observation 
            self.prev_pos1_cardinals = pos1_cardinals
            self.prev_observable = observable
            self.prev_step = current_step
            self.map = jnp.where((observable==1) & ~effected_mask,observation, self.map)
            self._set_symetry()
            return False
        # print(jnp.where(jnp.isnan(self.map.T), 0,self.map.T))
        # print(delta.T)
        self.found_unique,self.found_unique_value,self.change_rate =  update_change_rate(current_step, self.found_unique,
                                                                                        self.found_unique_value, self.change_rate,
                                                                                        self.previous_observed_change)  
        
        self.energy_node_drift_speed = closest_change_rate(self.change_rate)
        self.map = reset_map(current_step, self.energy_node_drift_speed, self.map)
        self.map = jnp.where((observable==1) & ~effected_mask, observation, self.map )  
        self._set_symetry()
        self.prev_observation = observation 
        self.prev_pos1_cardinals = pos1_cardinals
        self.prev_observable = observable
        self.prev_step = current_step
        self.previous_observed_change = current_step

        return True
    
    def should_reset(self,current_step):
        if current_step ==2:
            return True
        if self.previous_observed_change<=2:
            return should_check(current_step)
        value = jnp.where(current_step <= 100, current_step, current_step - round_down_to_nearest_100(current_step))
        value= value-2
        return (value % (1/self.energy_node_drift_speed)==0)

    def predict(self,current_step):
        predictions = []
        prediction = self.map.copy()
        predictions.append(prediction)
        for i in range(1,self.horizon+1):
            if self.should_reset(current_step+i):
                prediction = jnp.full((24,24), np.nan)
            predictions.append(prediction)
        return jnp.array([p.T for p in predictions])


    

