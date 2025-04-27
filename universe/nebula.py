"""
This module defines the Nebula class and supporting functions for predicting and modeling the movement of dynamic obstacles (nebulae and asteroids) in the Lux AI Season 3 environment.

The Nebula class learns obstacle drift speed, predicts future positions based on observed changes, and detects obstacle entries and exits using JAX-accelerated computations.
Supporting functions handle probability propagation, symmetry adjustments, and movement pattern detection for enhanced prediction accuracy.
Includes unit tests for obstacle detection functionality when run as a standalone script.
"""

from universe.base_component import base_component
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple,List

from jax import jit
# region helper functions
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

# ===============
## for fast propegation of probability 
change_steps = jnp.array([7, 10, 14, 20, 27, 30, 34, 40, 47, 50, 54, 60, 67, 70, 74, 80, 87, 90, 94, 100])

@jit
def round_down_to_nearest_100(step:int)->int:
        return (step // 100) * 100

@jit
def should_check(step: jnp.ndarray) -> jnp.ndarray:
    """
    Checks if the given step(s) should apply a probability shift.
    
    Args:
        step (jnp.ndarray): Array of step values.

    Returns:
        jnp.ndarray: Boolean indicating if should check along the way.
    """
    step_adjusted = step - round_down_to_nearest_100(step)
    return jnp.any(change_steps[:, None] == step_adjusted,axis=0)

@jit
def move_probability(carry, i:int)->jnp.ndarray:
    """
    Propagates probability in the given state by shifting values in predefined directions.

    Args:
        carry (tuple): Contains:
            - pred (jnp.ndarray): The current probability state grid.
            - step (int): The current step number.
        i (int): The iteration index.

    Returns:
        tuple: 
            - Updated probability state.
            - Copy of the updated state for visualization.
    """
    pred,step,checks = carry
    moved_up_right = jnp.roll(pred, (-1, 1), axis=(0, 1))     # (-1,1)
    moved_down_left = jnp.roll(pred, (1, -1), axis=(0, 1))    # (1,-1)
    moved = should_check(step+i)
    
    ## this line should check if there is a true in checks array before current step. must be jit complient
    checks_masked = jnp.where(jnp.arange(checks.shape[0]) < i, checks, False)
    denom_factor = jnp.where(jnp.any(checks_masked), 2, 0)

    
    # Apply movement if the step is in the change_steps list
    moved_up_right = jnp.where(
                                moved, # Check if it's time to apply the shift
                                    moved_up_right, # Apply the shifted map if condition is met (step interval reached)
                                    pred, # Otherwise, keep the original map unchanged
                                )
    moved_down_left = jnp.where(
                                moved, # Check if it's time to apply the shift
                                    moved_down_left, # Apply the shifted map if condition is met (step interval reached)
                                    pred, # Otherwise, keep the original map unchanged
                                )
    value = jnp.where(moved, pred / (i + 2),  pred ) # if moved:1 / (i + 2): else: 1
    remove_mask = (pred>0) & (pred<jnp.inf)
     
    mask_up_right = ((0<moved_up_right) & (moved_up_right<jnp.inf)) | ((0<moved_down_left) & (moved_down_left<jnp.inf))
    pred = jnp.where(remove_mask, 1/(1+denom_factor), pred)
    pred = jnp.where(mask_up_right , 1/(1+denom_factor), pred)
    
    return (pred,step,checks), pred

@jit
def set_probs(pred,value, mask,remove_mask):
    pred = jnp.where(remove_mask, 0, pred)
    pred = jnp.where(mask, value, pred)
    return
    


@jit
def predict_probabilities(state:jnp.ndarray, step:int, iterations: jnp.ndarray):
    """
    Predicts probability propagation over multiple iterations using JAX scan.

    Args:
        state (jnp.ndarray): Initial probability grid.
        step (int): Initial step number.
        iterations (int): Number of iterations to simulate.

    Returns:
        jnp.ndarray: Array of state grids at each iteration. shape (number_of_predictions, 24, 24)
    """
    num_iterations = iterations.shape[0]
    all_states = jnp.zeros((num_iterations +1, 24, 24) )
    all_states = all_states.at[0].set(state)

    iteration_steps = iterations + step

    checks = should_check(iteration_steps)
    (final_state, _,_), all_predictions = jax.lax.scan(move_probability, (state, step, checks), iterations)
    all_states = all_states.at[1:].set(all_predictions)
    return all_states


@jit
def remove_unwanted_columns(matrix):
    valid_mask = jnp.logical_or(
        np.all(matrix == np.array([[1], [-1]]), axis=0),
        np.all(matrix == np.array([[-1], [1]]), axis=0)
    )
    valid_indices = jnp.where(valid_mask, size=matrix.shape[1], fill_value=0)[0]

    return jnp.take(matrix,valid_indices,axis =1)


# endregion


# ===============

class Nebula(base_component):
    """
        This class learns the drift speed for nebula and astroid and keeps track of their positions. Can be used to predict future positions.

        This class can be used for both nebula and astroid. 

    """

    def __init__(self, horizon:int, name:str = "nebula"):
        """
           Args:
            horizon (int): The raw observation dictionary provided by the game environment.
            name (str): identifier is it track 'nebula' or 'astroid'.

        """
        super().__init__()
        self.horizon = horizon
        self.nebula_tile_drift_speed:float = 0.0
        self.change_steps:list = [7, 10, 14, 20, 27, 30, 34, 40, 47, 50, 54, 60, 67, 70, 74, 80, 87, 90, 94, 100]
        self.change_steps_set:set = set(self.change_steps)
        self.previous_observed_change:int = 0
        self.change_rate:int = 0
        self.prev_step:int = 0
        self.map = jnp.full((24,24),jnp.nan)
       
        self.found_unique:bool = False
        self.found_unique_value:float = 0.0
        self.direction:float = 0.0
        self.name = name

    def _move_astroid_or_nebula(self, map_object: jnp.ndarray,steps:int)->jnp.ndarray:
        """
            Shift objects around in space
            Move the nebula tiles in state.map_features.tile_types up by 1 and to the right by 1
            this is also symmetric nebula tile movement
            copied from https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/501ada7575b6dc0cf7f0bcbf2558322efcef190f/src/luxai_s3/env.py#L640
        """
        new_map = jnp.roll(
        map_object,
        shift=(
            1 * jnp.sign(self.nebula_tile_drift_speed),  # Shift down if drift speed is positive, up if negative
            -1 * jnp.sign(self.nebula_tile_drift_speed), # Shift left if drift speed is positive, right if negative
        ),
        axis=(0, 1),  # Apply the shifts to both row (0) and column (1) axes
        )
        
        # new
        # Conditionally update the map based on drift speed and step count
        new_map = jnp.where(
            (steps - 1) * abs(self.nebula_tile_drift_speed) % 1 > steps * abs(self.nebula_tile_drift_speed) % 1, # Check if it's time to apply the shift
            new_map, # Apply the shifted map if condition is met (step interval reached)
            map_object, # Otherwise, keep the original map unchanged
        )
        return new_map

    def round_down_to_nearest_100(self,step:int)->int:
        return (step // 100) * 100
    
    def should_check(self,step:int)->bool:
        return (step - self.round_down_to_nearest_100(step)) in self.change_steps_set
    
    def get_found_unique(self, value:float)->Tuple[bool,float]:
        value = value - self.round_down_to_nearest_100(value)
        if value % 10 != 0:
            return True, 0.15
        elif (value // 10) % 2 == 1:
            return True, 0.1
        return False, 0.0
        

    def update_change_rate(self,current_step:int)->None:
        if not self.found_unique:
            self.found_unique, self.found_unique_value = self.get_found_unique(current_step)
        
        if self.found_unique:
            new_rate = self.found_unique_value
        else:
            
           
            new_rate = 1/(current_step - self.previous_observed_change)
        if self.previous_observed_change ==0:
            self.change_rate = new_rate
        else:
            self.change_rate = np.round(0.7*self.change_rate + 0.3*new_rate,4)

    def closest_change_rate(self,change_rate:float, possible_rates:List[float]=[0.15, 0.1, 0.05, 0.025]):
        """
        Selects the closest value from the list of possible change rates to the given change_rate.
        
        Args:
            change_rate (float): The estimated change rate.
            possible_rates (list of float): The predefined list of possible change rates.
        
        Returns:
            float: The closest change rate from the list.
        """
        return min(possible_rates, key=lambda x: abs(x - change_rate))

    def detect_obstacle_entry(self,state:jnp.ndarray, next_state:jnp.ndarray):
        """
        Detects obstacles entering the grid from:
        - Enters from bottom row or first column: (-1,1)
        - Enters from top row or last column: (1,-1)

        Args:
            state (jnp.ndarray): The current grid state (before the update).
            next_state (jnp.ndarray): The next grid state (after the update).

        Returns:
            tuple: (bool, direction) where:
                - bool: True if an obstacle is detected entering, False otherwise.
                - direction: (-1,1) or (1,-1) if an obstacle is entering, None otherwise.
        """
         # Movement simulation
        moved_up_right = jnp.roll(state, (-1, 1), axis=(0, 1))  # (-1,1)
        moved_down_left = jnp.roll(state, (1, -1), axis=(0, 1))  # (1,-1)

        # (-1,1) direction: Entry should appear at bottom row or first column
        bottom_row_entries = jnp.any((state[-1,:] ==0)  & (next_state[-1,:]==1)) and jnp.all(next_state[-2, :] == moved_up_right[-2,:] ) and jnp.all(next_state[-3,:] == moved_up_right[-3,:])  and not jnp.all(next_state[-1, :] == moved_up_right[-1,:] )  and not jnp.all(state[-1, :] == next_state[-1, :])
        first_col_entries = jnp.any((state[:, 0]==0) &  (next_state[:, 0]==1)) and jnp.all(next_state[:, 1] == moved_up_right[:,1] ) and jnp.all(next_state[:,2] == moved_up_right[:,2])  and not jnp.all(next_state[:,0] == moved_up_right[:,0])  and not jnp.all(state[:,0] == next_state[:,0])

        if bottom_row_entries or first_col_entries:
            return (True, (-1, 1))

        # (1,-1) direction: Entry should appear at top row or last column
        top_row_entries =  jnp.any((state[0, :] ==0)   & (next_state[0, :]==1))  and jnp.all(next_state[1, :] == moved_down_left[1,:] )   and jnp.all(next_state[2,:] == moved_down_left[2,:])    and not jnp.all(next_state[0,:] == moved_up_right[0,:])   and not jnp.all(state[0, :] == next_state[0, :])
        last_col_entries = jnp.any((state[:, -1] ==0)  & (next_state[:, -1]==1)) and jnp.all(next_state[:, -2] == moved_down_left[:,-2] ) and jnp.all(next_state[:,-1] == moved_down_left[:,-1])  and not jnp.all(next_state[:,-1] == moved_up_right[:,-1]) and not jnp.all(state[:, -1] == next_state[:, -1])

        if (top_row_entries or last_col_entries):
            return (True, (1, -1))

        return (False, None)
    
    def detect_obstacle_leaving(self, state:jnp.ndarray, next_state:jnp.ndarray)->Tuple[bool,Tuple[int,int]]:
        """
        Detects obstacles leaving the grid from:
        - Top row or last column if direction is (-1,1)
        - Bottom row or first column if direction is (1,-1)

        Args:
            state (jnp.ndarray): The current grid state (before the update).
            next_state (jnp.ndarray): The next grid state (after the update).

        Returns:
            tuple: (bool, direction) where:
                - bool: True if an obstacle is detected leaving, False otherwise.
                - direction: (-1,1) or (1,-1) if an obstacle is leaving, None otherwise.
        """

        # Movement simulation
        moved_up_right = jnp.roll(state, (-1, 1), axis=(0, 1))  # (-1,1)
        moved_down_left = jnp.roll(state, (1, -1), axis=(0, 1))  # (1,-1)

        # (-1,1) direction: Exit should disappear from the top row or last column
        top_row_exits = jnp.all(next_state[0,:] == moved_up_right[0,:]) and jnp.all(next_state[1,:] == moved_up_right[1,:]) and not jnp.all(next_state[1,:] == moved_down_left[1,:])  and not jnp.all(state[0,:] == next_state[0,:])
        last_col_exits = jnp.all(next_state[:,-1] == moved_up_right[:,-1]) and jnp.all(next_state[:,-2] == moved_up_right[:,-2]) and not jnp.all(next_state[:,-2] == moved_down_left[:,-2])  and not jnp.all(state[:,-1] == next_state[:,-1] )

        if top_row_exits or last_col_exits:
            return (True, (-1, 1))

        # (1,-1) direction: Exit should disappear from the bottom row or first column
        
        jnp.all(next_state[-1,:] == moved_down_left[-1,:]) # last col next states matches last col in moving moved_down_left
        jnp.all(state[-2,:] == moved_down_left[-1,:])


        bottom_row_exits = jnp.all(next_state[-1,:] == moved_down_left[-1,:]) and (jnp.all(next_state[-2,:] == moved_down_left[-2,:])) and not jnp.all(next_state[-2,:] == moved_up_right[-2,:])  and not jnp.all(state[-1,:] == next_state[-1,:])
        first_col_exits = jnp.all(next_state[:,0] == moved_down_left[:,0]) and jnp.all(next_state[:,1] == moved_down_left[:,1]) and not jnp.all(next_state[:,1] == moved_up_right[:,1]) and not jnp.all(state[:,0] == next_state[:,0])

        if bottom_row_exits or first_col_exits:
            return (True, (1, -1))

        return (False, None)
        
    
    def normalize_signs_jnp(self, arr:jnp.ndarray)->jnp.ndarray:
        """
        Replaces all values in each column with the most common sign in that column using JAX.

        Parameters:
        arr (jnp.ndarray): Input 2D array.

        Returns:
        jnp.ndarray: Array where each column is replaced with its most common sign.
        """
        arr = jnp.array(arr)  # Ensure input is a JAX array
        signs = jnp.sign(arr)  # Get signs (-1, 0, or 1)
        size = arr.shape[1]
        # Compute most common sign per column
        def most_common_sign(column):
            unique, counts = jnp.unique(column, return_counts=True, size=size)
            return unique[jnp.argmax(counts)]  # Most frequent sign

        most_common_signs = jnp.apply_along_axis(most_common_sign, axis=0, arr=signs)

        # Replace values with the most common sign of their column
        normalized_arr = jnp.where(signs != 0, most_common_signs, signs)

        return normalized_arr

    def _set_symetry(self):
        ones_or_zeroes = jnp.where((self.map ==1 ) | (self.map ==0))
        symetry_ones_or_zeroes = get_symmetric_coordinates(ones_or_zeroes)
        self.map = self.map.at[symetry_ones_or_zeroes].set(self.map[ones_or_zeroes])


    def learn(self, observation:jnp.ndarray, observable:jnp.ndarray, current_step:jnp.ndarray, prev_change_step:jnp.ndarray, dbg = False)->bool:
        """
            check whether nebula_tile_drift_speed in one of the following values {-0.5: move up-right every other step,-0.25: move up-right every 4th step,0.25: move down-left every 4th step,0.5 move down-left every second step}
            obs: (5x24,24) four time steps of nebula positions, (current_step-5, current_step-3, current_step-2, current_step-1, current_step) current time step + 4 previous
            current_step: current step number
        """
        assert observation.shape == (24,24) and observable.shape == (24,24)
        self.previous_observed_change = prev_change_step
        
        if not self.should_check(current_step) or current_step ==0:
            self.prev_observation = observation
            self.prev_observable = observable
            self.prev_step = current_step
           
            
          
            self.map = jnp.where(observable==1, observation, self.map)
           
            self._set_symetry()
            return False



        observation_masked = observation.at[self.prev_observable==0].set(0)
        prev_observation_masked = self.prev_observation.at[observable==0].set(0)    
        

        delta = observation_masked - prev_observation_masked
        delta = delta.at[observation_masked==1].set(1)
        shift_up_right = jnp.pad(delta[:-1, 1:], ((1, 0), (0, 1)), constant_values=0)  # (-1, 1)
        shift_down_left = jnp.pad(delta[1:, :-1], ((0, 1), (1, 0)), constant_values=0)  # (1, -1)
        shift_down_right = jnp.pad(delta[1:, 1:], ((0, 1), (0, 1)), constant_values=0)  # (1, 1)
        shift_up_left = jnp.pad(delta[:-1, :-1], ((1, 0), (1, 0)), constant_values=0)  # (-1, -1)
        adjacent_neg_ones = (shift_up_right == -1) | (shift_down_left == -1) | (shift_down_right == -1) | (shift_up_left == -1)

        adjacent_neg_ones = (shift_up_right == -1) | (shift_down_left == -1) | (shift_down_right == -1) | (shift_up_left == -1)
        adjacent_ones = (shift_up_right == 1) | (shift_down_left == 1) | (shift_down_right == 1) | (shift_up_left == 1)

        # Find positions where -1 exists in diagonal directions of 1
        delta = jnp.where(delta == -1, -1, jnp.where((delta == 1) & adjacent_neg_ones, 1, 0))
        #Find positions where 1 exists in diagonal directions of -1
        delta = jnp.where(delta == 1, 1, jnp.where((delta == -1) & adjacent_ones, -1, 0)) #jnp.where((delta == -1) & adjacent_ones, -1, delta)

    

        entering,enetering_direction = self.detect_obstacle_entry(prev_observation_masked,observation_masked)
        leaving, leaving_direction = self.detect_obstacle_leaving(prev_observation_masked,observation_masked)
        
        if entering or leaving:
            if entering and leaving:
                if enetering_direction != leaving_direction:
                    if dbg:
                        print(entering,leaving)
                        print(enetering_direction,leaving_direction)
                        print(prev_observation_masked)
                        print(observation_masked)
                        print(f"{enetering_direction} != {leaving_direction}")

                elif entering and not leaving:
                    self.direction = enetering_direction[0] #-1: (-1,1), 1: (1,-1)
                else:
                    self.direction = leaving_direction[0] #-1: (-1,1), 1: (1,-1)
            
            self.update_change_rate(current_step)
            
            self.nebula_tile_drift_speed = self.direction*self.closest_change_rate(self.change_rate)
            #print(current_step, self.get_found_unique(current_step))
            if dbg:
                print(f"Change detected at change step {current_step} by detect_obstacle. Nebula speed is {self.nebula_tile_drift_speed}, {entering} {leaving})")    

        if not np.any(delta == -1) and (not(entering or leaving) or enetering_direction != leaving_direction):
          
            self.prev_observation = observation
            self.prev_observable = observable
            self.prev_step = current_step
            self.map = self._move_astroid_or_nebula(self.map ,current_step)
            self.map = jnp.where(observable==1, observation, self.map)
            self._set_symetry()
            return False
        else:
            
            self.update_change_rate(current_step)
   
          
            moved_from_indices = jnp.array(jnp.where(delta==-1))
            moved_to_indices = jnp.array(jnp.where(delta==1))
            if moved_from_indices.shape != moved_to_indices.shape:
                if dbg:
                    print(prev_observation_masked[:10,:10])
                    print(observation_masked[:10,:10])

                cand1 = jnp.roll(prev_observation_masked, (1, -1), axis=(0,1))
                cand2 = jnp.roll(prev_observation_masked, (-1, 1), axis=(0,1))
                delta1 = jnp.abs(observation_masked ==cand1).sum()
                delta2 = jnp.abs(observation_masked == cand2).sum()
                directions = [1, -1]
                ind = jnp.argmax(jnp.array([delta1, delta2]))  # Find direction with minimal difference
                self.direction = directions[ind] 

            else:
                if dbg:
                    print("standard")
                    print(delta)
                    print(observation_masked[:10,10])
                    print(prev_observation_masked[:10,10])
                directions = (moved_from_indices - moved_to_indices)
                #print(self.prev_step,current_step, direction, self.change_rate)
                directions = self.handle_rollovers(directions)
                expected_direction_1 = jnp.array([[1], [-1]])  # Down-left movement
                expected_direction_2 = jnp.array([[-1], [1]])  # Up-right movement
                # Check if all movements follow the same pattern
                directions = remove_unwanted_columns(directions)
                if dbg:
                    print("standard", directions)
                if jnp.all(directions == expected_direction_1):
                    self.direction  = -1
                elif jnp.all(directions == expected_direction_2):
                    self.direction  = 1
                else:
                    if dbg:
                        print("Mixed movement or no consistent pattern")
                    #raise Exception("Mixed movement or no consistent pattern")
            
            self.nebula_tile_drift_speed = self.direction*self.closest_change_rate(self.change_rate)
           
            if dbg:
                print(f"Change detected at change step {current_step}. Nebula speed is {self.nebula_tile_drift_speed}")    
           



        self.prev_observation = observation
        self.prev_observable = observable
        self.prev_step = current_step

        self.map = self._move_astroid_or_nebula(self.map, current_step)
        

        self.map = jnp.where(observable==1, observation, self.map)

        
        self._set_symetry()

        return True


    
    def handle_rollovers(self,arr:jnp.ndarray)->jnp.ndarray:
        """
        Adjusts values in an array where abs(value) > 1 by flipping the sign of the entire column.
        
        If any value in a column exceeds ±1, all values in that column are adjusted to maintain 
        consistency by aligning them with the sign of the first affected value.

        Parameters:
        arr (jnp.ndarray): Input 2D JAX array.

        Returns:
        jnp.ndarray: Modified array with normalized values.
        """
        if arr.shape[1] == 0:
            return arr
        # Check if all values in the array are the same
        if jnp.all(arr == arr[:, 0].reshape(2,-1)):
            return arr
        # Identify indices where |value| > 1 (i.e., need to be fixed)
        arr = jnp.array(arr,dtype=jnp.float32)

        need_fix = jnp.where(jnp.abs(arr) >1)

        rows = need_fix[0] - need_fix[0]
        
        # Extract affected column indices
        cols = need_fix[1]
        next_need_fix = (rows,cols)
        sign = jnp.sign(arr[next_need_fix])

        values = arr[need_fix]*(sign)
        values= values/jnp.abs(values)

        next_values = arr[next_need_fix]*(sign)
        next_values= next_values/jnp.abs(next_values)
        
        arr = arr.at[need_fix].set(values)
        arr = arr.at[next_need_fix].set(next_values)
        return arr

    

    def predict(self,observation:jnp.ndarray, observable:jnp.ndarray,current_step:int)->List[jnp.ndarray]:
        if self.direction ==0 or self.nebula_tile_drift_speed ==0:
            iterations_array = jnp.arange(1,int(self.horizon+1))
            prediction = self.map.copy()
            predictions = predict_probabilities(prediction, current_step, iterations_array)
            return [pred for pred in predictions]
        else:
            predictions = []
            predictions.append(self.map)
            prediction = self.map.copy()
            for i in range(1,self.horizon+1):
                prediction = self._move_astroid_or_nebula(prediction, current_step+i)
                predictions.append(prediction)
            return predictions


if __name__ == "__main__":
    

    def test_detect_obstacle_entry_leaving():
        # Example 1: entering (1,-1)
        state1 = jnp.array([ 
            [ 0,  0,  1, 1],
            [ 0,  1,  1, 1],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0]
        ])

        next_state1 = jnp.array([ 
            [ 0,  0,  1, 0],
            [ 0,  1,  1, 0],
            [ 1,  1,  1, 0],
            [ 0,  0,  0, 0]
        ])

        # Example 2: leavning (1,-1)
        state2 = jnp.array([ 
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  1,  1, 0],
            [ 0,  1,  0, 0]
        ])
        next_state2 = jnp.array([ 
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 1,  1,  0, 0]
        ])

        # Example 3: leavning (1,-1)
        state3 = jnp.array([ 
            [ 0,  0,  0, 0],
            [ 1,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0]
        ])
        next_state3 = jnp.array([ 
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0]
        ])

        # Example 4: entering (-1,1)
        state4 = jnp.array([ 
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0]
        ])
        next_state4 = jnp.array([ 
            [ 0,  0,  0, 0],
            [ 1,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0]
        ])

        # Example 5: no entrering or leaving.
        state5 = jnp.array([ 
            [ 0,  1,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0]
        ],dtype=float)

        next_state5 = jnp.array([ 
            [ 0,  0,  0, 0],
            [ 1,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0]
        ],dtype=float)


        state6 = jnp.array([ 
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 1,  0,  0, 0],
            [ 0,  0,  0, 0]
        ],dtype=float)

        # Example 6: no entrering or leaving.
        next_state6 = jnp.array([ 
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  0,  0, 0],
            [ 0,  1,  0, 0]
        ],dtype=float)

        nebula = Nebula(3)
        # Run the tests
        print("Example 1")
        print("detect_obstacle_entry:", nebula.detect_obstacle_entry(state1, next_state1))
        print("detect_obstacle_leaving:", nebula.detect_obstacle_leaving(state1, next_state1))

        print("\nExample 2")
        print("detect_obstacle_entry:", nebula.detect_obstacle_entry(state2, next_state2))
        print("detect_obstacle_leaving:", nebula.detect_obstacle_leaving(state2, next_state2))

        print("\nExample 3")
        print("detect_obstacle_entry:", nebula.detect_obstacle_entry(state3, next_state3))
        print("detect_obstacle_leaving:", nebula.detect_obstacle_leaving(state3, next_state3))

        print("\nExample 4")
        print("detect_obstacle_entry:", nebula.detect_obstacle_entry(state4, next_state4))
        print("detect_obstacle_leaving:", nebula.detect_obstacle_leaving(state4, next_state4))

        print("\nExample 5")
        print("detect_obstacle_entry:", nebula.detect_obstacle_entry(state5, next_state5))
        print("detect_obstacle_leaving:", nebula.detect_obstacle_leaving(state5, next_state5))


        print("\nExample 6")
        print("detect_obstacle_entry:", nebula.detect_obstacle_entry(state6, next_state6))
        print("detect_obstacle_leaving:", nebula.detect_obstacle_leaving(state6, next_state6))

    test_detect_obstacle_entry_leaving()


 