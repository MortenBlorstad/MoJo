from world.base_component import base_component
import jax
import jax.numpy as jnp
import numpy as np


def get_symmetric_coordinates(indecies, nrows = 24, ncols = 24):
    """
    Given coordinates (i, j), returns the symmetric coordinates (j, i)
    along the main diagonal of a square grid.
    
    Args:
        i (int): Row index.
        j (int): Column index.
    
    Returns:
        (int, int): Swapped (j, i) coordinates.
    """
    i = indecies[0]
    j = indecies[1]
    return ncols-j-1, nrows-i-1  # Swap i and j



class Nebula(base_component):

    def __init__(self, horizon, name:str = "nebula"):
        super().__init__()
        self.horizon = horizon
        #self.map = jnp.zeros((1+horizon,24,24))
        self.nebula_tile_drift_speed:float = 0.0
        self.change_steps:list = [7, 10, 14, 20, 27, 30, 34, 40, 47, 50, 54, 60, 67, 70, 74, 80, 87, 90, 94, 100]
        self.change_steps_set:set = set(self.change_steps)
        self.previous_observed_change:int = 0
        self.change_rate:int = 0
        self.prev_step:int = 0
        self.map = jnp.ones((24,24))/3
        self.found_unique:bool = False
        self.found_unique_value:float = 0.0
        self.direction:float = 0.0
        self.name = name

    def _move_astroid_or_nebula(self, map_object,steps):
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
        #print(steps,(steps - 1) * abs(self.nebula_tile_drift_speed) % 1 > steps * abs(self.nebula_tile_drift_speed) % 1)
        new_map = jnp.where(
            (steps - 1) * abs(self.nebula_tile_drift_speed) % 1 > steps * abs(self.nebula_tile_drift_speed) % 1, # Check if it's time to apply the shift
            new_map, # Apply the shifted map if condition is met (step interval reached)
            map_object, # Otherwise, keep the original map unchanged
        )
        return new_map

    def round_down_to_nearest_100(self,step):
        return (step // 100) * 100
    
    def should_check(self,step):
        return (step - self.round_down_to_nearest_100(step)) in self.change_steps_set
    
    def get_found_unique(self, value):
        value = value - self.round_down_to_nearest_100(value)
        if value % 10 != 0:
            return True, 0.15
        elif (value // 10) % 2 == 1:
            return True, 0.1
        return False, 0.0
        

    def update_change_rate(self,current_step):
        if not self.found_unique:
            self.found_unique, self.found_unique_value = self.get_found_unique(current_step)
        
        if self.found_unique:
            new_rate = self.found_unique_value
        else:
            
            print("current", current_step, "previous_observed_change", self.previous_observed_change)
            new_rate = 1/(current_step - self.previous_observed_change)
        if self.previous_observed_change ==0:
            self.change_rate = new_rate
        else:
            self.change_rate = np.round(0.7*self.change_rate + 0.3*new_rate,4)

    def closest_change_rate(self,change_rate, possible_rates=[0.15, 0.1, 0.05, 0.025]):
        """
        Selects the closest value from the list of possible change rates to the given change_rate.
        
        Args:
            change_rate (float): The estimated change rate.
            possible_rates (list of float): The predefined list of possible change rates.
        
        Returns:
            float: The closest change rate from the list.
        """
        return min(possible_rates, key=lambda x: abs(x - change_rate))

    def detect_obstacle_entry(self,state, next_state):
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
    
    def detect_obstacle_leaving(self, state, next_state):
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
        
    
    def normalize_signs_jnp(self, arr):
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


    def learn(self, observation, observable, current_step, prev_change_step):
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
            self.map = self._move_astroid_or_nebula(self.map ,current_step)
            if self.nebula_tile_drift_speed !=0:
                mask = jnp.where((self.map !=1) & (observable==1))
            else: 
                mask = jnp.where((observable==1))

            # if (current_step) == 40:
            #     print(self.nebula_tile_drift_speed)
            #     np.save("MoJo/mask.npy", np.asarray(mask))
            #     np.save("MoJo/map_prev.npy", np.asarray(self.map))
            #     np.save("MoJo/observation.npy", np.asarray(observation))
            #     np.save("MoJo/observable.npy", np.asarray(observable))
            self.map = self.map.at[mask].set(observation[mask])
            # if (current_step) ==40:
            #     np.save("MoJo/map_after.npy",np.asarray(self.map))
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
                    print(entering,leaving)
                    print(enetering_direction,leaving_direction)
                    print(prev_observation_masked)
                    print(observation_masked)
                    raise Exception(f"{enetering_direction} != {leaving_direction}")

            if entering and not leaving:
                self.direction = enetering_direction[0] #-1: (-1,1), 1: (1,-1)
            else:
                self.direction = leaving_direction[0] #-1: (-1,1), 1: (1,-1)
            
            self.update_change_rate(current_step)
            
            self.nebula_tile_drift_speed = self.direction*self.closest_change_rate(self.change_rate)
            print(current_step, self.get_found_unique(current_step))
            print(f"Change detected at change step {current_step} by detect_obstacle. Nebula speed is {self.nebula_tile_drift_speed}")    

        if not np.any(delta == -1) and not (entering or leaving):
            print(f"no change detected at change step {current_step}")
            self.prev_observation = observation
            self.prev_observable = observable
            self.prev_step = current_step
            self.map = self._move_astroid_or_nebula(self.map ,current_step)
            if self.nebula_tile_drift_speed !=0:
                mask = jnp.where((self.map !=1) & (observable==1))
            else: 
                mask = jnp.where((observable==1))

            # if (current_step) == 40:
            #     print(self.nebula_tile_drift_speed)
            #     np.save("MoJo/mask.npy", np.asarray(mask))
            #     np.save("MoJo/map_prev.npy", np.asarray(self.map))
            #     np.save("MoJo/observation.npy", np.asarray(observation))
            #     np.save("MoJo/observable.npy", np.asarray(observable))
            self.map = self.map.at[mask].set(observation[mask])
            # if (current_step) ==40:
            #     np.save("MoJo/map_after.npy",np.asarray(self.map))
            self._set_symetry()
            return False
        else:
            
            self.update_change_rate(current_step)
   
            # print(current_step, delta.T)   
            # print(current_step, observation_masked.T) 
            # print(self.prev_step, self.prev_observation.T) 
            moved_from_indices = jnp.array(jnp.where(delta==-1))
            moved_to_indices = jnp.array(jnp.where(delta==1))
            if moved_from_indices.shape != moved_to_indices.shape:
                possible_directions = jnp.array([[1, -1], [-1, 1]])  # (row change, column change)
                directions = [1,-1]
                moved_from_plus_dir1 = moved_from_indices + possible_directions[0][:, None]  # Move (1,-1)
                moved_from_plus_dir2 = moved_from_indices + possible_directions[1][:, None]  # Move (-1,1)
                matches_dir1 = jnp.sum((moved_from_plus_dir1[:, :, None] == moved_to_indices[:, None, :]).all(axis=0), axis=1)
                matches_dir2 = jnp.sum((moved_from_plus_dir2[:, :, None] == moved_to_indices[:, None, :]).all(axis=0), axis=1)
                self.direction = directions[jnp.argmax(jnp.array([matches_dir1.sum(), matches_dir2.sum()]))]
            else:
                directions = (moved_from_indices - moved_to_indices)
                #print(self.prev_step,current_step, direction, self.change_rate)
                directions = self.handle_rollovers(directions)
                expected_direction_1 = jnp.array([[1], [-1]])  # Down-left movement
                expected_direction_2 = jnp.array([[-1], [1]])  # Up-right movement
                # Check if all movements follow the same pattern
              
                if jnp.all(directions == expected_direction_1):
                    self.direction  = -1
                elif jnp.all(directions == expected_direction_2):
                    self.direction  = 1
                else:
                    print("Mixed movement or no consistent pattern")
                    #raise Exception("Mixed movement or no consistent pattern")
            
            self.nebula_tile_drift_speed = self.direction*self.closest_change_rate(self.change_rate)
            #print(current_step, self.nebula_tile_drift_speed, direction*self.change_rate)
            print(f"Change detected at change step {current_step}. Nebula speed is {self.nebula_tile_drift_speed}")    
            print(self.get_found_unique(current_step))



        self.prev_observation = observation
        self.prev_observable = observable
        self.prev_step = current_step

        self.map = self._move_astroid_or_nebula(self.map, current_step)
        if self.nebula_tile_drift_speed !=0:
            mask = jnp.where((self.map !=1) & (observable==1))
        else: 
            mask = jnp.where((observable==1))


        # if (current_step) == 40:
        #     print(self.nebula_tile_drift_speed)
        #     np.save(f"MoJo/{self.name}_mask.npy", np.asarray(mask))
        #     np.save(f"MoJo/{self.name}_map_prev.npy", np.asarray(self.map))
        #     np.save(f"MoJo/{self.name}_oservation.npy", np.asarray(observation))
        #     np.save(f"MoJo/{self.name}_observable.npy", np.asarray(observable))
        self.map = self.map.at[mask].set(observation[mask])
        # if (current_step) ==40:
        #     np.save(f"MoJo/{self.name}_map_after.npy",np.asarray(self.map))
        
        self._set_symetry()

        return True


    
    def handle_rollovers(self,arr):
        """
        Adjusts values in an array where abs(value) > 1 by flipping the sign of the entire column.
        
        If any value in a column exceeds ±1, all values in that column are adjusted to maintain 
        consistency by aligning them with the sign of the first affected value.

        Parameters:
        arr (jnp.ndarray): Input 2D JAX array.

        Returns:
        jnp.ndarray: Modified array with normalized values.
        """
        if jnp.all(arr == arr[:, 0].reshape(2,-1)):
            return arr
        # Identify indices where |value| > 1 (i.e., need to be fixed)
        arr = jnp.array(arr,dtype=jnp.float32)
        print(arr)
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

    def predict(self,observation, observable,current_step):

        predictions = []
        # self.map = self.map.at[observable==1].set(observation[observable==1])
        # self.map = self.map.at[observable==0].set(observation[observable==0])
        # self.map = self._move_astroid_or_nebula(self.map, current_step)
        predictions.append(self.map)
        prediction = self.map.copy()
        for i in range(1,self.horizon+1):
            prediction = self._move_astroid_or_nebula(prediction, current_step+i)
            predictions.append(prediction)

        return predictions


if __name__ == "__main__":
    

    def detect_obstacle_entry_leaving():
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

        nebula = Nebula()
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

    def test(nebula_tile_drift_speed):
        from utils import getObservation
        from obs_to_state import State
        seed = 223344
        step, player, obs, cfg, timeleft = getObservation(seed,0)
        state = State(obs, "player_0")
        prev_nebulas = jnp.array(state.nebulas.copy())
        prev_observable = jnp.array(state.observeable_tiles.copy())
        def move_astroid_or_nebula(map_object, steps,nebula_tile_drift_speed):
            """
                Shift objects around in space
                Move the nebula tiles in state.map_features.tile_types up by 1 and to the right by 1
                this is also symmetric nebula tile movement
                copied from https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/501ada7575b6dc0cf7f0bcbf2558322efcef190f/src/luxai_s3/env.py#L640
            """
            
            new_map = jnp.roll(
            map_object,
            shift=(
                1 * jnp.sign(nebula_tile_drift_speed),
                -1 * jnp.sign(nebula_tile_drift_speed),
            ),
            axis=(0, 1),
            )

            new_map = jnp.where(
                steps * nebula_tile_drift_speed % 1 == 0,
                new_map,
                map_object,
                )
            return new_map
        nebula = Nebula(3)
        for step in range(1,42):
            
            step, player, obs, cfg, timeleft = getObservation(seed,step)
            state = State(obs, "player_0")
            nebulas = jnp.array(state.nebulas.copy())
            
            #print(jnp.allclose(nebulas,prediction, atol=1e-6))
            # if not jnp.allclose(nebulas,prediction, atol=1e-6):
            #     print(step)
            #     #print(nebulas)
            #     #print(prediction)
            observable = jnp.array(state.observeable_tiles.copy())
            nebula.learn(nebulas,observable,step-1)
            prediction = nebula.predict(nebulas,step)
            print(step,jnp.allclose(nebulas,prediction, atol=1e-6) ,"\n",prediction.T, "\n")
            prev_nebulas = nebulas
      


  
      

        #Use jnp.allclose to handle potential floating-point inaccuracies
        assert jnp.allclose(nebula.nebula_tile_drift_speed, nebula_tile_drift_speed, atol=1e-6), \
            f"Expected {nebula_tile_drift_speed}, but got {nebula.nebula_tile_drift_speed}"

        print(f"Test passed: Detected drift speed = {nebula.nebula_tile_drift_speed}")

    test(-0.05)
 