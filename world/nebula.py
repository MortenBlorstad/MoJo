from world.base_component import base_component
import jax
import jax.numpy as jnp
import numpy as np


class Nebula(base_component):

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        #self.map = jnp.zeros((1+horizon,24,24))
        self.nebula_tile_drift_speed = 0.025
        self.change_steps = [7, 10, 14, 20, 27, 30, 34, 40, 47, 50, 54, 60, 67, 70, 74, 80, 87, 90, 94, 100]
        self.change_steps_set = set(self.change_steps)
        self.previous_observed_change = 0
        self.change_rate = 0
        self.prev_step = 0
        self.map = jnp.ones((24,24))/3

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

    def update_change_rate(self,new_rate):
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
        Detects obstacles entering the grid from the top row or last column.

        Args:
            state (jnp.ndarray): The current grid state (before the update).
            next_state (jnp.ndarray): The next grid state (after the update).

        Returns:
            dict: Contains boolean values indicating if obstacles entered from top row or last column.
        """
        # Calculate the difference between next_state and state


        # Detect entries from the top row (row index 0) where it changed from 0 to 1
        last_col_entries = jnp.where((state[-1, :] == 0) & (next_state[-1, :] == 1))[0]
        
        # Detect entries from the last column (column index -1) where it changed from 0 to 1
        top_row_entries = jnp.where((state[:, 0] == 0) & (next_state[:, 0] == 1))[0]
        return top_row_entries.size > 0 or last_col_entries.size > 0
    
    def detect_obstacle_leaving(self,state, next_state):
        """
        Detects obstacles entering the grid from the top row or last column.

        Args:
            state (jnp.ndarray): The current grid state (before the update).
            next_state (jnp.ndarray): The next grid state (after the update).

        Returns:
            dict: Contains boolean values indicating if obstacles entered from top row or last column.
        """
        # Calculate the difference between next_state and state


        # Detect entries from the top row (row index 0) where it changed from 0 to 1
        first_col_entries = jnp.where((state[0, :] == -1) & (next_state[-1, :] == 0))[0]
        
        # Detect entries from the last column (column index -1) where it changed from 0 to 1
        last_row_entries = jnp.where((state[:, -1] == -1) & (next_state[:, -1] == 0))[0]
        return last_row_entries.size > 0 or first_col_entries.size > 0
        
    



    def learn(self, observation, observable, current_step):
        """
            check whether nebula_tile_drift_speed in one of the following values {-0.5: move up-right every other step,-0.25: move up-right every 4th step,0.25: move down-left every 4th step,0.5 move down-left every second step}
            obs: (5x24,24) four time steps of nebula positions, (current_step-5, current_step-3, current_step-2, current_step-1, current_step) current time step + 4 previous
            current_step: current step number
        """
        assert observation.shape == (24,24) and observable.shape == (24,24)

        
        if not self.should_check(current_step) or current_step ==0:
            self.prev_observation = observation
            self.prev_observable = observable
            self.prev_step = current_step
            self.map = self._move_astroid_or_nebula(self.map ,current_step)
            self.map = self.map.at[observable==1].set(observation[observable==1])
            return



        observation_masked = observation.at[self.prev_observable==0].set(0)

        

        delta = observation_masked - self.prev_observation
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

        change = self.detect_obstacle_entry(self.prev_observation,observation_masked)
        change = change or self.detect_obstacle_leaving(self.prev_observation,observation_masked)

        if not np.any(delta == -1):
            print(f"no change detected at change step {current_step}")
        else:
            
            self.update_change_rate(1/(current_step- self.previous_observed_change))
            self.previous_observed_change = current_step
            # print(current_step, delta.T)   
            # print(current_step, observation_masked.T) 
            # print(self.prev_step, self.prev_observation.T) 
            moved_from_indices = jnp.array(jnp.where(delta==-1))
            moved_to_indices = jnp.array(jnp.where(delta==1))
            if moved_from_indices.shape != moved_to_indices.shape:
                possible_directions = jnp.array([[1, -1], [-1, 1]])  # (row change, column change)
                directions = [1,-1]
                moved_from_plus_dir1 = moved_from_indices + directions[0][:, None]  # Move (1,-1)
                moved_from_plus_dir2 = moved_from_indices + directions[1][:, None]  # Move (-1,1)
                matches_dir1 = jnp.sum((moved_from_plus_dir1[:, :, None] == moved_to_indices[:, None, :]).all(axis=0), axis=1)
                matches_dir2 = jnp.sum((moved_from_plus_dir2[:, :, None] == moved_to_indices[:, None, :]).all(axis=0), axis=1)
                direction = directions[jnp.argmax(jnp.array([matches_dir1.sum(), matches_dir2.sum()]))]
            else:
                directions = (moved_from_indices - moved_to_indices)
                #print(self.prev_step,current_step, direction, self.change_rate)
            
                expected_direction_1 = jnp.array([[1], [-1]])  # Down-left movement
                expected_direction_2 = jnp.array([[-1], [1]])  # Up-right movement
                # Check if all movements follow the same pattern
                if jnp.all(directions == expected_direction_1):
                    direction = -1
                elif jnp.all(directions == expected_direction_2):
                    direction = 1
                else:
                    raise Exception("Mixed movement or no consistent pattern")
        
            self.nebula_tile_drift_speed = direction*self.closest_change_rate(self.change_rate)
            #print(current_step, self.nebula_tile_drift_speed, direction*self.change_rate)
            print(f"Change detected at change step {current_step}. Nebula speed is {self.nebula_tile_drift_speed}")    
            

        self.prev_observation = observation
        self.prev_observable = observable
        self.prev_step = current_step

        self.map = self._move_astroid_or_nebula(self.map, current_step)
        self.map = self.map.at[observable==1].set(observation[observable==1])

    


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
 