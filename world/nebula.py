from base_component import base_component
import jax
import jax.numpy as jnp
import numpy as np


class Nebula(base_component):

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.map = jnp.zeros((1+horizon,24,24))
        self.nebula_tile_drift_speed = np.mean(self.env_params_ranges["nebula_tile_drift_speed"],dtype=np.float16) # 0
        self.steps = 0

    def _move_astroid_or_nebula(self, map_object,step):
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

        # Conditionally update the map based on drift speed and step count
        new_map = jnp.where(
            step * self.nebula_tile_drift_speed % 1 == 0,  # Check if it's time to apply the shift
            new_map,   # Apply the shifted map if condition is met (step interval reached)
            map_object # Otherwise, keep the original map unchanged
        )
        return new_map

    def learn(self, obs,current_step):
        """
            check whether nebula_tile_drift_speed in one of the following values {-0.5: move up-right every other step,-0.25: move up-right every 4th step,0.25: move down-left every 4th step,0.5 move down-left every second step}
            obs: (5x24,24) four time steps of nebula positions, (current_step-5, current_step-3, current_step-2, current_step-1, current_step) current time step + 4 previous
            current_step: current step number
        """
        possible_drift_speeds = {
            -0.5: (-1, 1),   # Move up-right every other step
            -0.25: (-1, 1),  # Move up-right every 4th step
            0.25: (1, -1),   # Move down-left every 4th step
            0.5: (1, -1)     # Move down-left every second step
        }

        odd = int(current_step%2==1)
        second_check_ind = 6-1-odd
        first_check_ind = second_check_ind-2
        
        start_state = obs[first_check_ind-2]
        first_check = obs[first_check_ind]
        second_check = obs[second_check_ind]

        # Compare movement patterns to find matching drift speed
        for speed, (row_shift, col_shift) in possible_drift_speeds.items():
            # Check if the difference in state matches the expected movement pattern
            first_state = jnp.roll(start_state, shift=(row_shift, col_shift), axis=(0, 1)) #first potential change
            
            second_state = jnp.roll(first_check, shift=(row_shift, col_shift), axis=(0, 1)) #second potential change
            
            # If movement pattern matches, assign the corresponding drift speed
            # Compare calculated movement with observed changes
            if jnp.allclose(first_check, first_state,atol=1e-6) and jnp.allclose(second_check, second_state,atol=1e-6):
                self.nebula_tile_drift_speed = jnp.sign(speed).item()*0.5
                return
            if jnp.allclose(first_check, first_state,atol=1e-6) or jnp.allclose(second_check, second_state,atol=1e-6):
                self.nebula_tile_drift_speed = jnp.sign(speed).item()*0.25
                return
           
        # Default to zero if no movement pattern is detected
        self.nebula_tile_drift_speed = 0.0
        
    def predict(self):
        self.map = self._move_astroid_or_nebula(self.map)
        return self.map


if __name__ == "__main__":
    

    def test(nebula_tile_drift_speed):
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
        nebula = Nebula(4)
        obs = jnp.zeros((6,10,10),dtype=int)
        state = jnp.zeros((10,10),dtype=int)
        state = state.at[5:(5+4),1:(1+4)].set(1)
        for step in range(6):
            state = move_astroid_or_nebula(state,step,nebula_tile_drift_speed)
            obs = obs.at[step,:,:].set(state)
            nebula.steps+=1
        nebula.learn(obs,step)

        # Use jnp.allclose to handle potential floating-point inaccuracies
        assert jnp.allclose(nebula.nebula_tile_drift_speed, nebula_tile_drift_speed, atol=1e-6), \
            f"Expected {nebula_tile_drift_speed}, but got {nebula.nebula_tile_drift_speed}"

        print(f"Test passed: Detected drift speed = {nebula.nebula_tile_drift_speed}")

    test(-0.25)
    test(-0.5)
    test(0.25)
    test(0.5)