import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
import numpy as np
from jax import jit
from world.utils import getObservation
from world.obs_to_state import State
from world.unitpos import Unitpos
from world.relic import Relics
from world.nebula_astroid import NebulaAstroid
from world.energy import Energy
from world.scalarencoder import NaiveScalarEncoder
from world.obs_to_state import State


def generate_arch_positions(start_pos=(0, 0), radius=18, num_positions=16):
    """
    Generates evenly spread positions in an arch from 0° to 90° relative to a given start position.

    Parameters:
        start_pos (tuple): The starting (x, y) position, default is (0,0).
        radius (int): The radius of the arch.
        num_positions (int): Number of positions to generate.

    Returns:
        np.array: A (num_positions, 2) array of (x, y) positions.
    """
    angles = np.linspace(0, np.pi / 2, num_positions)  # Convert degrees (0 to 90) to radians
    positions = np.array([
        (
            start_pos[0] + radius * np.cos(angle),  # X-coordinate
            start_pos[1] + radius * np.sin(angle)   # Y-coordinate
        )
        for angle in angles
    ])

    even_indices = positions[::2]  # Get positions at even indices (0,2,4,...)
    odd_indices = positions[1::2][::-1]  # Get positions at odd indices in reverse order (15,13,11,...)
    reordered_positions = np.zeros((num_positions, 2))
    reordered_positions[::2] = even_indices  # Assign even indices
    reordered_positions[1::2] = odd_indices  # Assign odd indices
    #reordered_positions = np.vstack((even_indices, odd_indices))
    
    return reordered_positions

def compute_distances_to_arch(unit_positions, arch_positions):
    """
    Computes the Euclidean distance from each unit position to its respective arch position.

    Parameters:
        unit_positions (np.array): (16,2) array of (x, y) positions of units.
        arch_positions (np.array): (16,2) array of (x, y) positions in the arch.

    Returns:
        np.array: (16,) array of distances.
    """
    # Ensure inputs are valid
    assert unit_positions.shape == (16, 2), "unit_positions must have shape (16,2)"
    assert arch_positions.shape == (16, 2), "arch_positions must have shape (16,2)"

    # Compute Euclidean distance for each corresponding unit and arch position
    distances = np.linalg.norm(unit_positions - arch_positions, axis=1)

    return distances



@jit
def get_unobserved_terrain(nebulas:jnp.ndarray, astroids:jnp.ndarray)->jnp.ndarray:
    return jnp.where(jnp.isnan(nebulas) | jnp.isnan(astroids), 1, 0)


env_params_ranges = dict(
    unit_move_cost=                 list(range(1, 6)), # list(range(x, y)) = [x, x+1, x+2, ... , y-1]
    unit_sensor_range=              [1, 2, 3, 4],
    nebula_tile_vision_reduction=   list(range(0, 8)),
    nebula_tile_energy_reduction=   [0, 1, 2, 3, 5, 25],
    unit_sap_cost=                  list(range(30, 51)),
    unit_sap_range=                 list(range(3, 8)),
    unit_sap_dropoff_factor=        [0.25, 0.5, 1],
    unit_energy_void_factor=        [0.0625, 0.125, 0.25, 0.375],
    # map randomizations
    nebula_tile_drift_speed=        [-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15],
    energy_node_drift_speed=        [0.01, 0.02, 0.03, 0.04, 0.05],
    energy_node_drift_magnitude=    list(range(3, 6)),
)

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def positional_encoding(x, y, map_size):
    return np.array([x / map_size[0], y / map_size[1]])



def is_unit_within_radius(grid, unit_positions, radius):
    """
    Checks if each unit is within a given radius of any cell containing `1`.

    Args:
    - grid (np.array): 2D array where `1` represents target cells.
    - unit_positions (np.array): (N,2) array of unit (row, col) positions.
    - radius (float): Maximum allowed distance.

    Returns:
    - np.array: (N,) boolean array, True if the unit is within the radius of any `1`, else False.
    """
    # Find all positions where the grid has `1`s
    target_positions = np.argwhere(grid == 1)  # Shape (M, 2), M = number of ones
    if target_positions.size == 0:
        return np.zeros(len(unit_positions), dtype=bool)  # No targets, all False

    # Compute pairwise Euclidean distance between units and target positions
    unit_positions = unit_positions[:, None, :]  # Reshape to (N,1,2) for broadcasting
    target_positions = target_positions[None, :, :]  # Reshape to (1,M,2) for broadcasting

    distances = np.linalg.norm(unit_positions - target_positions, axis=2)  # (N, M) distances

    # Return True for each unit that is within the given radius of ANY target position
    return np.any(distances <= radius, axis=1)  # (N,) boolean array


def calculate_stacking_penalty(in_points_zone, player_units_count, p0ShipPos_unfiltered, stacking_in_pointzone_penalty):
    """
    Calculates the stacking penalty for units in a designated zone.

    Parameters:
        in_points_zone (np.array): Boolean array indicating units in the zone (shape: (N,))
        player_units_count (np.array): 2D array representing unit counts at each grid location (shape: (grid_x, grid_y))
        p0ShipPos_unfiltered (np.array): Array of unit positions (shape: (N, 2))
        stacking_in_pointzone_penalty (np.array): Penalty array (same shape as in_points_zone)

    Returns:
        np.array: Updated stacking penalty array
    """
    # Get indices of units in the zone
    valid_indices = np.where(in_points_zone)[0]

    if valid_indices.size > 0:
        # Get positions of units in the zone
        positions = p0ShipPos_unfiltered[valid_indices]

        # Filter out invalid positions (-1, -1)
        valid_positions_mask = (positions[:, 0] >= 0) & (positions[:, 1] >= 0)
        valid_positions = positions[valid_positions_mask]
        valid_indices = valid_indices[valid_positions_mask]  # Filter indices accordingly

        if valid_positions.shape[0] > 0:
            # Assign penalties only for valid indices
            stacking_in_pointzone_penalty[valid_indices] = (1 - player_units_count[valid_positions[:, 0], valid_positions[:, 1]])/6
    
    return stacking_in_pointzone_penalty

def compute_distances_from_map_center(unit_positions, grid_size=(24, 24)):
    # Compute the center of the grid
    center_x, center_y = (grid_size[0] // 2, grid_size[1] // 2)  # (12, 12)
    # Compute Manhattan distances
    manhattan_distances = np.abs(unit_positions[:, 0] - center_x) + np.abs(unit_positions[:, 1] - center_y)
    return manhattan_distances/(center_x + center_y - 2) # Normalize by maximum distance 


class Universe():

    def __init__(self, player:str, configuration:dict, horizont:int = 3, seed:int = None):
      
        #The initial observation has this structure
        #-------------------------------------------------------------------------------------------
        #   {
        #   "step":0,
        #   "obs":...},                                 <------ A normal observation. Add to queue
        #   "remainingOverageTime":600,                 <------ Useful
        #   "player":"player_1",                        <------ We need this
        #   "info":{
        #       "env_cfg":{
        #           "max_units":16,
        #           "match_count_per_episode":5,
        #           "max_steps_in_match":100,
        #           "map_height":24,
        #           "map_width":24,
        #           "num_teams":2,
        #           "unit_move_cost":5,                 <------ Useful
        #           "unit_sap_cost":36,                 <------ Useful
        #           "unit_sap_range":5,                 <------ Useful
        #           "unit_sensor_range":4               <------ Useful
        #           }
        #       }
        #   }
        #
        #-------------------------------------------------------------------------------------------

        #The observable parameters
        self.configuration = configuration
        self.unit_sap_range = configuration["unit_sap_range"] # needed to compute sapping for agent

        self.unit_sap_range_norm = normalize(self.unit_sap_range, min_val=3, max_val=7) #unit_sap_range=list(range(3, 8))

        self.unit_move_cost = normalize(configuration["unit_move_cost"],min_val=1, max_val=5) # unit_move_cost=list(range(1, 6))
        self.unit_sap_cost = normalize(configuration["unit_sap_cost"], min_val=30, max_val=50) #unit_sap_cost=list(range(30, 51))
        self.unit_sensor_range = normalize(configuration["unit_sensor_range"],min_val=1,max_val=4) #unit_sensor_range=[1, 2, 3, 4]

        self.scaler_features = np.array([self.unit_sap_range_norm, self.unit_move_cost, self.unit_sap_cost, self.unit_sensor_range])

        #Number of 'future universes' we predict
        self.horizont = horizont

        #Determine players
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0 

        #Other stuff
        self.totalscore = 0     #Overall score
        self.thiscore = 0       #Score current step    

        self.opponent_totalscore = 0     #Overall score
        self.opponent_thiscore = 0       #Score current step   
        
        self.nebula_astroid = NebulaAstroid(self.horizont)
        self.p0pos = Unitpos(self.horizont)
        self.p1pos = Unitpos(self.horizont)
        self.relics = Relics()
        self.energy = Energy(self.horizont)
        self.scalar = NaiveScalarEncoder(env_params_ranges)


        self.arc_positions = generate_arch_positions(start_pos=(0, 0), radius=18, num_positions=16)
        self.zap_options = jnp.zeros((24,24))

    


    def get_reward(self,state:State, unexplored_count: int) -> float:
        """Calculate the reward for the current step.
        Args:
            a (int): The points of the team
            b (int): The points of the opponent team"""
        match_steps = state.match_steps
        steps = state.steps

        #close_to_start = (np.sqrt(23**2 +23**2) - np.linalg.norm(state.p0ShipPos_unfiltered, axis=1)) / np.sqrt(23**2 +23**2) # (16x1)
        
        in_points_zone = is_unit_within_radius(state.relic_nodes, state.p0ShipPos_unfiltered, radius= 4)

        #distance_from_center = compute_distances_from_map_center(state.p0ShipPos_unfiltered)

        distance_from_arch = compute_distances_to_arch(state.p0ShipPos_unfiltered, self.arc_positions)

        a, b = self.teampoints, self.opponent_teampoints
        points_ratio = (a - b) / (a + b + 1)

        a, b = self.thiscore, self.opponent_thiscore
        this_points_ratio = (a - b) / (a + b + 1)

        n_units = self.unit_mask.sum()
        n_units_inplay = self.units_inplay.sum()
        unit_score = (n_units - n_units_inplay) / (n_units + 1)
        unexplored_ratio = unexplored_count/(24*24) 
        num_in_points_zone = in_points_zone.sum()
        point_factor = np.where(in_points_zone, 1/max(num_in_points_zone, 1), 0.01/max(16-num_in_points_zone,1 ))

        relic_found = np.any(state.relic_nodes == 1)

        if steps < 50:
            factor = 0.8
        else:
            if match_steps < 50:
                factor = 0.25
            elif relic_found:
                factor = np.linspace(0.25,0.01, 16)
            else:
                factor = 0.5
            #factor = 0.05*(505/(steps**1.333+505))

        distance_reward = distance_from_arch*factor #(distance_from_center + close_to_start) * factor

        stacking_in_pointzone_penalty = np.zeros(16)
        stacking_in_pointzone_penalty = calculate_stacking_penalty(in_points_zone, state.player_units_count, state.p0ShipPos_unfiltered, stacking_in_pointzone_penalty)
        reward = np.expand_dims(points_ratio + 0.2*this_points_ratio + point_factor*self.thiscore - unit_score*0.01 - unexplored_ratio * 0.01 - distance_reward, axis=0) + stacking_in_pointzone_penalty
        return reward

    def get_one_hot_pos(self, idx:int)->np.ndarray:
        available = self.unit_mask[idx]
        one_hot_pos = np.zeros((1,24*24))
        if not available:
            return available, one_hot_pos, (-1,-1)
        x, y = self.unit_positions[idx]
        assert 0 <= x < 24 and 0 <= y < 24, f"Invalid coordinates: ({x}, {y})"

        index = y * 24 + x # Convert 2D (x, y) to 1D index
        one_hot_pos[0, index] = 1
        return available, one_hot_pos, (int(y),int(x))
    
    def get_one_hot_unit_id(self, idx:int)->np.ndarray:
        one_hot_pos = np.zeros((1,16))
        one_hot_pos[0, idx] = 1
        return one_hot_pos
    

    #s_{t:t+h} | o_{t}
    def predict(self, observation:dict):        

        #Create state from observation        
        state:State = State(observation,self.player)

        self.teampoints = state.teampoints
        self.opponent_teampoints = state.opponent_teampoints
        if state.match_steps == 0:
            self.totalscore = 0     #Overall score
            self.thiscore = 0       #Score current step    
            self.opponent_totalscore = 0     #Overall score
            self.opponent_thiscore = 0       #Score current step   

       
        self.units_inplay = state.player_units_inplay

        self.unit_positions = state.unit_positions
        self.unit_mask = state.unit_mask
        
        if state.steps <= 101:
            self.nebula_astroid.learn(state.nebulas, state.asteroids, state.observeable_tiles, current_step=state.steps)               
            self.energy.learn(current_step=state.steps, observation=state.energy, pos1=state.player_units_count, pos2=state.opponent_units_count, observable=state.observeable_tiles)
        


        self.p0pos.learn(state.p0ShipPos)        
        self.p1pos.learn(state.p1ShipPos)        
        
        #Update points
        self.thiscore = state.teampoints - self.totalscore
        self.opponent_thiscore = state.opponent_teampoints - self.opponent_totalscore


        self.totalscore += self.thiscore
        self.opponent_totalscore += self.opponent_thiscore
        
        
        
        #Learn relic tiles        
        self.relics.learn(            
            state.relicPositions,   #Position of (visible) relic nodes 
            self.thiscore,          #How many points we scored                     
            state.p0ShipPos         #The whereabouts of our mighty fleet
        )        

        #Predict Nebula and Astroid here        
        nebulas, astroids = self.nebula_astroid.predict(state.nebulas,state.asteroids, state.observeable_tiles, current_step=state.steps)

        #OBS: get unobserved_terrain before calling nan_to_num on nebulas & astr0oids. (nan == unobserved_terrain)        
        unobserved_terrain = get_unobserved_terrain(nebulas,astroids)
        unexplored_count = unobserved_terrain[0].sum().item() # number of unexplored tiles
              
        self.reward  = self.get_reward(state, unexplored_count)     

        #Create list of predictions
        predictions = [jnp.nan_to_num(nebulas), jnp.nan_to_num(astroids), unobserved_terrain]
        
        #Add player position predictions 
        p0pos = self.p0pos.predict(astroids[1:]) 
        p1pos = self.p1pos.predict(astroids[1:])

        self.zap_options = p1pos[1]
           
        predictions.append(p0pos)        
        predictions.append(p1pos)
        

        #Predict Relic tiles        
        predictions.append(jnp.nan_to_num(self.relics.predict()))   # quick fix: added jnp.nan_to_num to avoid NaNs in the output
       
        #Predict Energy  
        predictions.append(jnp.nan_to_num(self.energy.predict(current_step=state.steps) ))    


        # ship energy
        predictions.append(state.player_sparse_energy_map)      
        #predictions.append(state.opponent_sparse_energy_map)  
        
        #Add the scalar parameters, encoded into a 24x24 grid        
        # predictions.append(
        #      self.scalar.Encode(
        #         unit_move_cost = 2,
        #         nebula_tile_drift_speed=0.05,
        #         unit_sap_cost = 50                
        #     )
        # )


        #Add positional encoding

        step_embedding = np.expand_dims(state.step_embedding, axis=0) # Expand to batch shape
        

        stacked_array = np.concatenate(predictions, axis=0)  
        stacked_array = np.expand_dims(stacked_array, axis=0)  # Expand to batch shape
        

        sum_score = max(self.teampoints+self.opponent_teampoints,1)
        scalers = np.expand_dims(self.scaler_features, axis=0) # Expand to batch shape
        new_features = np.array([[self.teampoints / sum_score, self.opponent_teampoints / sum_score]])  # Shape: (1,2)

        scalers = np.concatenate((scalers, new_features), axis=1) # Shape: (1, 6)
        
        one_hot_unit_id = np.expand_dims(np.eye(16), axis=0)

        
        one_hot_unit_energy = np.diag(state.unit_energies) / 100 # init_unit_energy: int = 100, max_unit_energy: int = 400
        one_hot_unit_energy = np.expand_dims(one_hot_unit_energy, axis=0)

        state = {"image": stacked_array, "step_embedding": step_embedding, "scalars": scalers, "one_hot_unit_id": one_hot_unit_id, "one_hot_unit_energy": one_hot_unit_energy} 
        return state  


        

#Test function for Jørgen
def jorgen():

    print("Running Jørgens tests")

    #Fix a seed for testing. 
    seed = 223344

    #Get initial observation
    step, player, obs, cfg, timeleft = getObservation(seed,0)
    
    #Create a fixed seed universe
    u = Universe(player,obs,cfg,horizont=3, seed=seed)

    for i in range(1,2):
        
        #Get another observation
        _, _, obs, _, timeleft = getObservation(seed,i)        

        #Test universe prediction
        u.predict(obs)

if __name__ == "__main__":
    jorgen()
