import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
import numpy as np
from jax import jit
from universe.utils import getObservation
from universe.obs_to_state import State
from universe.unitpos import Unitpos
from universe.relic import Relics
from universe.nebula_astroid import NebulaAstroid
from universe.energy import Energy
from universe.scalarencoder import NaiveScalarEncoder
from universe.obs_to_state import State
from scipy.spatial.distance import cdist


# region Helper functions
def single_relic_heatmap(relic_position, shape):
    """
    Generates a heatmap for a single relic where:
    - 1.0 for the relic itself
    - 0.9 for adjacent cells
    - 0.8 for cells surrounding the 0.9 layer
    - 0.7 for next layer
    - 0.6 for next layer
    - 0.5 for next layer

    Overlapping influence is not handled here. This function processes one relic at a time.

    Args:
    relic_position (tuple): The (x, y) position of the relic.
    shape (tuple): The shape of the heatmap (H, W).

    Returns:
    np.array: A (H, W) heatmap with influence from a single relic.
    """
    H, W = shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    x, y = relic_position
    heatmap[x, y] = 1.0  # Set the relic position

    # Define influence layers
    influence_layers = [(0.9, 1), (0.8, 2), (0.7, 3), (0.6, 4), (0.5, 5)]  # (Value, distance)

    expanded_area = np.copy(heatmap)

    # Expand influence outward ensuring boundaries are respected
    for value, distance in influence_layers:
        temp_area = np.copy(expanded_area)

        for _ in range(distance):
            temp_area_shifted = np.copy(temp_area)

            if H > 1:
                temp_area_shifted[1:, :] = np.maximum(temp_area_shifted[1:, :], temp_area[:-1, :])  # Down
                temp_area_shifted[:-1, :] = np.maximum(temp_area_shifted[:-1, :], temp_area[1:, :]) # Up
            if W > 1:
                temp_area_shifted[:, 1:] = np.maximum(temp_area_shifted[:, 1:], temp_area[:, :-1])  # Right
                temp_area_shifted[:, :-1] = np.maximum(temp_area_shifted[:, :-1], temp_area[:, 1:]) # Left

            temp_area = temp_area_shifted 

        heatmap += temp_area  # Add influence layer
    
    return heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

# Re-defining and executing the aggregation function

def aggregate_relic_heatmaps(relic_grid):
    """
    Generates an aggregated heatmap by summing the influence of multiple relics.
    
    Args:
    relic_grid (np.array): A (24,24) array with 1s where relics are placed and 0s elsewhere.

    Returns:
    np.array: A (24,24) aggregated heatmap.
    """
    H, W = relic_grid.shape
    aggregated_heatmap = np.zeros((H, W), dtype=np.float32)

    # Get relic positions
    relic_positions = np.argwhere(relic_grid == 1)

    # Sum up heatmaps for each relic
    for pos in relic_positions:
        relic_heatmap = single_relic_heatmap(tuple(pos), (H, W))
        aggregated_heatmap += relic_heatmap  # Sum overlapping influence

    # Avoid zero values appearing in visualization
    aggregated_heatmap = np.clip(aggregated_heatmap, 0, 1.0)

    return aggregated_heatmap


def compute_distance_matrix(positions):
    """
    Computes the pairwise Euclidean distance matrix for 16 units
    and the distance of each unit to the origin (0,0).

    Args:
        positions (np.array): (16,2) array with (x,y) coordinates.
    return:
        distance_matrix (np.array): (16,16) array with pairwise distances.
        padded_distance_matrix (np.array): (24,24) array with padded pairwise distances.
    """
    max_possible_distance = np.sqrt(23**2 + 23**2)  # ~32.52
    distance_matrix = cdist(positions, positions)/max_possible_distance  # Shape: (16,16)
    padded_distance_matrix = np.pad(distance_matrix, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    return distance_matrix, padded_distance_matrix

def penalty_for_proximity(positions, grid_size=23):
    """
    Computes a penalty for an array of positions, penalizing proximity to (0,0).
    Normalized between 0 and 1, rewarding movement in any direction equally.

    Args:
        positions (np.array): Array of shape (N, 2) with (x, y) coordinates.
        grid_size (int): Defines the farthest possible distance.

    Returns:
        np.array: Array of shape (N,) with penalty values between 0 and 1.
    """
    max_distance = grid_size  # Max possible movement in any one direction
    
    # Compute max deviation from (0,0) for each position
    current_distances = np.max(np.abs(positions), axis=1)
    
    # Normalize penalty values
    penalties = current_distances / max_distance
    
    return 1-penalties

import scipy.special

def compute_softmax_penalty(heatmap, unit_positions):
    """
    Converts the heatmap into probabilities (softmax) and penalizes positions 
    that have a low probability of being optimal.

    Returns:
    - np.array: (N,) array of penalties for each unit.
    """
    # Flatten heatmap and apply softmax (normalize into probabilities)
    heatmap_flat = heatmap.flatten()
    heatmap_probs = scipy.special.softmax(heatmap_flat).reshape(heatmap.shape)

    # Get probability of each unit's location
    unit_probs = np.array([heatmap_probs[x, y] for x, y in unit_positions])

    # Define penalty as "1 - probability of being in a good location"
    penalty = 1 - unit_probs  # Higher penalty for less optimal locations

    return penalty


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
    max_possible_distance = 2 * 18  # Max distance between any two points in the arch

    return distances/max_possible_distance



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


import numpy as np

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
            # Compute stacking penalties (increase penalty with more stacking)
            penalties = (player_units_count[valid_positions[:, 0], valid_positions[:, 1]] - 1)/2
            penalties = np.clip(penalties, 0, None)  # Ensure penalty is never negative
            
            # Assign penalties only for valid indices
            stacking_in_pointzone_penalty[valid_indices] = penalties

    return stacking_in_pointzone_penalty


def compute_distances_from_map_center(unit_positions, grid_size=(24, 24)):
    # Compute the center of the grid
    center_x, center_y = (grid_size[0] // 2, grid_size[1] // 2)  # (12, 12)
    # Compute Manhattan distances
    manhattan_distances = np.abs(unit_positions[:, 0] - center_x) + np.abs(unit_positions[:, 1] - center_y)
    return manhattan_distances/(center_x + center_y - 2) # Normalize by maximum distance 


def get_closest_relic_distance(ship_positions, relic_nodes, max_distance=12):
    """
    Computes the minimum distance from each ship to the closest relic.
    Handles cases where no relics exist.

    Parameters:
    - ship_positions (np.array): (N,2) array of (x, y) ship positions.
    - relic_nodes (np.array): 2D grid where `1` represents relic locations.
    - max_distance (int): The maximum grid distance for normalization.

    Returns:
    - np.array: (N,) array of normalized distances.
    """
    # Ensure ship_positions is a NumPy array and properly shaped
    ship_positions = np.asarray(ship_positions)
    if ship_positions.ndim != 2 or ship_positions.shape[1] != 2:
        raise ValueError(f"ship_positions must have shape (N,2), but got {ship_positions.shape}")

    # Extract (x, y) positions of relics
    relic_positions = np.argwhere(relic_nodes == 1)  # Ensures shape (M,2)

    # If there are no relics, return max penalty distance for all ships
    if relic_positions.shape[0] == 0:
        return np.ones(len(ship_positions))  # Max penalty (distance = 1)

    # Compute distances between ships and relics
    distances = cdist(ship_positions, relic_positions)  # (N, M)

    # Get the closest relic distance for each ship
    closest_distances = np.min(distances, axis=1)  # (N,)

    # Normalize by max possible diagonal distance in a 12x12 space
    max_possible_distance = np.sqrt(max_distance**2 + max_distance**2)
    return closest_distances / max_possible_distance


# endregion

class Universe():

    def __init__(self, player:str, configuration:dict, horizont:int = 1, seed:int = None):
      
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
        
        self.relic_heatmap = np.zeros((24,24))
        self.observed_map = np.zeros((24,24))



        self.arc_positions = generate_arch_positions(start_pos=(0, 0), radius=18, num_positions=16)
        self.zap_options = jnp.zeros((24,24))

    def get_position_info(self, unit_index:int)->np.ndarray:
        """
        Get the position information for a unit.

        Args:
            unit_index (int): The index of the unit.
        return:
            np.ndarray: The position information, shape (2 + 16). Normalized x, y, relative distance to the other units.
        """
        return np.concat([self.p0ShipPos_unfiltered[unit_index]/23 ,self.distance_matrix[unit_index]] )
        


    def get_reward(self,state:State) -> np.ndarray:
        """
        Returns a (1, 16) array with 1 for each unit in a relic zone if the team receives points; else 0.
        """
        in_points_zone = is_unit_within_radius(state.relic_nodes, state.p0ShipPos_unfiltered, radius = 5)

        point_factor = np.where(in_points_zone, 1, 0)
        got_points = np.where(self.thiscore > 0, 1, 0)
        reward = np.expand_dims(point_factor*got_points, axis=0)
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
    def predict(self, observation:dict)->dict[str, np.ndarray]:        

        #Create state from observation        
        state:State = State(observation,self.player)
        self.p0ShipPos_unfiltered = state.p0ShipPos_unfiltered


        self.teampoints = state.teampoints
        self.opponent_teampoints = state.opponent_teampoints
        if state.match_steps == 0:
            self.totalscore = 0     #Overall score
            self.thiscore = 0       #Score current step    
            self.opponent_totalscore = 0     #Overall score
            self.opponent_thiscore = 0       #Score current step   
            self.observed_map = np.zeros((24,24))

       
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


        self.relic_heatmap = aggregate_relic_heatmaps(state.relic_nodes)
        self.observed_map = np.where(state.observeable_tiles == 1, 1, 0) 
        
        
        #Learn relic tiles        
        self.relics.learn(            
            state.relicPositions,   #Position of (visible) relic nodes 
            self.thiscore,          #How many points we scored                     
            state.p0ShipPos         #The whereabouts of our mighty fleet
        )        

        #Predict Nebula and Astroid here        
        nebulas, astroids = self.nebula_astroid.predict(state.nebulas,state.asteroids, state.observeable_tiles, current_step=state.steps)

        #OBS: get unobserved_terrain before calling nan_to_num on nebulas & astr0oids. (nan == unobserved_terrain)        
        unobserved_terrain = np.expand_dims(get_unobserved_terrain(nebulas,astroids)[0],axis=0) # dont need to horizon here.
      
              
        self.reward  = self.get_reward(state)     

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

        # relic heatmap
      
        predictions.append(np.expand_dims(self.relic_heatmap, axis=0))
        # observed map
        predictions.append(np.expand_dims(self.observed_map, axis=0))

        # add relative distances 
        self.distance_matrix, self.padded_distance_matrix = compute_distance_matrix(state.p0ShipPos_unfiltered)
        predictions.append(np.expand_dims(self.padded_distance_matrix, axis=0))

        

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

