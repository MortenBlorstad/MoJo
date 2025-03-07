import numpy as np
<<<<<<< HEAD
from world.utils import swapAndFilterObservation
=======
from world.utils import swapAndFilterObservation, get_symmetric_coordinates
>>>>>>> 7515b2ceab11c37e9fed4d289e351ddc4a00fcd7

def positional_encoding(game_step, embedding_dim:int=64):
    """
    Generates a sinusoidal positional encoding for match_step and match_number.
    
    Args:
    - match_step (int): The current step in the match (0-99).
    - match_number (int): The match index (0-4).
    - d_model (int): Dimensionality of encoding (default 16).
    
    Returns:
    - np.array: Positional encoding of shape (d_model,)
    """
    pos = np.array([game_step])  # Shape: (2,)

    # Create frequency scaling factors
    i = np.arange(embedding_dim // 2)  # Half of d_model for sin, half for cos
    denom = np.power(10000, (2 * i / embedding_dim))  # Scaling factor for each dimension

    # Compute sine and cosine encoding
    enc_sin = np.sin(pos[:, None] / denom)  # Shape: (2, d_model/2)
    enc_cos = np.cos(pos[:, None] / denom)  # Shape: (2, d_model/2)

    # Flatten and concatenate
    encoding = np.concatenate([enc_sin.flatten(), enc_cos.flatten()])

    return encoding


class State():
    """
    Represents the game state for a player in the Lux AI Challenge.

    This class processes the raw game observation dictionary into structured data, 
    extracting relevant features such as units, energy, tile types, and relic nodes.
    It provides a convenient interface for accessing game information in a structured format.

    Attributes:
        player (str): The player's identifier (e.g., "player_0" or "player_1").
        opp_player (str): The opponent's identifier.
        team_id (int): The player's team ID (0 for "player_0", 1 for "player_1").
        opp_team_id (int): The opponent's team ID.
        self.teampoints (int): Total points obtained in current game
        player_units_count (np.ndarray): A 24x24 grid indicating the number of player-owned units per tile.
        opponent_units_count (np.ndarray): A 24x24 grid indicating the number of opponent-owned units per tile.
        unit_energys (np.ndarray): Energy values of the player's units.
        observeable_tiles (np.ndarray): A 24x24 grid indicating which tiles are observable.
        energy (np.ndarray): A 24x24 grid showing the amount of energy available on each tile.
        nebulas (np.ndarray): A 24x24 binary grid marking nebula tiles.
        asteroids (np.ndarray): A 24x24 binary grid marking asteroid tiles.
        -relic_nodes (np.ndarray): A 24x24 binary grid marking the presence of relic nodes.
        p0ShipPos (np.ndarray): A 16x2 array of player_0 ship positions.
        p1ShipPos (np.ndarray): A 16x2 array of player_1 ship positions.
        relicPositions (np.ndarray): ox2 array with o = number of observed relic nodes

    Methods:
        make_state(obs: dict): Parses the raw observation dictionary and populates the state attributes.
        count_units(unit_mask, unit_positions: np.ndarray) -> np.ndarray:
            Counts the number of units per tile and returns a 24x24 grid.
        get_relic_node_pos(relic_nodes_mask, relic_node_positions) -> np.ndarray:
            Generates a 24x24 grid indicating the presence of relic nodes.
        get_tile_type(tile_type: np.ndarray, type: int) -> np.ndarray:
            Returns a 24x24 binary grid where the specified tile type is present.
    """

    def __init__(self,obs:dict,player: str):
        """
        Initializes the State object for a given player based on the observation data.

        Args:
            obs (dict): The raw observation dictionary provided by the game environment.
            player (str): The player's identifier, either "player_0" or "player_1".
        """
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.relic_nodes = np.zeros((24, 24), dtype=int)
        self.make_state(obs)

    def make_state(self, obs:dict):

        #Total points obtained in current game (Added by Jørgen 18.02.25)
        self.teampoints = int(obs["team_points"][self.team_id])  
        self.opponent_teampoints = int(obs["team_points"][self.opp_team_id])            
        
        #number of steps taken in the current game/episode
        self.steps = int(obs["steps"])
        
        
        #number of steps taken in the current match
        self.match_steps = int(obs["match_steps"])

        self.step_embedding = positional_encoding(self.match_steps, embedding_dim=64)

        # For obs struct see: https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/kits/README.md#observations 
        unit_mask = np.array(obs["units_mask"]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"]) # shape (max_units, 2)

        self.unit_positions = unit_positions[self.team_id]
        self.unit_mask = unit_mask[self.team_id]

        
        self.player_units_count = self.count_units(unit_mask[self.team_id],unit_positions[self.team_id]) # SP_1: 24x24
        self.opponent_units_count = self.count_units(unit_mask[self.opp_team_id],unit_positions[self.opp_team_id]) # SP_2:24x24
        
        # vi trenger denne også. TODO hvordan håndtere
        unit_energies = np.array(obs["units"]["energy"]) # shape (max_units, T)
<<<<<<< HEAD
=======
        self.unit_energies = unit_energies[self.team_id]
>>>>>>> 7515b2ceab11c37e9fed4d289e351ddc4a00fcd7
    

        self.player_sparse_energy_map = self.get_sparse_energy_map(unit_mask[self.team_id],unit_positions[self.team_id],unit_energies[self.team_id] )
        self.opponent_sparse_energy_map = self.get_sparse_energy_map(unit_mask[self.opp_team_id],unit_positions[self.opp_team_id],unit_energies[self.opp_team_id] )


        #Ship positions, per team (Added by Jørgen 18.02.25)
        self.p0ShipPos = swapAndFilterObservation(obs['units']['position'][self.team_id])
        self.p1ShipPos = swapAndFilterObservation(obs['units']['position'][self.opp_team_id])

        self.p0ShipPos_unfiltered = obs['units']['position'][self.team_id][:, [1, 0]]
        
                
        self.observeable_tiles = np.array(obs["sensor_mask"], dtype=int) #S_O: 24x24

        # amount of energy on the tile (including void field)
        self.energy = np.array(obs["map_features"]["energy"])

        self.nebulas = self.get_tile_type(obs["map_features"]["tile_type"],1 )
        
        self.asteroids = self.get_tile_type(obs["map_features"]["tile_type"],2 )

        #Observed relic positions (Added by Jørgen 18.02.25)
        self.relicPositions = swapAndFilterObservation(obs['relic_nodes'])

        self.player_units_inplay = self.get_units_inplay(unit_mask[self.team_id],unit_energies[self.team_id])

        



        
        #--------------- Removed by Jørgen. This is not needed? ---------------
        
        # added relic node to state. TODO we need to add functionality to add relic tiles
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        relic_nodes = self.get_relic_node_pos(observed_relic_nodes_mask, observed_relic_node_positions)

        self.relic_nodes = np.where((relic_nodes > 0)& (self.relic_nodes==0), 1, self.relic_nodes)

    def get_sparse_energy_map(self, unit_mask:np.ndarray,unit_positions:np.ndarray, unit_energy:np.ndarray)->np.ndarray:
        players_energies = np.zeros((24, 24,16), dtype=int)
        available_unit = np.where(unit_mask)[0]
        unit_energy_masked = unit_energy[available_unit]
        unit_positions_masked = unit_positions[available_unit]
        dim2, dim3 = unit_positions_masked[:,0],unit_positions_masked[:,1]
        #print("index",unit_positions_masked, dim2, dim3)
        players_energies[dim2, dim3, available_unit] = unit_energy_masked
        return players_energies/400

    def get_units_inplay(self, unit_mask: np.ndarray, unit_energy: np.ndarray,)->np.ndarray:
        players_units_inplay = np.zeros((16,), dtype=int)
        has_energy = np.where(unit_energy > 0, True, False)
        players_units_inplay = has_energy & unit_mask
        return players_units_inplay



    def count_units(self, unit_mask:np.ndarray, unit_positions:np.ndarray)->np.ndarray:
        player_units_count = np.zeros((24, 24), dtype=int)
        available_unit = np.where(unit_mask)[0]
        player_available_unit_count = unit_positions[available_unit]
        np.add.at(player_units_count, tuple(player_available_unit_count.T), 1)
        return player_units_count / max(player_units_count.sum(), 1)
    
    #--------------- Removed by Jørgen. This is not needed? ---------------
 
    def get_relic_node_pos(self, relic_nodes_mask:np.ndarray, relic_node_positions:np.ndarray)->np.ndarray:
        relic_node_grid = np.zeros((24, 24), dtype=int)
        visible_relics = np.where(relic_nodes_mask)[0]
        visible_relics_pos = relic_node_positions[visible_relics]
        np.add.at(relic_node_grid, tuple(visible_relics_pos.T), 1)
<<<<<<< HEAD
        return relic_node_grid.T
    
    
=======
        np.add.at(relic_node_grid, get_symmetric_coordinates(tuple(visible_relics_pos.T)), 1)
        return relic_node_grid.T
    
>>>>>>> 7515b2ceab11c37e9fed4d289e351ddc4a00fcd7
    def get_tile_type(self, tile_type:np.ndarray, type:int)->np.ndarray:
        """
            tile_type: 24x24 with values 0,1 or 2
            
            type of the tile. 0 is empty, 1 is a nebula tile, 2 is asteroid
            
            return 24x24 matrix with 1 where tile_type value is equal to type, otherwise 0
        """
        grid_tile_type = (tile_type == type).astype(int)
        return grid_tile_type


        

