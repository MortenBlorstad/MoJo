from luxai_s3.env import EnvObs
import numpy as np
class State():

    def __init__(self, obs:dict,player: str):

        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0



        # player and opponent unit counts
        unit_mask = np.array(obs["units_mask"]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"]) # shape (max_units, 2)
        
        self.player_units_count = self.count_units(unit_mask[self.team_id],unit_positions[self.team_id]) # SP_1: 24x24
        self.opponent_units_count = self.count_units(unit_mask[self.opp_team_id],unit_positions[self.opp_team_id]) # SP_2:24x24
        
        # vi trenger denne også. TODO hvordan håndtere
        self.unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)


        self.observeable_tiles = np.array(obs["sensor_mask"], dtype=int) #S_O: 24x24

        # amount of energy on the tile (including void field)
        self.energy = np.array(obs["map_features"]["energy"])

        self.nebulas = self.get_tile_type(obs["map_features"]["tile_type"],1 )
        self.asteroids = self.get_tile_type(obs["map_features"]["tile_type"],2 )

        # added relic node to state. TODO we need to add functionality to add relic tiles
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        self.relic_nodes = self.get_relic_node_pos(observed_relic_nodes_mask,observed_relic_node_positions)





    def count_units(self, unit_mask, unit_positions:np.ndarray):
        player_units_count = np.zeros((24, 24), dtype=int)
        available_unit = np.where(unit_mask)[0]
        player_available_unit_count = unit_positions[available_unit]
        np.add.at(player_units_count, tuple(player_available_unit_count.T), 1)
        return player_units_count
    
    def get_relic_node_pos(self, relic_nodes_mask, relic_node_positions):
        relic_node_grid = np.zeros((24, 24), dtype=int)
        visible_relics = np.where(relic_nodes_mask)[0]
        visible_relics_pos = relic_node_positions[visible_relics]
        np.add.at(relic_node_grid, tuple(visible_relics_pos.T), 1)
        return relic_node_grid
    
    def get_tile_type(self, tile_type, type:int):
        """
            tile_type: 24x24 with values 0,1 or 2
            
            type of the tile. 0 is empty, 1 is a nebula tile, 2 is asteroid
            
            return 24x24 matrix with 1 where tile_type value is equal to type, otherwise 0
        """
        grid_tile_type = (tile_type == type).astype(int)
        return grid_tile_type


        


