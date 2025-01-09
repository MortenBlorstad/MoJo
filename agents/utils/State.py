from luxai_s3.env import EnvObs
import numpy as np
class State():

    def __init__(self, obs:dict,player: str):

        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0

        self.player_units = np.array(obs["units_mask"][self.team_id],dtype=int )
        self.opponent_units = np.array(obs["units_mask"][self.opp_team_id],dtype=int )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) 
    
    def count_units(unit_mask, unit_positions:np.ndarray):
        player_units_count = np.zeros((24, 24), dtype=int)
        available_unit = np.where(unit_mask)[0]
        player_available_unit_count = unit_positions[available_unit]
        np.add.at(player_units_count, tuple(player_available_unit_count.T), 1)

        


