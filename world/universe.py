#from luxai_s3.wrappers import LuxAIS3GymEnvCheat

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
from jax import jit
import json
from world.utils import getObservation, getObsNamespace, getPath, from_json
from world.obsqueue import ObservationQueue
from abc import ABC, abstractmethod
import flax
import flax.serialization

from world.obs_to_state import State
from world.unitpos import Unitpos
from world.relic import Relics
from world.nebula_astroid import NebulaAstroid
from world.energy import Energy
from world.scalarencoder import NaiveScalarEncoder
from world.obs_to_state import State

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



class Universe():

    def __init__(self, player:str, observation:jnp.ndarray, configuration:dict, horizont:int = 3, seed:int = None):
      
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

        #Number of 'future universes' we predict
        self.horizont = horizont

        #History of observations
        self.obsQueue = ObservationQueue(10)
        
        #Add initial observation to queue
        self.obsQueue(observation)

        #Determine players
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0 

        #Other stuff
        self.totalscore = 0     #Overall score
        self.thiscore = 0       #Score current step    
        
        self.nebula_astroid = NebulaAstroid(self.horizont)
        self.p1pos = Unitpos(self.horizont)
        self.p2pos = Unitpos(self.horizont)
        self.relics = Relics()
        self.energy = Energy(self.horizont)
        self.scalar = NaiveScalarEncoder(env_params_ranges)

    #s_{t:t+h} | o_{t}
    def predict(self, observation:dict):

        #Create state from observation
        state:State = State(observation,self.player)        
        self.obsQueue(observation)
        
        self.nebula_astroid.learn(state.nebulas,state.asteroids,state.observeable_tiles, current_step=state.steps)
        self.energy.learn(current_step=state.steps, observation=state.energy,
                          pos1=state.player_units_count, pos2=state.opponent_units_count, observable=state.observeable_tiles)
        
        self.p1pos.learn(self.obsQueue.Last(['units','position',self.team_id]))
        self.p2pos.learn(self.obsQueue.Last(['units','position',self.opp_team_id]))

        #Update points
        self.thiscore = self.obsQueue.Last(['team_points', self.team_id])[0] - self.totalscore
        self.totalscore += self.thiscore

        #Learn relic tiles
        self.relics.learn(
            self.obsQueue.Last(['relic_nodes']),                    #Position of (visible) relic nodes            
            self.thiscore,                                          #How many points we scored               
            self.obsQueue.Last(['units', 'position', self.team_id]) #The whereabouts of our mighty fleet
        )              
        
        #Predict Nebula and Astroid here
        nebulas, astroids = self.nebula_astroid.predict(state.nebulas,state.asteroids, state.observeable_tiles, current_step=state.steps)
        
        #OBS: get unobserved_terrain before calling nan_to_num on nebulas & astr0oids. (nan == unobserved_terrain)
        unobserved_terrain = get_unobserved_terrain(nebulas,astroids)

        #Create list of predictions
        predictions = [jnp.nan_to_num(nebulas), jnp.nan_to_num(astroids), unobserved_terrain]

        #Add player position predictions
        predictions.append(self.p1pos.predict(astroids[1:]))
        predictions.append(self.p2pos.predict(astroids[1:]))     

        #Predict Relic tiles
        predictions.append(self.relics.predict())

        #Predict Energy
        predictions.append(self.energy.predict(current_step=state.steps))
        
        #Add the scalar parameters, encoded into a 24x24 grid
        predictions.append(
             self.scalar.Encode(
                unit_move_cost = 2,
                nebula_tile_drift_speed=0.05,
                unit_sap_cost = 50
            )
        )

        stacked_array = jnp.concat(predictions)
        print(stacked_array.shape)
        return stacked_array  
        

#Test function for Jørgen
def jorgen():

    print("Running Jørgens tests")

    #Fix a seed for testing. 
    seed = 223344

    #Get initial observation
    step, player, obs, cfg, timeleft = getObservation(seed,0)
    
    #Create a fixed seed universe
    u = Universe(player,obs,cfg,horizont=3, seed=seed)

    for i in range(1,3):
        
        #Get another observation
        _, _, obs, _, timeleft = getObservation(seed,i)        

        #Test universe prediction
        u.predict(obs)

if __name__ == "__main__":
    jorgen()