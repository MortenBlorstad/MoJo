#from luxai_s3.wrappers import LuxAIS3GymEnvCheat

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
import json
from world.utils import getObservation, getObsNamespace, getPath, from_json
from world.obsqueue import ObservationQueue
from abc import ABC, abstractmethod
import flax
import flax.serialization
from world.obs_to_state import State
import socket

from world.unitpos import Unitpos
from world.relic import Relics
from world.nebula_astroid import NebulaAstroid
from world.obs_to_state import State

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

        # self.relic = Relic()
        self.nebula_astroid = NebulaAstroid(self.horizont)
        self.p1pos = Unitpos(self.horizont)
        self.p2pos = Unitpos(self.horizont)
        self.relics = Relics(self.horizont, self.team_id)
    
    def learnuniverse(self):
        


        #Update points
        self.thiscore = self.obsQueue.Last(['team_points', self.team_id])[0] - self.totalscore
        self.totalscore += self.thiscore

        #self.nebula.learn()
        #self.p1pos.learn(self.obsQueue.Last(['units','position',self.team_id]))
        #self.p2pos.learn(self.obsQueue.Last(['units','position',self.opp_team_id]))

        #self.nebula_astroid.learn()
        self.relics.learn(
            self.obsQueue.Last(['match_steps']),                    #Current time. For debugging.
            self.obsQueue.Last(['relic_nodes']),                    #Position of (visible) relic nodes
            self.obsQueue.Last(['relic_nodes_mask']),               #List (bool) of nodes visible or not
            self.thiscore,                                          #How many points we scored               
            self.obsQueue.Last(['units', 'position', self.team_id]) #The whereabouts of our mighty fleet
        )

        #predict parameters here
        pass
        #relic.learn()    
        #ToDo:
        #flax.serialization.from_state_dict(obs, raw_state_dict)
            #-> Lage noen demo obs for å teste koden vår

        #Nebula - Kan lese state direkte -> env.step(actioen=Empty) vil gi oss neste HORIZON states
        #Astriod    -||-
        #Observable Tile - Lese fra OBS - Bare stacke state, ikke HORIZON
        #Relic      -||-
        #Energy     Hva er void? Vi må uansett holde styr på energy per tile
            #Holde styr på energy per tile
            #Holde styr på motstander units
            #Estimere effektiv energy per tile for s_t:t+H
        
        #
        #Energy per unit - id,

        #Egne units: ProbProp, men ta hensyn til astriods
        #Mostander units: ProbProp, men ta hensyn til astriods
        #ProbProp må ta Astriod som parameter

        #Class Agent:

        #U = Universe(....)
        #P = Policies(...)      #Policy for battle, transition,....
        #act():
            #s_t:t+H, OT, Relic, P1_pos, P1_nrg = u.predict(obs)

            #Mission control
            #Lage mission per unit

            #for hvert skip:
                #getAction(m,s_t:t+H,pos[unitId],energy[unitId])

            #return list of actions



    #s_{t:t+h} | o_{t}
    def learn_and_predict(self, observation:dict):

        #print("Remaining time (Overage) is", timeleft)
        state:State = State(observation,self.player)
        print(state.steps)
        self.obsQueue(observation)
        
        self.nebula_astroid.learn(state.nebulas,state.asteroids,state.observeable_tiles, current_step=state.steps)
        
        self.p1pos.learn(self.obsQueue.Last(['units','position',self.team_id]))

        #self.p2pos.learn(self.obsQueue.Last(['units','position',self.opp_team_id]))

        #timeleft:int = state.remainingOverageTime
        #Add observation to queue
        

        #Learn universe
        #self.learnuniverse()
        
        
        #Predict Nebula and Astroid here
        # nebula : list [current, pred_1, ... pred_horizon] 
        # astroid : list [current, pred_1, ... pred_horizon] 
        nebula, astroid = self.nebula_astroid.predict(state.nebulas,state.asteroids,
                                                      state.observeable_tiles, current_step=state.steps)

        #Ship positions        

        #Demo astroid field, while Morten finishes up        
        #astroidField = jnp.zeros((24,24))
        #astroidPredictions = [astroidField for i in range(self.horizont+1)]
        #Manually set a value for testing purposes
        #astroidPredictions[1] = astroidPredictions[1].at[2,3].set(.5)

        #Predict P1 positions
        p1pos = self.p1pos.predict(astroid[1:], debug=False)

        #Predict P2 positions
        #self.p2pos.predict(astroid[1:], debug=False)        

        #Predict Relic tiles
        # self.relics.predict()

        #Predict Energy
        #energy.precict(...)                
        
        #print("Done predicicting future")

        #Return...
        #return jnp.stack((s1,s2,s3),axis=2)
        return nebula, astroid, p1pos
        
        

#Test function for Jørgen
def jorgen():

    print("Running Jørgens tests")

    #Fix a seed for testing. 
    seed = 223344

    #Get initial observation
    step, player, obs, cfg, timeleft = getObservation(seed,0)
    
    #Create a fixed seed universe
    u = Universe(player,obs,cfg,horizont=3, seed=seed)

    for i in range(1,30):
        
        #Get another observation
        _, _, obs, _, timeleft = getObservation(seed,i)
        

        #Test universe prediction
        u.predict(obs, timeleft)



if __name__ == "__main__":
    pass