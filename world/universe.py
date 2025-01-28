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
from world.nebula import Nebula
from world.unitpos import Unitpos

class Universe():

    def __init__(self, player, observation, configuration, horizont = 3, seed = None):
      
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

        # self.relic = Relic()
        self.nebula = Nebula(self.horizont)
        self.p1pos = Unitpos(self.horizont)
        self.p2pos = Unitpos(self.horizont)
    
    def learnuniverse(self):

        #self.nebula.learn()
        self.p1pos.learn(self.obsQueue.Last(['units','position',0]))
        self.p2pos.learn(self.obsQueue.Last(['units','position',1]))

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
    def predict(self, observation, timeleft):

        print("Remaining time (Overage) is", timeleft)

        #Add observation to queue
        self.obsQueue(observation)

        #Learn universe
        self.learnuniverse()

        #Predict Nebula here
        #nebula = .... 

        #Predict Astroid here
        #astroid = .... 

        #Ship positions        

        #Demo astroid field, while Morten finishes up        
        astroidField = jnp.zeros((24,24))
        astroidPredictions = [astroidField for i in range(self.horizont+1)]
        #Manually set a value for testing purposes
        astroidPredictions[1] = astroidPredictions[1].at[2,3].set(.5)

        #Predict P1 positions
        self.p1pos.predict(astroidPredictions, debug=False)

        #Predict P2 positions
        self.p2pos.predict(astroidPredictions, debug=False)        

        #Predict Relic tiles
        #relic.precict(...)

        #Predict Energy
        #energy.precict(...)                
        
        print("Done predicicting future")

        #Return...
        #return jnp.stack((s1,s2,s3),axis=2)
        
        

#Test function for Jørgen
def jorgen():

    print("Running Jørgens tests")

    #Fix a seed for testing. 
    seed = 223344

    #Get initial observation
    step, player, obs, cfg, timeleft = getObservation(seed,0)
    
    #Create a fixed seed universe
    u = Universe(player,obs,cfg,horizont=3, seed=seed)

    #Get another observation
    _, _, obs, _, timeleft = getObservation(seed,27)

    #Test universe prediction
    u.predict(obs, timeleft)

#Test function for Morten
def morten():

    print("Running Mortens tests")

    seed = 223344
    #u = Universe(player="player_0")
    #env = LuxAIS3GymEnv(numpy_output=True)                  #Are we using torch? Supported? Maybe stick to jax...
    #obs, info = env.reset(seed=data['metadata']['seed'])    #Start with seed from dump
    step, player, obs, cfg, timeleft = getObservation(seed,0)
    print(player)
    nebula = Nebula(horizon=3)
    #observations = jnp.zeros((20,24,24))
    for t in range(1,42):
        step, player, obs, cfg, timeleft = getObservation(seed,t)
        state = State(obs, "player_0")
        nebulas = jnp.array(state.nebulas.copy())
        observable = jnp.array(state.observeable_tiles.copy())
        #print(t, player,"\n", nebulas, "\n\n" )
        
        nebula.learn(nebulas,observable, t-1)
        
        #observations = observations.at[t-1].set(nebulas)
    
    # print(observations[0])
    # print(observations[1])
    # print(observations[2])
    
    
    #print(flax.serialization.to_state_dict(env.state))
    step, player, obs, cfg, timeleft = getObservation(seed,1)
    
    state = State(obs, "player_0")
    # print("player_units_count")
    # print(state.player_units_count)

    # print("\n nebulas")
    # print(state.nebulas)


def test():
    import numpy as np
    def check_prediction_accuracy(correct_nebulas, prediction):
        # Check if all values where prediction == 1 are also 1 in correct_nebulas
        condition_1 = jnp.all(correct_nebulas[prediction == 1] == 1)

        # Check if all values where correct_nebulas == 0 are also 0 in prediction
        condition_2 = jnp.all(prediction[correct_nebulas == 0] == 0)

        # Return True if both conditions are met, otherwise False
        return condition_1 & condition_2
    env = LuxAIS3GymEnvCheat() 
    seed = 223344
    obs, info = env.reset(seed=seed)
    actions = {
            "player_0": jnp.zeros((16, 3), dtype=int),
            "player_1": jnp.zeros((16, 3), dtype=int)
            }
    #env.env_params["nebula_tile_drift_speed"] = -0.05
    nebula = Nebula(3)
    print(env.env_params)
    for step in range(1,22):
        _, _, _, _, _, state = env.step(actions)
        _, _, obs, _, _ = getObservation(seed,step)
        
        my_state = State(obs, "player_1")
        nebulas = jnp.array(my_state.nebulas.copy())
        observable = jnp.array(my_state.observeable_tiles.copy())
        nebula.learn(nebulas,observable,step-1)
        prediction = nebula.predict(nebulas,step)

        state = flax.serialization.to_state_dict(state)
        correct_state = State(state, "player_1")
        correct_nebulas = jnp.array(correct_state.nebulas.copy())

        print(step -1, check_prediction_accuracy(correct_nebulas, prediction))
        if not check_prediction_accuracy(correct_nebulas, prediction):
            print(correct_nebulas.T)
            print(prediction.T)
            break
    


if __name__ == "__main__":

    # #Branch out to avoid any more GIT HASSLE
    # if socket.gethostname() == "MSI":
    #     jorgen()
    # else:
    
    #morten()
    #test()
    jorgen()
