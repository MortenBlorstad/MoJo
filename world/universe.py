from luxai_s3.wrappers import LuxAIS3GymEnv
import jax.numpy as jnp
import json
from utils import getObsDict
from obsqueue import ObservationQueue
from abc import ABC, abstractmethod

from nebula import Nebula


class Universe():

    def __init__(self, initialObservation, horizont = 3, seed = None):
      
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
        #   Example:
        #       print(initialObservation['info']['env_cfg']['unit_sap_cost'])
        #-------------------------------------------------------------------------------------------


        #Number of 'future universes' we predict
        self.horizont = horizont

        #History of observations
        self.obsQueue = ObservationQueue(10)
        
        #Add initial observation to queue
        self.obsQueue(initialObservation['obs'])

        #Determine players
        self.player = initialObservation['player']
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0         

        # self.relic = Relic()
        self.nebula = Nebula(self.horizont)
    
    def learnuniverse(self):

        #self.nebula.learn()

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
    def predict(self, observation):

        #Add observation to queue
        self.obsQueue(observation['obs'])

        #Learn universe
        self.learnuniverse()

        #For now, let's use the Env
        #R relic.precict() (R_1,R_2, R_3)
        #A astroid.precict() (A_1,A_2, A_3)

        print("Nebula")
        print(self.nebula.predict())
        print('')
        print("Done predicicting future")

        #Return...
        #return jnp.stack((s1,s2,s3),axis=2)
        
        
    #Step through all actions included in dump and assert that the resulting energy map matches the one we loaded from file
    def testuniverse(self, dumpfile):

        #Get data from dumpfile
        def loaddata(dump):
            # Open and read the JSON file        
            with open(dump, 'r') as file:
                return json.load(file)
            
        #Get actual actions taken at time step t as (jax) dictionary
        def getjaxtions(t):
            return {
            "player_0": jnp.array(data['actions'][t]['player_0']),
            "player_1": jnp.array(data['actions'][t]['player_1'])
            }
        
        #Load data 
        data = loaddata(dumpfile)
        
        #Initiate the gym
        env = LuxAIS3GymEnv(numpy_output=True)                  #Are we using torch? Supported? Maybe stick to jax...
        obs, info = env.reset(seed=data['metadata']['seed'])    #Start with seed from dump

        for t in range(len(data['actions'])):
            env.step(getjaxtions(t))


        assert jnp.all(env.state.map_features.energy == jnp.array(data['observations'][-1]['map_features']['energy']))
        print("Stepped through everything....")


if __name__ == "__main__":

    # #Just checking that everything is working as expected
    # u = Universe()
    # u.testuniverse('./../MoJo/world/seed54321.json')
  
    #Use the example observations from the provided kit
    firstObs = getObsDict('./../MoJo/world/sample_step_0_input.txt')
    secondObs = getObsDict('./../MoJo/world/sample_step_input.txt')

    #Create a fixed seed universe
    u = Universe(firstObs, seed=12345)

    #Test universe prediction
    u.predict(secondObs)

