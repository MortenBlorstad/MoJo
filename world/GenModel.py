from luxai_s3.wrappers import LuxAIS3GymEnv
import jax.numpy as jnp
import json
from abc import ABC, abstractmethod


from nebula import Nebula


class Universe():

    def __init__(self, horizont = 3, seed = None):

        #Initiate the gym
        env = LuxAIS3GymEnv(numpy_output=False)     #Are we using torch? Supported? Maybe stick to jax...
        obs, info = env.reset(seed=seed)            #Start with seed from dump        

        #Number of 'future universes' we predict
        self.horizont = horizont

        #History of observations
        self.obsQueueLen = 10   #Picked arbitrary. Must be long enough to entail nebula/asteroid movement
        self.obsQueue = []
        # self.relic = Relic()
        nebula = Nebula(self.horizont)
    
    

    #Add to observation queue
    def enqueue(self, observation):
        self.obsQueue.append(observation)
        if(len(self.obsQueue) > self.obsQueueLen):
            self.obsQueue.pop()

    def learnuniverse(self):
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

        #This will be needed later, when we actually predict stuff
        self.enqueue(observation)

        #Learn universe
        self.learnuniverse()

        #For now, let's use the Env
        #R relic.precict() (R_1,R_2, R_3)
        #A astroid.precict() (A_1,A_2, A_3)

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

    #Create a fixed seed universe
    u = Universe(seed=12345)
    # import numpy as np

    # def getObs(path):

    #     def from_json(state):
    #         if isinstance(state, list):
    #             return np.array(state)
    #         elif isinstance(state, dict):
    #             out = {}
    #             for k in state:
    #                 out[k] = from_json(state[k])
    #             return out
    #         else:
    #             return state 

    #     with open(path, 'r') as file:
    #         content = file.read()
    #     return from_json(content)

        



    # stp_0 = './../MoJo/world/sample_step_0_input.txt'
    # stp_2 = './../MoJo/world/sample_step_input.txt'






    #print(u.data['params'])
    #print(u.env.state.energy_node_fns)


    # from luxai_s3.state import (
    #     ASTEROID_TILE,
    #     ENERGY_NODE_FNS,
    #     NEBULA_TILE,
    #     EnvObs,
    #     EnvState,
    #     MapTile,
    #     UnitState,
    #     gen_state
    # )


    # #obs = getObs(stp_2)

    # with open(stp_2, 'r') as file:
    #     raw_state_dict = file.read()

    # import flax

    # flax.serialization.from_state_dict(obs, raw_state_dict)