from luxai_s3.wrappers import LuxAIS3GymEnv
import jax.numpy as jnp
import json

class Universe:
    def __init__(self, horizont = 3, seed = None):

        #Initiate the gym
        env = LuxAIS3GymEnv(numpy_output=False)  #Are we using torch? Supported? Maybe stick to jax...
        obs, info = env.reset(seed=seed)        #Start with seed from dump

        

        #Number of 'future universes' we predict
        self.horizont = horizont

        #History of observations
        self.obsQueueLen = 10   #Picked arbitrary. Must be long enough to entail nebula/asteroid movement
        self.obsQueue = []
    
    #Add to observation queue
    def enqueue(self, observation):
        self.obsQueue.append(observation)
        if(len(self.obsQueue) > self.obsQueueLen):
            self.obsQueue.pop()

    def learnuniverse(self):
        #predict parameters here
        pass


    #s_{t:t+h} | o_{t}
    def predict(self, observation):

        #This will be needed later, when we actually predict stuff
        self.enqueue(observation)

        #Learn universe
        self.learnuniverse()

        #

        #For now, let's use the Env


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
            obs, reward, terminated, truncated, info = env.step(getjaxtions(t))
            if(t == 50):
                print(obs)
                print('-------------------------------')
                print(reward)

        assert jnp.all(env.state.map_features.energy == jnp.array(data['observations'][-1]['map_features']['energy']))
        print("Stepped through everything....")

#Just checking that everything is working as expected
#u = Universe()
#u.testuniverse('./../MoJo/world/seed54321.json')

#Create a fixed seed universe
#u = Universe(seed=12345)
import numpy as np
def from_json(state):
    if isinstance(state, list):
        return np.array(state)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state 

stp_0 = './../MoJo/world/sample_step_0_input.txt'
stp_2 = './../MoJo/world/sample_step_input.txt'
with open(stp_2, 'r') as file:
    content = file.read()

obs = from_json(content)
print(obs)


#print(u.data['params'])
#print(u.env.state.energy_node_fns)
