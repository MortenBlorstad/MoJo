from world.base_component import base_component
import jax
import jax.numpy as jnp

class Unitpos(base_component):

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.mapsize = (24,24)

    def getMap(self, obs):
        
        #Remove empty observations
        obs = obs[jnp.where((obs[:,0] != -1) & (obs[:,1] != -1))]        

        #Create indices of ship positions, using y,x
        indices = (obs[:,1],obs[:,0])        

        #Place ship at indices
        luxmap = jnp.zeros(self.mapsize).at[indices].add(1)
        
        #Return map
        return luxmap

    def learn(self, obs):
        
        print("---------------------------------------")
        print(self.getMap(jnp.array(obs[0])))
        print("---------------------------------------")
        
    def predict(self):
        pass