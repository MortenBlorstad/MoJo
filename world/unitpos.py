from base_component import base_component
import jax
import jax.numpy as jnp

class Unitpos(base_component):

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        #self.map = jnp.zeros((1+horizon,24,24))   
        # 

    def learn(self, obs):
        self.lo = obs
        
    def predict(self):
        print(self.lo)


