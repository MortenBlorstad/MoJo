from base_component import base_component
import jax
import jax.numpy as jnp
import numpy as np


class Nebula(base_component):

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.map = jnp.zeros((1+horizon,24,24))
        self.nebula_tile_drift_speed = np.mean(self.env_params_ranges["nebula_tile_drift_speed"],dtype=np.float16) # 0
        
        print("Nebula", self.nebula_tile_drift_speed)


    def learn(self):
        pass


    def predict(self):
        pass
