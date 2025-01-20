from base_component import base_component
import jax
import jax.numpy as jnp


class Nebula(base_component):

    def __init__(self, horizon):
        self.horizon = horizon
        self.map = jnp.zeros((1+horizon,24,24))
        self.nebula_tile_drift_speed = jnp.mean(self.nebula_tile_drift_speed_possibilities) # 0



        print("Nebula")


    def learn(self):
        pass


    def predict(self):
        pass
