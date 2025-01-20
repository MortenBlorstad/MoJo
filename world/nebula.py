from base_component import base_component
import jax
import jax.numpy as jnp


class Nebula(base_component):

    def __init__(self, horizon):
        self.horizon = horizon
        self.map = jnp.zeros((1+horizon,24,24))
        self.drift = None
        print("Nebula")


    def learn(self):
        pass


    def predict(self):
        pass
