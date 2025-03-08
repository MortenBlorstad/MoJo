import jax.numpy as jnp

class NaiveScalarEncoder():

    def __init__(self, env_params_ranges:dict):

        #Convert all lists to jnp arrays
        for key, value in env_params_ranges.items():
            env_params_ranges[key] = jnp.array(value)

        self.env_params_ranges = env_params_ranges

    def __lookup(self,key,value):
        return jnp.where(self.env_params_ranges[key] == value)[0].item()
    
    def Encode(self,**kwargs):
        indices = jnp.array([self.__lookup(key,value) for key, value in kwargs.items()])     
        return jnp.zeros((24,24)).at[(list(range(len(indices))),indices)].set(1)[jnp.newaxis,:]