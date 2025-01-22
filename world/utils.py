import json
from argparse import Namespace
import numpy as np
import jax.numpy as jnp


#Taken from kit
def from_json(state):
    if isinstance(state, list):
        return jnp.array(state)                                             #<-----------Change this to use numpy instead
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state 

#Agent main.py uses Namespace for parsing the json as observation. Let's do the same.
def getObsNamespace(file):   

    # Open and read the (semi) JSON file        
    with open(file, 'r') as file:
        inputs = file.read()
        
    #Parse
    raw_input = json.loads(inputs)

    #Return all data as Namespace
    return Namespace(**dict(step=raw_input["step"], obs=raw_input["obs"], remainingOverageTime=raw_input["remainingOverageTime"], player=raw_input["player"], info=raw_input["info"]))

#Path to replay files
def getPath(seed, obs):
    return "../MoJo/world/replay/{seed}/obs_{obs}".format(seed = seed, obs = obs)        

#Call this function to get an observation. Note that only first observations (step = 0) contains 'env_cfg'. Otherwise, empty dict is returned.
#'step' & 'player' is also contained in 'observation.obs' It is returned seperately to mimimc the behaviour of main.py:agent_fn(...) 
def getObservation(seed,step):
    observation = getObsNamespace(getPath(seed,step))    

    config = {}
    if(step == 0):
        observation.info['env_cfg']

    #Match parameters in Agent:main.py. We don't need all of this
    return observation.step, observation.player, from_json(observation.obs), config, observation.remainingOverageTime