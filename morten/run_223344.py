
import sys
import os
import json 
import numpy as np

# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from world.utils import getObservation, getObsNamespace, getPath, from_json
import jax.numpy as jnp
import flax
from morten.utils.plotting import plot_state_with_predictions,create_gif
from agents.lux.kit import to_json,from_json
from world.universe import Universe

from world.obs_to_state import State

#Fix a seed for testing. 
seed = 223344
#Get initial observation
step, player, obs, cfg, timeleft = getObservation(seed,0)

#Create a fixed seed universe
universe =  Universe(player,obs,cfg,3,seed)

astroid_predictions= {}

for step in range(1,30):
    
    #Get another observation
    _, _, obs, _, timeleft = getObservation(seed,step)
    state = State(obs,player)
    predictions = universe.learn_and_predict(obs)


    # astroids = [a.T for a in predictions[1]]
    # astroid_predictions[step] = astroids


    plot_state_with_predictions(step, predictions, state.observeable_tiles)

# with open(f"MoJo/morten/astroid_predictions_{seed}.json", "w") as outfile:
#             json.dump(to_json(astroid_predictions), outfile)

#create_gif(f"{seed}_test")

# with open(f"MoJo/morten/astroid_predictions_{seed}.json", "r") as infile:
#     predictions = from_json(json.load(infile))  # Use json.load() instead of json.loads()

# print(predictions["7"][-1])
