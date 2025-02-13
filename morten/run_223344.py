
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


def plot_predictions(current_step,predictions):
    import matplotlib.pyplot as plt
    plots_dir = "MoJo/morten/plots"
    os.makedirs(plots_dir, exist_ok=True)
    ncols, nrows = predictions.shape[:2] # time, features
    fig,axs = plt.subplots(ncols=ncols,nrows =nrows , figsize=(12, 3*nrows))
    for col,p in enumerate(predictions):
        axs[0,col].imshow(1 -p[0], cmap="gray",vmin = 0, vmax = 1)
        axs[0,col].set_title(f"nebula step {current_step+(col)}")

        axs[1,col].imshow(1- p[1], cmap="gray",vmin = 0, vmax = 1)
        axs[1,col].set_title(f"astroids step {current_step+(col)}")
        
        axs[2,col].imshow(1-p[2], cmap="gray",vmin = 0, vmax = 1)
        axs[2,col].set_title(f"unobserved_terrain step {current_step+(col)}")

        axs[3,col].imshow(1-p[3], cmap="gray", vmin = 0, vmax = 1)
        axs[3,col].set_title(f"pos1 step {current_step+(col)}")

        axs[4,col].imshow(1-p[4], cmap="gray")
        axs[4,col].set_title(f"energy step {current_step+(col)}")
    
    plot_filename = os.path.join(plots_dir, f"plot_{current_step}.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig) 


for step in range(1,75):
    
    #Get another observation
    _, _, obs, _, timeleft = getObservation(seed,step)
    state = State(obs,player)
    predictions = universe.learn_and_predict(obs)


    # astroids = [a.T for a in predictions[1]]
    # astroid_predictions[step] = astroids
    plot_predictions(step,predictions)

    #plot_state_with_predictions(step, predictions, state.observeable_tiles)

create_gif(f"predictions_{seed}")


# with open(f"MoJo/morten/astroid_predictions_{seed}.json", "w") as outfile:
#             json.dump(to_json(astroid_predictions), outfile)

#create_gif(f"{seed}_test")

# with open(f"MoJo/morten/astroid_predictions_{seed}.json", "r") as infile:
#     predictions = from_json(json.load(infile))  # Use json.load() instead of json.loads()

# print(predictions["7"][-1])
