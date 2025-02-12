import sys
import os
import json 
import numpy as np
# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from agents.agent import Agent
from agents.lux.kit import to_json,from_json

from morten.utils.plotting import plot_state_comparison,create_gif
from world.utils import getObservation, getObsNamespace, getPath, from_json
from world.energy import Energy

import jax.numpy as jnp
import flax

from world.universe import Universe

from world.obs_to_state import State

seed = 123 #223344 #2#
dump_name = f"MoJo/morten/episode_{seed}.json"



def plot_energy(current_step,energy):
    import matplotlib.pyplot as plt
    plots_dir = "MoJo/morten/plots"
    os.makedirs(plots_dir, exist_ok=True)
    fig = plt.figure( figsize=(12, 6))
    plt.imshow(energy, cmap="gray")
    plt.tight_layout()
    plot_filename = os.path.join(plots_dir, f"plot_{current_step}.png")
    plt.savefig(plot_filename)
    plt.close(fig) 

def plot_predictions(current_step,predictions):
    import matplotlib.pyplot as plt
    plots_dir = "MoJo/morten/plots"
    os.makedirs(plots_dir, exist_ok=True)
    fig,axs = plt.subplots(ncols=len(predictions), figsize=(12, 6))
    for i,p in enumerate(predictions):

        axs[i].imshow(p, cmap="gray")
        axs[i].set_title(f"step {current_step+(i)}")
    
    plot_filename = os.path.join(plots_dir, f"plot_{current_step}.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig) 

#create_gif(f"energy_{seed}")
n_steps = 74

#print(obs.keys())


step, player, obs, cfg, timeleft = getObservation(seed,0)

universe =  Universe(player,obs,cfg,3,seed)
energy = Energy(3)
for step in range(1,n_steps+1):
    
    _, _, obs, _, timeleft = getObservation(seed,step)
    state = State(obs,player)
    #print("energy", state.energy.T)
    energy.learn(step,state.energy,state.player_units_count,state.opponent_units_count, state.observeable_tiles)
    print(step, energy.energy_node_drift_speed)
    predictions = energy.predict(step)
    plot_predictions(step,predictions)


create_gif(f"energy_{seed}")


