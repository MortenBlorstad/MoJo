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

from luxai_s3.wrappers import LuxAIS3GymEnv

import jax.numpy as jnp
import flax

from world.universe import Universe

from world.obs_to_state import State

seed = 223344 #223344 #2#
dump_name = f"MoJo/morten/episode_{seed}.json"


def extract_solution(episode, step):
    full_state = State(episode[str(step)]["state"], "player_1")
    solution = jnp.stack(
                            [full_state.nebulas,
                            full_state.asteroids,
                            jnp.clip(full_state.player_units_count, a_min =0, a_max = 1)
                            ],
                        axis=-1)
    return solution

def get_solution(episode, step):
    solutions = [extract_solution(episode,step+i) for i in range(4)]
    return solutions

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


n_steps = 50
player = "player_1"
env = LuxAIS3GymEnv( numpy_output = True) 
obs, info = env.reset(seed=seed)
config = {"max_units":16, "map_width":24, "map_height": 24}

#print(obs.keys())
universe =  Universe(player,obs,config,3,seed)

with open(dump_name, "r") as infile:
    episode = from_json(json.load(infile))  # Use json.load() instead of json.loads()

test_dump = {}

for step in range(1,n_steps-4):
    obs = episode[str(step)]["obs"]
    state = State(obs,player)
    full_state = State(episode[str(step)]["state"], "player_1")
    plot_energy(step,full_state.energy)

