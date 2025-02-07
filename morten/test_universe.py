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
    

n_steps = 50
player = "player_1"
env = LuxAIS3GymEnv( numpy_output = True) 
obs, info = env.reset(seed=seed)
config = {"max_units":16, "map_width":24, "map_height": 24}
obs = flax.serialization.to_state_dict(obs)
#print(obs.keys())
universe =  Universe(player,obs,config,3,seed)
if not os.path.exists(dump_name):
    episode = {}
    print(env.env_params.nebula_tile_drift_speed)
    #config = info["env_cfg"]
    actions = {
        "player_0": jnp.ones((16, 3), dtype=int),
        "player_1": jnp.ones((16, 3), dtype=int)
        }
    
    agent = Agent("player_1",config)
    for step in range(1,n_steps):
        actions[player] = agent.act(step,obs[player] )
        obs, _, _, _, info,state = env.step(actions)
        obs = flax.serialization.to_state_dict(obs)
        #state = State(flax.serialization.to_state_dict(info['final_state']), "player_1")
        state = flax.serialization.to_state_dict(info['final_state'])
        
        episode[step] = {}
        episode[step]["state"] = state
        episode[step]["obs"] = obs[player] 



    with open(dump_name, "w") as outfile:
        json.dump(to_json(episode), outfile)

with open(dump_name, "r") as infile:
    episode = from_json(json.load(infile))  # Use json.load() instead of json.loads()

test_dump = {}

for step in range(1,n_steps-4):
    obs = episode[str(step)]["obs"]
    state = State(obs,player)
    predictions = universe.learn_and_predict(obs)
    solutions = get_solution(episode,step)
    plot_state_comparison(step, solutions, predictions, state.observeable_tiles)
    if (step==18):
        test_dump["obs"] = obs
        test_dump["predictions"] = predictions
        test_dump["solutions"] = solutions
        with open(f"MoJo/morten/test_dump_{seed}.json", "w") as outfile:
            json.dump(to_json(test_dump), outfile)
        break


create_gif(seed)