import sys
import os
# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from agents.agent import Agent

from morten.utils.plotting import plot_state_comparison,create_gif

from luxai_s3.wrappers import LuxAIS3GymEnv

import jax.numpy as jnp
import flax

from world.universe import Universe

from world.obs_to_state import State

seed = 223344 #223344 #2#

env = LuxAIS3GymEnv() 
obs, info = env.reset(seed=seed)

#config = info["env_cfg"]
actions = {
        "player_0": jnp.ones((16, 3), dtype=int),
        "player_1": jnp.ones((16, 3), dtype=int)
        }
config = {"max_units":16, "map_width":24, "map_height": 24}

print(env.env_params.nebula_tile_drift_speed)
episode = {}
n_steps = 50
player = "player_1"
agent = Agent("player_1",config)
obs = flax.serialization.to_state_dict(obs)[player] 
#print(obs.keys())
universe =  Universe(player,obs,config,3,seed)

for step in range(1,n_steps):
    actions[player] = agent.act(step,obs)
    obs, _, _, _, info,state = env.step(actions)
    obs = flax.serialization.to_state_dict(obs)[player] 
    state = State(flax.serialization.to_state_dict(info['final_state']), "player_1")
    
    episode[step] = {}
    episode[step]["state"] = state
    episode[step]["obs"] = obs


for step in range(1,n_steps-4):
    obs = episode[step]["obs"]
    state = State(obs,player)
    predictions = universe.learn_and_predict(obs)

    solution = jnp.stack([episode[step]["state"].nebulas,episode[step]["state"].asteroids,
                           jnp.clip(episode[step]["state"].player_units_count, a_min =0, a_max = 1)],axis=-1)
    solutions = [jnp.stack([episode[step+i]["state"].nebulas,
                            episode[step+i]["state"].asteroids,
                             jnp.clip(episode[step+i]["state"].player_units_count, a_min =0, a_max = 1)],
                             axis=-1) for i in range(4)]
    
    plot_state_comparison(step, solutions, predictions, state.observeable_tiles)

create_gif(seed)