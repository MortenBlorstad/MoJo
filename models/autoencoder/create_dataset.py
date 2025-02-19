import sys
import os
import json 
import numpy as np

# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from world.utils import getObservation, getObsNamespace, getPath, from_json
import jax.numpy as jnp
import flax
from world.obs_to_state import State

max_unit_energy: int = 400
player_id = 0

dataset = []
print(max_unit_energy)
for seed in [123,125,126,127,128,129,130,131,132,133,134,135,223344 ]:
    step, player, obs, cfg, timeleft = getObservation(seed,0)
    for step in range(1,75):
        _, _, obs, _, timeleft = getObservation(seed,step)
        state = State(obs, player)
        dataset.append(jnp.array(state.player_sparse_energy_map))



player = "player_1"
from luxai_s3.wrappers import LuxAIS3GymEnv
from agents.agent import Agent
actions = {
        "player_0": jnp.ones((16, 3), dtype=int),
        "player_1": jnp.ones((16, 3), dtype=int)
        }
    
config = {"max_units":16, "map_width":24, "map_height": 24}
agent = Agent("player_1",config)
n_steps = 500
env = LuxAIS3GymEnv( numpy_output = True) 
for seed in range(20):
        obs, info = env.reset(seed=seed)
        obs = flax.serialization.to_state_dict(obs)
        for step in range(1,n_steps):
                actions[player] = agent.act(step,obs[player] )
                obs, _, _, _, info,state = env.step(actions)
                obs = flax.serialization.to_state_dict(obs)
                state = State(obs[player] , player)
                dataset.append(jnp.array(state.player_sparse_energy_map))
                
    
dataset = jnp.stack(dataset)
print(dataset.shape)
        #state = State(obs,player)

np.savez_compressed("MoJo/models/autoencoder/energy_map_dataset.npz", dataset)

