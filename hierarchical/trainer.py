
import sys
import os
import sys

if sys.stderr is None:
    sys.stderr = sys.__stderr__  # Reset stderr to the default
    print("🔥 Restored sys.stderr. Now catching the real error.")

import numpy as np
import jax.numpy as jnp

# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from base_agent import Agent
from director import Director

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

env = LuxAIS3GymEnv( numpy_output = True)
#env = RecordEpisode(env, save_dir="MoJo/trainer/recorded_episodes")

terminated = {'player_0': jnp.array(False), 'player_1': jnp.array(False)}

num_games = 1

for game in range(num_games):
    
    obs, info = env.reset()

    agents = [
        Director(player="player_0", env_cfg = info['params']),
        Agent(player="player_1", env_cfg = info['params'])
    ]
    done = False

    while not done:
        step = obs["player_0"]["match_steps"]
        print("Game =",game," step =",step)
        
        actions = {}
        for i, agent in enumerate(agents):
            if terminated[agent.player]:
                continue
           
            action = agent.act(step, obs[f"player_{i}"])
            actions[agent.player] = action        
        obs, reward, terminated, truncated, info = env.step(actions)
        done = step == 100            

env.close()