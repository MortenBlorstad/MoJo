
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

from hierarchical.config import Config
from base_agent import Agent
from director import Director

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

cfg = Config().Get("Trainer")
env = LuxAIS3GymEnv( numpy_output = True)
if cfg['logepisodes']:
    env = RecordEpisode(env, save_dir=cfg['episodelogdir'])

num_games = 1

for game in range(num_games):

    print("Starting game",game+1)
    
    obs, info = env.reset()

    agents = [
        Director(player="player_0", env_cfg = info['params'],training=True),
        Agent(player="player_1", env_cfg = info['params'])
    ]
    done = False

    while not done:
        step = obs["player_0"]["match_steps"]
        
        actions = {}
        for i, agent in enumerate(agents):           
            action = agent.act(step, obs[f"player_{i}"])
            actions[agent.player] = action        
        obs, reward, _, _, _ = env.step(actions)
        done = step == 100

    if game > 0 and game % cfg['modelSaveFrequency']:
        agents[0].save()
env.close()