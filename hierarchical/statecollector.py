import sys
import os
import sys
from datetime import datetime
import numpy as np
from tqdm import tqdm

if sys.stderr is None:
    sys.stderr = sys.__stderr__  # Reset stderr to the default
    print("🔥 Restored sys.stderr. Now catching the real error.")

# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from base_agent import Agent
from luxai_s3.wrappers import LuxAIS3GymEnv
from world.universe import Universe
from hierarchical.config import Config

num_games = 200
cfg = Config().Get("Director")
savePlayer = "player_0"

for game in tqdm(range(num_games)):
    
    env = LuxAIS3GymEnv( numpy_output = True)
    obs, info = env.reset()    

    agents = [
        Agent(player="player_0", env_cfg = info['params']),
        Agent(player="player_1", env_cfg = info['params'])
    ]

    u = Universe("player_0",info['params'], horizont=3)
    history = []
    done = False     
    
    while not done:
        step = obs["player_0"]["match_steps"]        
        actions = {}
        for i, agent in enumerate(agents):           
            action = agent.act(step, obs[f"player_{i}"])
            actions[agent.player] = action        

        #Save predictions, but skip first, empty universe
        if step != 0:
            history.append(u.predict(obs[savePlayer])["image"])

        obs, _, _, _, _ = env.step(actions)
        done = (step == 100)


    savedata = np.concat(history)    
    file = os.path.join(cfg["Worldmodel"]["datapath"], datetime.now().strftime("%d_%m_%Y_%H_%M_%S.npy"))
    #print("Saving data with shape",savedata.shape,"to file",file)
    with open(file, 'wb') as f:
        np.save(file, savedata)


env.close()