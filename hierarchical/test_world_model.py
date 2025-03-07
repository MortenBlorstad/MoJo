# Example usage
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from pathlib import Path
import wandb
# Initialize wandb
wandb.init(project="world_model")


config_path = Path(sys.argv[0]).parent / "world_model" / "config.yml"
config = yaml.safe_load(config_path.read_text())["world_model"]
print(config)
import torch


from base_agent import Agent
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode


#env = RecordEpisode(env, save_dir="MoJo/trainer/recorded_episodes")

from world.universe import Universe
from world_model.world_model import WorldModel

num_games = 100
world_model = WorldModel(config)
model_path = Path(sys.argv[0]).parent / "world_model" / "world_model.pt"
if model_path.exists():
    world_model.load_model(model_path)

env = LuxAIS3GymEnv(numpy_output=True)
player = "player_0"
for game in range(num_games):
    
    obs, info = env.reset(seed=game)
    universe = Universe(player, info['params'], horizont=3)
    agents = [
        Agent(player=player, env_cfg = info['params']),
        Agent(player="player_1", env_cfg = info['params'])
    ]
    done = False

    game_step = 0
    while not done:
        game_step +=1
        step = obs[player]["match_steps"]
        state = universe.predict(obs[player])
        print("Game =", game, " step =", step, " game step =", game_step)
        
        actions = {}
        for i, agent in enumerate(agents):
            action = agent.act(step, obs[f"player_{i}"])
            actions[agent.player] = action   
             
        obs, reward, terminated, truncated, info = env.step(actions)
        done = game_step  >= 505   
        reward = universe.reward
     

        is_first = step == 1
        
        #assert reward.shape == (16,), f"shape={reward.shape}"
        
        metrics = world_model.train(step, state, actions[player], reward, is_first, done)
        if step % 10 == 0 and step > 0:
            for key, value in metrics.items():
                print(f"{key} = {value}")
            wandb.log(metrics)    

        if step % 50 == 0 and step > 16:
            world_model.save_model(model_path) 
            print("Model saved after step:", step)
    
env.close()
