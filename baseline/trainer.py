"""
This script trains a baseline PPO agent (PPOAgent) to play Lux AI Season 3.

It initializes the environment, loads configurations, manages training loops, and periodically saves models and logs metrics via Weights & Biases (wandb).
The opponent agent uses a rule-based baseline (Agent class).
The script supports resuming from checkpoints and records overall match performance during training.
"""

import sys
import os
import sys
import numpy as np
import jax.numpy as jnp
import yaml
from base_agent import Agent
import wandb
from agent.ppo_agent import PPOAgent
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
if sys.stderr is None:
    sys.stderr = sys.__stderr__  # Reset stderr to the default
    print("🔥 Restored sys.stderr. Now catching the real error.")




# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first



with open('MoJo/baseline/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


env = LuxAIS3GymEnv( numpy_output = True)

#env = RecordEpisode(env, save_dir="MoJo/baseline/recorded_episodes")

terminated = {'player_0': jnp.array(False), 'player_1': jnp.array(False)}
print("============================================================================================")


update_timestep = config['update_timestep']      # update policy every n timesteps

directory = "MoJo/baseline/weights/"
if not os.path.exists(directory):
          os.makedirs(directory)


checkpoint_path = os.path.join(directory, "PPO.pth")
print("save checkpoint path : " + checkpoint_path)

from datetime import datetime
# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)


# Initialize wandb run
wandb.init(
    project="baseline",  # change this to your project name
    name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    config=config,
)


print("============================================================================================")
num_games = 20
update_step = 0 # counter for number of training step for ppo agent
total_match_count = 0 # counter for number of matches played
np.random.seed(0)
for episode in range(num_games):
    running_return = 0
    total_timesteps = 0

    obs, info = env.reset(seed = np.random.randint(0, 10000))

    agents = [
                PPOAgent(player="player_0", env_cfg = info['params'], config=config),
                Agent(player="player_1", env_cfg = info['params'] )
            ]
    agents[0].train()
    if os.path.exists(checkpoint_path):
        agents[0].load(checkpoint_path)
        print("loaded model from : " + checkpoint_path)
    
    match_count = 0
    player_0_score = 0
    player_1_score = 0
    
    while True:
        total_timesteps += 1
        
        step = obs["player_0"]["match_steps"]
        print(f"Step {step} total_timesteps {total_timesteps} started")
        actions = {}
        for i, agent in enumerate(agents):
            if terminated[agent.player]:
                continue
            action = agent.act(step, obs[f"player_{i}"])
            actions[agent.player] = action
                   
        # actions["player_0"] = actions["player_1"]
        obs, reward, terminated, truncated, info = env.step(actions)
        running_return += np.mean(agents[0].universe.reward)
        agents[0].append_to_buffer(terminated[agent.player])   
        done = total_timesteps >= 505
        if step > 1 and (step % update_timestep == 0 or done):
            update_step += 1
            loss = agents[0].learn()
            print(f"Step {step}: loss: {loss}: running return: {running_return/(total_timesteps)} score: player_0 {agents[0].universe.teampoints} player_1 {agents[0].universe.opponent_teampoints}")
            # Log to wandb
            
            wandb.log({
                "step": step,
                "update_step": update_step,
                "loss": loss,
                "running_return": running_return / total_timesteps,
            })


        if step == 100 or step ==101:
            print("match done", step, total_timesteps)
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            agents[0].save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            if agents[0].universe.teampoints > agents[0].universe.opponent_teampoints:
                player_0_score += 1
            elif agents[0].universe.opponent_teampoints > agents[0].universe.teampoints:
                player_1_score += 1
            match_count += 1
            total_match_count += 1

            wandb.log({
                "total_match_count": total_match_count,
                "player_0_score": agents[0].universe.teampoints,
                "player_1_score": agents[0].universe.opponent_teampoints
            })
    
        if done:
        
            print("Game done") 
            print(f"Game {match_count}/{5}: score: player_0 {player_0_score} - {player_1_score} player_1")
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            agents[0].save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            break   
env.close()  








