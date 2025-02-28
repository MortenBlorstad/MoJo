
import sys
import os
import json 
import sys

if sys.stderr is None:
    sys.stderr = sys.__stderr__  # Reset stderr to the default
    print("🔥 Restored sys.stderr. Now catching the real error.")



import numpy as np
import jax
import jax.numpy as jnp
import flax

# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from base_agent import Agent
from world.universe import Universe
from replay_memory import ReplayMemory, Transition


from agent.ppo_agent import PPOAgent
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode


# import subprocess

# # Define the command as a list of arguments
# command = [
#     "luxai-s3",  # The Lux AI executable/command
#     "MoJo/trainer/agent/main.py",  # Path to the first main.py script (your agent)
#     "MoJo/agents/main.py",  # Path to the second main.py script (opponent)
#     "-o", "replay.json"  # Output file for the replay
# ]

# # Run the command
# subprocess.run(command, check=True)






env = LuxAIS3GymEnv( numpy_output = True)

env = RecordEpisode(env, save_dir="MoJo/trainer/recorded_episodes")

terminated = {'player_0': jnp.array(False), 'player_1': jnp.array(False)}
print("============================================================================================")
has_continuous_action_space = True  # continuous action space; else discrete

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)

update_timestep = 50      # update policy every n timesteps

directory = "MoJo/trainer/weights/PPO_preTrained"
if not os.path.exists(directory):
          os.makedirs(directory)


checkpoint_path = os.path.join(directory, "PPO.pth")
print("save checkpoint path : " + checkpoint_path)

from datetime import datetime
# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

update_timestep = 10

print("============================================================================================")
running_return = 0
total_timesteps = 0

for episode in range(3):
    
    obs, info = env.reset()

    agents = [PPOAgent(player="player_0", env_cfg = info['params']),
    Agent(player="player_1", env_cfg = info['params'])]
    agents[0].train()
    if os.path.exists(checkpoint_path):
        agents[0].load(checkpoint_path)
        print("loaded model from : " + checkpoint_path)

    while True:
        total_timesteps += 1
        step = obs["player_0"]["match_steps"]
        print(f"Step {step} started")
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
        done = step >= 100
        if step > 1 and (step % update_timestep == 0 or done):
            loss = agents[0].learn()
            print(f"Step {step}: loss: {loss}: running return: {running_return/(total_timesteps)} score: player_0 {agents[0].universe.teampoints} player_1 {agents[0].universe.opponent_teampoints}")

       
        if done:
            print("episode done")
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            agents[0].save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            break

env.close()  








