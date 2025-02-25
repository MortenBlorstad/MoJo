
import sys
import os
import json 


import jax
import jax.numpy as jnp
import flax

# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from base_agent import Agent
from world.universe import Universe
from replay_memory import ReplayMemory, Transition



from luxai_s3.wrappers import LuxAIS3GymEnv

maxstep = 50
env = LuxAIS3GymEnv( numpy_output = True)

obs, info = env.reset()

universe = Universe(player="player_0", observation=obs, configuration = info['params'], horizont=3)

agents = [Agent(player="player_0", env_cfg = info['params']),
           Agent(player="player_1", env_cfg = info['params'])]

terminated = {'player_0': jnp.array(False), 'player_1': jnp.array(False)}
memory = ReplayMemory(1000)

while True:
    step = obs["player_0"]["match_steps"]
    print(f"Step {step} started")
    actions = {}
    for i, agent in enumerate(agents):
        if terminated[agent.player]:
            continue
        actions[agent.player] = agent.act(step, obs[f"player_{i}"])
    state = universe.predict(obs["player_0"])
    obs, reward, terminated, truncated, info = env.step(actions)

    next_state = universe.predict(obs["player_0"])
    
    transition0 = Transition(state, actions[agent.player], reward, next_state, terminated["player_0"])
    #transition1 = Transition(obs["player_1"], actions[agent.player], reward, obs["player_1"], terminated["player_1"])
    agent.learn()
    agent.push_memory(*transition0)
    #memory.push(*transition1)
    
    
    print(f"reward {reward} {universe.teampoints}, {universe.opponent_teampoints}, {universe.reward}")
    print(f"Step {step} completed")
    done = all(jnp.asarray(v) for v in terminated.values())
    if done or step >= maxstep:
        break

    






