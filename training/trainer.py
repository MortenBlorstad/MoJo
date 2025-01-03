
import sys
import os

# Add the parent directory (Lux-Design-S3 level) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from luxai_s3.wrappers import LuxAIS3GymEnv
from MoJo.agents.agent import Agent


env = LuxAIS3GymEnv(numpy_output=True)
obs, info = env.reset(seed=0)

env_cfg = info["params"]  

player_0 = Agent("player_0", info["params"])
player_1 = Agent("player_1", info["params"])


obs, info = env.reset()
game_done = False
step = 0

for i in range(10): # change to if while not game_done:
    actions = {}
    for agent in [player_0, player_1]:
        actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

    obs, rewards ,terminated, truncated, info = env.step(actions)
    dones = {k: terminated[k] | truncated[k] for k in terminated}

    rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id]
            }
    
    
    #TODO Add experience replay. Add experience to the replay memory

    # Add Learning from experiences
    # player_0.learn()
    # player_1.learn()
    print(f"step {step}: rewards {rewards}")
    step += 1

env.close()