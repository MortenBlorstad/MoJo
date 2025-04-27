"""
This script trains a Director-based hierarchical reinforcement learning agent to play Lux AI Season 3.

It initializes the environment, configures training from a YAML-based configuration system, and runs multiple games with 5 matches each.
Player 0 uses the Director agent with hierarchical planning and a learned world model, while Player 1 uses a simple rule-based baseline agent.
The script periodically logs match results and saves model checkpoints based on the configured frequency.
"""
import sys
import os
import sys
from hierarchical.config import Config
from base_agent import Agent
from hierarchical.director.director import Director
import numpy as np
import jax.numpy as jnp
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

if sys.stderr is None:
    sys.stderr = sys.__stderr__  # Reset stderr to the default
    print("🔥 Restored sys.stderr. Now catching the real error.")



# Get the absolute path of the MoJo directory
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first




cfg = Config().Get("Trainer")
env = LuxAIS3GymEnv( numpy_output = True)
if cfg['logepisodes']:
    env = RecordEpisode(env, save_dir=cfg['episodelogdir'])

num_games = 20
games_won = 0
total_match_count = 0 # counter for number of matches played
np.random.seed(0)
for game in range(num_games):
    game_step = 0
    match_number = 0
    seed = np.random.randint(0, 10000)
    player_0_wins = 0
    player_1_wins = 0

    print("Starting game", game + 1, "with seed", seed)

    obs, info = env.reset(seed=seed)

    agents = [
        Director(player="player_0", env_cfg = info['params'], training=True),
        Agent(player="player_1", env_cfg = info['params'])
    ]
    for match_idx in range(5):
        player_0_score = 0
        player_1_score = 0
        match_done = False
        match_steps = 0
        match_number += 1
        while not match_done and match_steps < 101:
            game_step += 1
            
            step = match_steps
            print("Game =", game, "match", match_number, " step =", step, " game step =",
                   game_step, f"score: player_0 {agents[0].u.totalscore} player_1 {agents[0].u.opponent_totalscore}")
            actions = {}
            for i, agent in enumerate(agents):           
                action = agent.act(step, obs[f"player_{i}"])
                actions[agent.player] = action        
            obs, reward, terminated, truncated, _ = env.step(actions)
            match_done = terminated["player_0"] or truncated["player_0"] or match_steps >= 101
            player_0_score, player_1_score = obs["player_0"]["team_points"]
                
            match_steps += 1

        total_match_count += 1
        agents[0].ww._WW_run.log({
            "total_match_count": total_match_count,
            "player_0_score": agents[0].u.teampoints,
            "player_1_score": agents[0].u.opponent_teampoints
        })

    if game > 0 and game % cfg['modelSaveFrequency'] == 0:
        agents[0].save()



env.close()
