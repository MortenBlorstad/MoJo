import os
import numpy as np
import jax.numpy as jnp
from itertools import combinations, permutations
from collections import defaultdict
from luxai_s3.wrappers import LuxAIS3GymEnv
from baseline.agent.ppo_agent import PPOAgent
from hierarchical.base_agent import Agent
from hierarchical.director.director import Director


import yaml
with open('MoJo/baseline/config.yaml', 'r') as f:
    baseline_config = yaml.safe_load(f)
directory = "MoJo/baseline/weights/"
checkpoint_path = os.path.join(directory, "PPO.pth")


checkpoint_path = os.path.join(directory, "PPO.pth")

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_winner, rating_loser, k=32):
    expected_winner = expected_score(rating_winner, rating_loser)
    expected_loser = expected_score(rating_loser, rating_winner)

    new_winner = rating_winner + k * (1 - expected_winner)
    new_loser = rating_loser + k * (0 - expected_loser)

    return new_winner, new_loser

env = LuxAIS3GymEnv(numpy_output=True)

# Create agent factory
agent_pool = {
    "PPOAgent": lambda name, cfg: PPOAgent(player=name, env_cfg=cfg, config=baseline_config),
    "Agent": lambda name, cfg: Agent(player=name, env_cfg=cfg),
    "Director": lambda name, cfg: Director(player=name, env_cfg=cfg, training=False),
}
agent_names = list(agent_pool.keys())

elo_scores = {name: 1000 for name in agent_pool}

# Store match outcomes
results = defaultdict(lambda: {"wins": 0, "losses": 0})

num_games_per_pair = 1

for a1, a2 in combinations(agent_names, 2):
    for i in range(num_games_per_pair):
        for player0_name, player1_name in [(a1, a2)]:
            print(f"Running match {i+1} between {player0_name} and {player1_name}...")
            player_0_score = 0
            player_1_score = 0
            obs, info = env.reset(seed=10000+i)
                # Create agents fresh per match
            agent0 = agent_pool[player0_name]("player_0", info['params'])
            agent1 = agent_pool[player1_name]("player_1", info['params'])
            agents = [agent0, agent1]
            if player0_name == "PPOAgent":
                agents[0].train()
                agents[0].load(checkpoint_path)
            if player_1_score == "PPOAgent":
                agents[0].train()
                agents[0].load(checkpoint_path)
            
            done = False
            step = obs["player_0"]["match_steps"]
            total_timesteps = 0
            while not done:
                total_timesteps += 1
                actions = {
                    agent.player: agent.act(step, obs[agent.player])
                    for agent in agents
                }
                obs, reward, terminated, truncated, _ = env.step(actions)
                done = total_timesteps >= 505
                step += 1

            team_wins = obs["player_0"]["team_wins"]
            p0_wins = team_wins[0]
            p1_wins = team_wins[1]
            print(f"Game {i+1} - {player0_name} vs {player1_name}: Player 0 wins: {p0_wins}, Player 1 wins: {p1_wins}")
            if p0_wins > 2:
                winner, loser = player0_name, player1_name
            else: 
                winner, loser = player1_name, player0_name
          


            results[(winner, loser)]["wins"] += 1
            results[(loser, winner)]["losses"] += 1
            elo_scores[winner], elo_scores[loser] = update_elo(
                    elo_scores[winner], elo_scores[loser]
                )
env.close()

# --- Report Results ---
print("\n--- Tournament Results ---")
for pair, record in results.items():
    print(f"{pair[0]} vs {pair[1]}: {record['wins']} wins")

print("\n--- ELO Ratings ---")
for agent, score in sorted(elo_scores.items(), key=lambda x: -x[1]):
    print(f"{agent}: {round(score)}")