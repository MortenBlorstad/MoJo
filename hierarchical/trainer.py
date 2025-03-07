
import sys
import os
import sys
import subprocess
import argparse

VM_IP = "158.39.201.251"  # Replace with your actual VM IP
VM_USER = "ubuntu"
SSH_KEY = "lux.pem"  # Make sure it's in the same directory or provide full path

LOCAL_EPISODE_PATH = "/episodes"  # Replace with your local episode destination
LOCAL_WEIGHTS_PATH = "/weights"  # Replace with your local weights destination
REMOTE_EPISODE_PATH = "~/Lux-Design-S3/MoJo/hierarchical/data/episodes"  # No trailing "/"
REMOTE_WEIGHTS_PATH = "~/Lux-Design-S3/MoJo/hierarchical/weights"  # No trailing "/"
os.makedirs(LOCAL_EPISODE_PATH, exist_ok=True)
os.makedirs(LOCAL_WEIGHTS_PATH, exist_ok=True)


# Argument parser to check if rsync should be used
parser = argparse.ArgumentParser(description="Train agent and optionally sync episodes and weights.")
parser.add_argument("--sync", action="store_true", help="Enable rsync file transfer after every 10 games.")
args = parser.parse_args()

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

num_games = 3

for game in range(num_games+1):
    game_step = 0
    match_number = 0
    seed = np.random.randint(0, 10000)
    print("Starting game", game + 1, "with seed", seed)

    obs, info = env.reset(seed=seed)

    agents = [
        Director(player="player_0", env_cfg = info['params'], training=True),
        Agent(player="player_1", env_cfg = info['params'])
    ]
    for match_idx in range(5):
        match_done = False
        match_steps = 0
        match_number += 1
        while not match_done and match_steps < 101:
            game_step += 1
            
            step = match_steps
            print("Game =", game, "match", match_number, " step =", step, " game step =", game_step)
            actions = {}
            for i, agent in enumerate(agents):           
                action = agent.act(step, obs[f"player_{i}"])
                actions[agent.player] = action        
            obs, reward, terminated, truncated, _ = env.step(actions)
            match_done = terminated["player_0"] or truncated["player_0"] or match_steps >= 101
            match_steps += 1
    if game > 0 and game % cfg['modelSaveFrequency'] == 0:
        agents[0].save()

    if args.sync and game > 0 and game % 10 == 0:
        print(f"📂 Copying episode_id.json after {game} games...")

        rsync_episode_command = [
            "rsync", "-avz", "-e", f"ssh -i {SSH_KEY}",
            f"{VM_USER}@{VM_IP}:{REMOTE_EPISODE_PATH}/episode_{game}.json",
            LOCAL_EPISODE_PATH
        ]
        # Rsync command for weights folder
        rsync_weights_command = [
            "rsync", "-avz", "-e", f"ssh -i {SSH_KEY}",
            f"{VM_USER}@{VM_IP}:{REMOTE_WEIGHTS_PATH}/",
            LOCAL_WEIGHTS_PATH  # Trailing "/" ensures directory sync
        ]

        # Execute the rsync commands
        try:
            subprocess.run(rsync_episode_command, check=True)
            print("✅ Episode JSON copied successfully.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error copying episode file: {e}")

        try:
            subprocess.run(rsync_weights_command, check=True)
            print("✅ Weights folder copied successfully.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error copying weights folder: {e}")

env.close()