import wandb
import matplotlib.pyplot as plt
import os
import numpy as np
plot_dir = "MoJo/plots"
os.makedirs(plot_dir, exist_ok=True)

api = wandb.Api()

# Correct W&B path
run = api.run("Team-Mojo/baseline/gdvvj4ws")
''

# Fetch history
history = run.history()


def plot_loss(history, out_path):
    """Plot the loss curve over training steps."""
    smoothed_loss = history["loss"].ewm(alpha=0.99, ignore_na=True).mean()
    plt.figure(figsize=(2*3.4, 2*2.5), dpi=300)
    plt.plot(smoothed_loss, label="Loss")
    plt.xlabel("Training Step", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scores(history, out_path):
    """Plot player 0 and player 1 scores over match count."""
    plt.figure(figsize=(2*3.4, 2*2.5), dpi=300)
    match_count = history["total_match_count"]
    mask = np.isnan(match_count) 
    match_count = match_count[~mask]
    player_0_score = history["player_0_score"][~mask]
    player_1_score = history["player_1_score"][~mask]
    plt.plot(match_count, player_0_score, label="PPO agent")
    plt.plot(match_count, player_1_score, label="Rule-based agent")

    plt.xlabel("Matches", fontsize=10)
    plt.ylabel("Score", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


plot_loss(history, os.path.join(plot_dir, "baseline_loss_plot.png"))
plot_scores(history, os.path.join(plot_dir, "combined_player_scores.png"))