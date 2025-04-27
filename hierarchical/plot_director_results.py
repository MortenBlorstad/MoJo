"""
This script fetches training metrics from a Weights & Biases (wandb) run for the Director agent and its world model, and generates detailed loss and performance plots.

It retrieves various losses related to model learning (e.g., dynamics, prediction, similarity, latent reconstruction) and plots both world model and Director policy losses.
Additionally, it tracks and visualizes match scores between the Director agent and a rule-based opponent, saving all plots.
"""

import wandb
import matplotlib.pyplot as plt
import os
import numpy as np
plot_dir = "MoJo/plots"
os.makedirs(plot_dir, exist_ok=True)

api = wandb.Api()

# Correct W&B path
run = api.run("Team-Mojo/Complete MoJo/m344e4h1")


# Fetch history
history = run.history(samples=250000)


def plot_loss(history, loss_name, out_path):
    """Plot the loss curve over training steps."""
    smoothed_loss = history[loss_name].dropna() # .ewm(alpha=0.99, ignore_na=True).mean()
    plt.figure(figsize=(2*3.4, 2*2.5), dpi=300)
    plt.plot(smoothed_loss, label="Loss")
    plt.xlabel("Training Step", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_dynamics_loss(history, out_path):
    """Plot the loss curve over training steps."""
    smoothed_loss = history["dyn_loss"].dropna() + history["rep_loss"].dropna() # .ewm(alpha=0.99, ignore_na=True).mean()
    plt.figure(figsize=(2*3.4, 2*2.5), dpi=300)
    plt.plot(smoothed_loss, label="Loss")
    plt.xlabel("Training Step", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_prediction_loss(history, out_path):
    """Plot the loss curve over training steps."""
    smoothed_loss = history["reward_loss"].dropna() + history["cont_loss"].dropna() # .ewm(alpha=0.99, ignore_na=True).mean()
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
    match_count = history["total_match_count"].to_numpy()
    mask = np.isnan(match_count) 
    match_count = match_count[~mask]
    player_0_score = history["player_0_score"].to_numpy()[~mask]
    player_1_score = history["player_1_score"].to_numpy()[~mask]
    plt.plot(match_count, player_0_score, label="Director agent")
    plt.plot(match_count, player_1_score, label="Rule-based agent")

    plt.xlabel("Matches", fontsize=10)
    plt.ylabel("Score", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

plot_loss(history,"model_loss", os.path.join(plot_dir, "world_model_total_loss_plot.png"))
plot_dynamics_loss(history, os.path.join(plot_dir, "world_model_dyn_loss_plot.png"))
plot_prediction_loss(history, os.path.join(plot_dir, "world_model_pred_loss_plot.png"))
plot_loss(history,"simsr_loss", os.path.join(plot_dir, "world_model_sim_loss_plot.png"))
plot_loss(history,"mbr_loss", os.path.join(plot_dir, "world_model_latent_recon_loss_plot.png"))


plot_loss(history,"mgrloss", os.path.join(plot_dir, "director_mgrloss_plot.png"))
plot_loss(history,"wrkloss", os.path.join(plot_dir, "director_wrkloss_plot.png"))
plot_loss(history,"goalloss", os.path.join(plot_dir, "director_goalloss_plot.png"))
plot_scores(history, os.path.join(plot_dir, "director_combined_player_scores.png"))