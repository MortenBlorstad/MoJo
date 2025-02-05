
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import glob

import imageio.v2 as imageio
import multiprocessing


white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
light_red_deep_red_cmap = LinearSegmentedColormap.from_list("light_red_deep_red_cmap", ["#ffcccb", "red"])  # Deep red
def check_prediction_accuracy(correct_nebulas, prediction):
    # Identify false positives: places where prediction is 1 but correct state is 0
    false_positives = jnp.any((prediction == 1) & (correct_nebulas == 0))

    # Check if all values where correct_nebulas == 0 are also 0 in prediction
    # False Negatives (FN): predicted is 0, but correct is 1 (missed an obstacle)
    false_negatives = jnp.any((prediction == 0) & (correct_nebulas == 1))

    # Return True if both conditions are met, otherwise False
    if false_positives or false_negatives:
        return False
    
    return True

multiprocessing.set_start_method("spawn", force=True)
def pad_image(image, block_size=16):
    """Pads an image so its dimensions are divisible by block_size."""
    h, w = image.shape[:2]
    new_h = (h + block_size - 1) // block_size * block_size
    new_w = (w + block_size - 1) // block_size * block_size

    padded_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)
    padded_image[:h, :w] = image
    return padded_image

def plot_state_comparison(current_step, correct_states, predicted_states, observable):
    """
    Plots the correct states in the first row and predicted states in the second row.
    
    Args:
        correct_states (list of np.array): List of correct 24x24 grids.
        predicted_states (list of np.array): List of predicted 24x24 grids.
    """
    
    plots_dir = "MoJo/morten/plots"
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    observable = np.array(observable,dtype=float)
    observable[observable==1]=np.inf
    nebula_predictions = predicted_states[0]
    astroid_predictions = predicted_states[1]
    if len(predicted_states)==3:
        p1_predictions = predicted_states[2]


    for i in range(4):
        correct_state = np.array(correct_states[i],dtype=float)
    
        nebula_prediction = np.array(nebula_predictions[i],dtype=float)
        astroid_prediction = np.array(astroid_predictions[i],dtype=float)
        if len(predicted_states)==3:
            p1_prediction = np.array(p1_predictions[i],dtype=float)
            p1_prediction[p1_prediction == 0.0] = np.inf
        

                
        # Identify false positives: places where prediction is 1 but correct state is 0
        false_positive_indices_nebula = np.array(np.where((nebula_prediction == 1) & (correct_state[:,:,0] == 0)))
        false_negative_indices_nebula = np.array(np.where((nebula_prediction == 0) & (correct_state[:,:,0] == 1)))

        false_positive_indices_astroid = np.array(np.where((astroid_prediction == 1) & (correct_state[:,:,1] == 0)))
        false_negative_indices_astroid = np.array(np.where((astroid_prediction == 0) & (correct_state[:,:,1] == 1)))


        correct_state[correct_state == 0.0] = np.inf
        nebula_prediction[nebula_prediction != 1.0] = np.inf
        astroid_prediction[astroid_prediction == 0.0] = np.inf
        

        # Plot correct state (top row)
        axes[0, i].imshow(1-correct_state[:,:,0], aspect="auto")
        axes[0, i].imshow(1-correct_state[:,:,1], aspect="auto", cmap = "gray")
        axes[0, i].imshow(correct_state[:,:,2], aspect="auto", cmap = "autumn")
        axes[0, i].set_title(f"Correct - t={current_step+i}")
        #axes[0, i].axis("off")

        # Plot predicted state (bottom row)
        axes[1, i].imshow(1-nebula_prediction, aspect="auto")
        axes[1, i].imshow(1-astroid_prediction, aspect="auto", cmap="gray", vmin = 0, vmax=1)
        if len(predicted_states)==3:
            axes[1, i].imshow(p1_prediction.T, aspect="auto", cmap = light_red_deep_red_cmap, vmin=0, vmax=1)

        axes[1, i].imshow(observable/2, aspect="auto", cmap="gray", alpha=0.25, vmin = 0, vmax=1)
        axes[1, i].set_title(f"Predicted - t={current_step+i}")
        axes[1, i].scatter(false_positive_indices_nebula[1], false_positive_indices_nebula[0], color="red", marker="x", label="False Positive", s=20)
        axes[1, i].scatter(false_negative_indices_nebula[1], false_negative_indices_nebula[0], color="blue", marker="s", label="False Negative", s=20)
        axes[1, i].scatter(false_positive_indices_astroid[1], false_positive_indices_astroid[0], color="red", marker="x", label="False Positive", s=20)
        axes[1, i].scatter(false_negative_indices_astroid[1], false_negative_indices_astroid[0], color="blue", marker="s", label="False Negative", s=20)
       
        #axes[1, i].axis("off")

    plt.tight_layout()
    plot_filename = os.path.join(plots_dir, f"plot_{current_step}.png")
    plt.savefig(plot_filename)
    plt.close(fig) 

def create_gif(seed):
    """
    Combines all saved plots into a GIF and deletes the individual plot images.
    """
    plots_dir = "MoJo/morten/plots"
    gif_dir = "MoJo/morten/gif"
    os.makedirs(gif_dir, exist_ok=True)

    #gif_filename = os.path.join(gif_dir, "state_comparison.gif")
    video_filename = os.path.join(gif_dir, f"state_comparison_{seed}.mp4")

    # Get list of all plots
    image_files = sorted(glob.glob(os.path.join(plots_dir, "plot_*.png")))

    if not image_files:
        print("No images found for GIF creation.")
        return
    
    frames = []
    for i in range(len(image_files)):
        img = imageio.imread(os.path.join(plots_dir, f"plot_{i+1}.png"))
        
        # Fix dimensions to be multiples of 16
        height, width = img.shape[:2]
        new_height = (height + 15) // 16 * 16  # Round up to multiple of 16
        new_width = (width + 15) // 16 * 16  # Round up to multiple of 16

        if (height, width) != (new_height, new_width):
            img = np.pad(img, ((0, new_height - height), (0, new_width - width), (0, 0)), mode='constant', constant_values=0)
        
        frames.append(img)
    imageio.mimsave(video_filename, frames, fps=1, codec="libx264", quality=10)

    # Remove individual plot images after GIF creation
    for img in image_files:
        os.remove(img)

    print(f"GIF saved at: {video_filename}")