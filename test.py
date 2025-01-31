from agents.agent import Agent
#from world.universe import Universe

from world.universe import Universe

from world.obs_to_state import State
import jax.numpy as jnp
from world.utils import getObservation, getObsNamespace, getPath, from_json
from world.nebula import Nebula
import flax
from agents.agent import Agent

#####
# create test env where we have the true state
#
#####

from luxai_s3.wrappers import LuxAIS3GymEnv


import numpy as np
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


import matplotlib.pyplot as plt
import numpy as np
import os
import glob

import imageio.v2 as imageio
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
def pad_image(image, block_size=16):
    """Pads an image so its dimensions are divisible by block_size."""
    h, w = image.shape[:2]
    new_h = (h + block_size - 1) // block_size * block_size
    new_w = (w + block_size - 1) // block_size * block_size

    padded_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)
    padded_image[:h, :w] = image
    return padded_image

def plot_state_comparison(current_step,correct_states, predicted_states):
    """
    Plots the correct states in the first row and predicted states in the second row.
    
    Args:
        correct_states (list of np.array): List of correct 24x24 grids.
        predicted_states (list of np.array): List of predicted 24x24 grids.
    """
    
    plots_dir = "MoJo/plots"
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        correct_state = np.array(correct_states[i])
        predicted_state = np.array(predicted_states[i])
                
        # Identify false positives: places where prediction is 1 but correct state is 0
        false_positive_indices = np.array(np.where((predicted_state == 1) & (correct_state == 0)))
        false_negative_indices = np.array(np.where((predicted_state == 0) & (correct_state == 1)))


        # Plot correct state (top row)
        axes[0, i].imshow(correct_state, aspect="auto")
        axes[0, i].set_title(f"Correct - t={current_step+i}")
        #axes[0, i].axis("off")

        # Plot predicted state (bottom row)
        axes[1, i].imshow(predicted_state, aspect="auto")
        axes[1, i].set_title(f"Predicted - t={current_step+i}")
        axes[1, i].scatter(false_positive_indices[1], false_positive_indices[0], color="red", marker="x", label="False Positive", s=20)
        axes[1, i].scatter(false_negative_indices[1], false_negative_indices[0], color="blue", marker="s", label="False Negative", s=20)
        #axes[1, i].axis("off")

    plt.tight_layout()
    plot_filename = os.path.join(plots_dir, f"plot_{current_step}.png")
    plt.savefig(plot_filename)
    plt.close(fig) 

def create_gif():
    """
    Combines all saved plots into a GIF and deletes the individual plot images.
    """
    plots_dir = "MoJo/plots"
    gif_dir = "MoJo/gif"
    os.makedirs(gif_dir, exist_ok=True)

    #gif_filename = os.path.join(gif_dir, "state_comparison.gif")
    video_filename = os.path.join(gif_dir, "state_comparison.mp4")

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



env = LuxAIS3GymEnv() 
seed = 1#0 #223344 #
obs, info = env.reset(seed=seed)
actions = {
        "player_0": jnp.ones((16, 3), dtype=int),
        "player_1": jnp.ones((16, 3), dtype=int)
        }
#env.env_params["nebula_tile_drift_speed"] = -0.05
nebula = Nebula(3)
#print(info)

env_cfg = {"max_units":16, "map_width":24, "map_height": 24}
agent = Agent("player_1",env_cfg)
obs = flax.serialization.to_state_dict(obs)["player_1"] 
print(env.env_params.nebula_tile_drift_speed)
episode = {}
for step in range(1,100):
    actions["player_1"] = agent.act(step,obs)
    obs, _, _, _, info,state = env.step(actions)
    obs = flax.serialization.to_state_dict(obs)["player_1"] 
    my_state = State(obs, "player_1")
    state = State(flax.serialization.to_state_dict(info['final_state']), "player_1")
    
    episode[step] = {}
    episode[step]["state"] = state
    episode[step]["obs"] = my_state


for step in range(1,96):
    obs = episode[step]["obs"]
    nebulas = jnp.array(obs.nebulas.copy())
    observable = jnp.array(obs.observeable_tiles.copy())
    
    nebula.learn(nebulas,observable,step-1)
    predictions = nebula.predict(nebulas,observable,step-1)
    prediction = predictions[0]
    solution = episode[step]["state"].nebulas
    solutions = [episode[step+i]["state"].nebulas for i in range(4)]


    print(step)
    plot_state_comparison(step, solutions, predictions)
    print("current", check_prediction_accuracy(solution, prediction))
    # if not check_prediction_accuracy(solution, prediction):
    #     print(solution.T)
    #     print(prediction.T)
    print("\n")
    prediction = predictions[1]
    solution = episode[step+1]["state"].nebulas
    
    print("next prediction", check_prediction_accuracy(solution, prediction))
    # if not check_prediction_accuracy(solution, prediction):
    #     print(solution.T)
    #     print(prediction.T)

    
    print("\n")
    prediction = predictions[2]
    solution = episode[step+2]["state"].nebulas
    
    print("next next prediction", check_prediction_accuracy(solution, prediction))
    # if not check_prediction_accuracy(solution, prediction):
    #     print(solution.T)
    #     print(prediction.T)
    
    print("\n")
    prediction = predictions[3]
    solution = episode[step+3]["state"].nebulas
    
    print("next next next prediction", check_prediction_accuracy(solution, prediction))
    # if not check_prediction_accuracy(solution, prediction):
    #     print(solution.T)
    #     print(prediction.T)
    print("=====\n")
    
#create_gif()

create_gif()
    
 

    # _, _, obs, _, _ = getObservation(seed,step-1)
    # state = flax.serialization.to_state_dict(info['final_state'])

    # my_state = State(obs, "player_1")
    # nebulas = jnp.array(my_state.nebulas.copy())
    # observable = jnp.array(my_state.observeable_tiles.copy())
    # nebula.learn(nebulas,observable,step-1)
    # predictions = nebula.predict(nebulas,observable,step-1)
    # prediction = predictions[1]

    # correct_state = State(state, "player_1")
    # correct_nebulas = jnp.array(correct_state.nebulas.copy())

    # print(step -1, check_prediction_accuracy(correct_nebulas, prediction))
    # if not check_prediction_accuracy(correct_nebulas, prediction):
    #     print(correct_nebulas.T)
    #     print(prediction.T)



# all_true = True
# next_nebulas = None
# for step in range(1,16):
#     actions["player_1"] = agent.act(step,obs)
#     nex_obs, _, _, _, info = env.step(actions)
#     #print(info.keys())
#     #print(info['final_state'])
#     state = flax.serialization.to_state_dict(info['final_state'])
#     nex_obs = flax.serialization.to_state_dict(nex_obs)["player_1"] 
#     #print(obs)
#     print("========")
#     print(step)
#     my_state = State(obs, "player_1")
#     my_next_state = State(nex_obs, "player_1")
#     nebulas = jnp.array(my_state.nebulas,copy=True)
#     observable = jnp.array(my_state.observeable_tiles.copy(), copy=True)
    
#     nebula.learn(nebulas,observable,step-1)
#     predictions = nebula.predict(nebulas,observable,step)
#     next_pred = predictions[1]
#     state = flax.serialization.to_state_dict(state)
#     correct_state = State(state, "player_1")
#     correct_nebulas = jnp.array(correct_state.nebulas, copy=True)
#     if next_nebulas is not None:
#         print(check_prediction_accuracy(correct_nebulas, next_pred),env.env_params.nebula_tile_drift_speed, nebula.nebula_tile_drift_speed)
#         if not check_prediction_accuracy(correct_nebulas, next_pred):
#             print(correct_nebulas.T)
#             print(next_pred.T)
#     print("========")
#     obs = nex_obs
#     next_nebulas = correct_nebulas.copy()
    

    
    
#     print(correct_nebulas.T)
#     print(prediction.T)
    
#     if not check_prediction_accuracy(correct_nebulas, prediction):
#         all_true = False
#     print(step, check_prediction_accuracy(correct_nebulas, prediction),env.env_params.nebula_tile_drift_speed, nebula.nebula_tile_drift_speed)
#     print("========\n")
#     # if not check_prediction_accuracy(correct_nebulas, prediction):
#     #     print(correct_nebulas.T)
#     #     print(prediction.T)
#     #     break

# print(f"passed_test {all_true} and found correct drift_speed {np.round(env.env_params.nebula_tile_drift_speed,2)} {abs(env.env_params.nebula_tile_drift_speed-nebula.nebula_tile_drift_speed)<1e-3}")
