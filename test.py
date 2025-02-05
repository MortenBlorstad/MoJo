from agents.agent import Agent
#from world.universe import Universe

from world.universe import Universe

from world.obs_to_state import State
import jax.numpy as jnp
from world.utils import getObservation, getObsNamespace, getPath, from_json
from world.nebula import Nebula
from world.nebula_astroid import NebulaAstroid
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

def plot_state_comparison(current_step, correct_states, predicted_states, observable):
    """
    Plots the correct states in the first row and predicted states in the second row.
    
    Args:
        correct_states (list of np.array): List of correct 24x24 grids.
        predicted_states (list of np.array): List of predicted 24x24 grids.
    """
    
    plots_dir = "MoJo/plots"
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    observable = np.array(observable,dtype=float)
    observable[observable==1]=np.inf
    nebula_predictions, astroid_predictions = predicted_states
    for i in range(4):
        correct_state = np.array(correct_states[i],dtype=float)
        
        

        nebula_prediction = np.array(nebula_predictions[i],dtype=float)
        astroid_prediction = np.array(astroid_predictions[i],dtype=float)
        

                
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
    plots_dir = "MoJo/plots"
    gif_dir = "MoJo/gif"
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



env = LuxAIS3GymEnv() 
#seed 0: -0.05 ok
#seed 1: -0.1 ok
#seed 2: 0.1 ok
#seed 3: -0.1
#seed 223344: -0.15 ok
seed = 223344 #223344 #2#
obs, info = env.reset(seed=seed)
actions = {
        "player_0": jnp.ones((16, 3), dtype=int),
        "player_1": jnp.ones((16, 3), dtype=int)
        }
#env.env_params["nebula_tile_drift_speed"] = -0.05
#nebula = Nebula(3)
nebula =  NebulaAstroid(3)
#print(info)

env_cfg = {"max_units":16, "map_width":24, "map_height": 24}
agent = Agent("player_1",env_cfg)
obs = flax.serialization.to_state_dict(obs)["player_1"] 
print(env.env_params.nebula_tile_drift_speed)
episode = {}
n_steps = 100
for step in range(1,n_steps):
    actions["player_1"] = agent.act(step,obs)
    obs, _, _, _, info,state = env.step(actions)
    obs = flax.serialization.to_state_dict(obs)["player_1"] 
    my_state = State(obs, "player_1")
    state = State(flax.serialization.to_state_dict(info['final_state']), "player_1")
    
    episode[step] = {}
    episode[step]["state"] = state
    episode[step]["obs"] = my_state


for step in range(1,n_steps-4):
    obs = episode[step]["obs"]

    nebulas =  jnp.array(obs.nebulas.copy(), ) #jnp.array(obs.asteroids.copy()) #
    astroids = jnp.array(obs.asteroids.copy())
    observable = jnp.array(obs.observeable_tiles.copy())
    print(step)
    nebula.learn(nebulas,astroids,observable,step)
    predictions = nebula.predict(nebulas,astroids,observable,step)
    
    solution = jnp.stack([episode[step]["state"].nebulas,episode[step]["state"].asteroids,
                           jnp.clip(episode[step]["state"].player_units_count, a_min =0, a_max = 1)],axis=-1)
    solutions = [jnp.stack([episode[step+i]["state"].nebulas,
                            episode[step+i]["state"].asteroids,
                             jnp.clip(episode[step+i]["state"].player_units_count, a_min =0, a_max = 1)],
                             axis=-1) for i in range(4)]
    



    
    plot_state_comparison(step, solutions, predictions,observable)
    #print("current", check_prediction_accuracy(solution, prediction))
    # if not check_prediction_accuracy(solution, prediction):
    #     print(solution.T)
    #     print(prediction.T)
    # print("\n")
    # prediction = predictions[1]
    # solution = episode[step+1]["state"].nebulas
    
    # print("next prediction", check_prediction_accuracy(solution, prediction))
    # if not check_prediction_accuracy(solution, prediction):
    #     print(solution.T)
    #     print(prediction.T)

    
    # print("\n")
    # prediction = predictions[2]
    # solution = episode[step+2]["state"].nebulas
    
    #print("next next prediction", check_prediction_accuracy(solution, prediction))
    # if not check_prediction_accuracy(solution, prediction):
    #     print(solution.T)
    #     print(prediction.T)
    
    # print("\n")
    # prediction = predictions[3]
    # solution = episode[step+3]["state"].nebulas
    
    # print("next next next prediction", check_prediction_accuracy(solution, prediction))
    # if not check_prediction_accuracy(solution, prediction):
    #     print(solution.T)
    #     print(prediction.T)
    #print("=====\n")
    
#create_gif()

create_gif(seed)
    
 

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
