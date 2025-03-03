import sys
import os
from os import listdir
from os.path import isfile, join
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hierarchical.ppoalg import PPO
from hierarchical.vae import VAE

import jax.numpy as jnp

def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1
        

#Filter out possible attacking positions based on zap range and the ships current location
def getMapRange(position, zaprange, probmap):
    position = position.astype(int)
    x_lower = max(0,position[0]-zaprange)         #Avoid x pos < 0
    x_upper = min(24,position[0]+zaprange+1)    #Avoid x pos > map height
    y_lower = max(0,position[1]-zaprange)         #Avoid y pos < 0
    y_upper = min(24,position[1]+zaprange+1)    #Avoid y pos > map width

    #Filter out the probabilities of ships within zap-range
    return probmap[x_lower:x_upper,y_lower:y_upper], x_lower,y_lower
    
#Get the coordinates of the tile with the highest probability of enemy ship given a zap range and a ship location
def getZapCoords(position, zaprange, probmap):
    filteredMap, x_l, y_l = getMapRange(position, zaprange, probmap)    
    x,y = divmod(int(jnp.argmax(filteredMap)),filteredMap.shape[0])
    
    #Add back global indexing
    x+=x_l 
    y+=y_l

    #Return target coordinates
    return (x,y),probmap[(x,y)]

def getfiles(mypath):
    return [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

def InitPPO(cfg):

    return PPO(
        cfg['state_dim'],
        cfg['action_dim'],
        cfg['lr_actor'],
        cfg['lr_critic'],
        cfg['gamma'],
        cfg['K_epochs'],
        cfg['eps_clip'],
        cfg['has_continuous_action_space'],
        cfg['action_std']
    )

def InitVAE(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
    return VAE(
        device,
        cfg['input_dim'],
        cfg['hid1_dim'],
        cfg['hid2_dim'],
        cfg['latent_dim'],
        cfg['lr']        
    ).to(device)