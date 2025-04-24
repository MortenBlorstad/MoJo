import sys
import os
from os import listdir
from os.path import isfile, join
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hierarchical.director.multiagentppo import MultiAgentPPO
from hierarchical.director.vae import VAE
from collections import deque
import numpy as np

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

def getZapCoordsOnly(x, y, zaprange, probmap):
    pos, probmap = getZapCoords((x,y), zaprange, probmap)
    return pos[0], pos[1]

def getfiles(mypath):
    return [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


#Running averages for training
class RunningAverages:

    def __init__(self, window_size=10):
        
        self.mgrloss = deque(maxlen=window_size)
        self.wrkloss = deque(maxlen=window_size)
        self.wmloss = deque(maxlen=window_size)
        self.goalloss = deque(maxlen=window_size)

    def append(self, losstype, value):        

        if losstype == "mgrloss":
            self.mgrloss.append(value)
        elif losstype == "wrkloss":
            self.wrkloss.append(value)
        elif losstype == "wmloss":
            self.wmloss.append(value)
        elif losstype == "goalloss":
            self.goalloss.append(value)
    
    def __str__(self):
        return f"Mgr loss: {np.mean(self.mgrloss)}\tWrk loss: {np.mean(self.wrkloss)}\tWmloss: {np.mean(self.wmloss)}\tGoalloss: {np.mean(self.goalloss)}"


def InitWorker(cfg, fromfile = True):

    wrk = MultiAgentPPO(
        cfg['state_dim'],
        cfg['action_dim'],
        float(cfg['lr_actor']), #WTF!!!
        cfg['lr_critic'],
        cfg['gamma'],
        cfg['K_epochs'],
        cfg['eps_clip'],
        cfg['cntns_actn_spc'],
        cfg['action_std'],
        cfg['behaviors'],
        cfg['isWorker']
    )
    if fromfile and os.path.exists(cfg['modelfile']):
        wrk.load(cfg['modelfile'])
    return wrk

def InitManager(cfg, fromfile = True):

    mgr = MultiAgentPPO(
        cfg['state_dim'],
        cfg['action_dim'],
        float(cfg['lr_actor']), #WTF!!!
        cfg['lr_critic'],
        cfg['gamma'],
        cfg['K_epochs'],
        cfg['eps_clip'],
        cfg['cntns_actn_spc'],
        cfg['action_std'],
        cfg['behaviors'],
        cfg['isWorker']
    )
    if fromfile and os.path.exists(cfg['modelfile']):
        mgr.load(cfg['modelfile'])
    return mgr