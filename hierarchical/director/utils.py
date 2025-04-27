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
from typing import List, Tuple, Union, Dict, Any
from pathlib import Path
import jax.numpy as jnp

def direction_to(src: np.ndarray, target: np.ndarray) -> int:
    """
    Calculate the direction from source to target position.
    Returns an integer representing the direction:
    1: Up
    2: Right
    3: Down
    4: Left
    
    Args:
        src (np.ndarray): Source position coordinates
        target (np.ndarray): Target position coordinates
        
    Returns:
        int: Direction code (1-4)
    """
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
def getMapRange(position: Tuple[int, int], zaprange: int, 
               probmap: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Filter out possible attacking positions based on zap range and current location.
    
    Args:
        position (Tuple[int, int]): Current position coordinates
        zaprange (int): Maximum zap range
        probmap (np.ndarray): Probability map
        
    Returns:
        Tuple[np.ndarray, int, int]: 
            - Filtered probability map
            - Lower x bound
            - Lower y bound
    """
    
    x_lower = max(0,position[0]-zaprange)         #Avoid x pos < 0
    x_upper = min(24,position[0]+zaprange+1)    #Avoid x pos > map height
    y_lower = max(0,position[1]-zaprange)         #Avoid y pos < 0
    y_upper = min(24,position[1]+zaprange+1)    #Avoid y pos > map width

    #Filter out the probabilities of ships within zap-range
    return probmap[x_lower:x_upper,y_lower:y_upper], x_lower,y_lower
    
#Get the coordinates of the tile with the highest probability of enemy ship given a zap range and a ship location
def getZapCoords(position: Tuple[int, int], zaprange: int, 
                probmap: np.ndarray) -> Tuple[Tuple[int, int], float]:
    """
    Get coordinates of tile with highest probability of enemy ship.
    
    Args:
        position (Tuple[int, int]): Current position coordinates
        zaprange (int): Maximum zap range
        probmap (np.ndarray): Probability map
        
    Returns:
        Tuple[Tuple[int, int], float]: 
            - Target coordinates
            - Probability at target
    """
    filteredMap, x_l, y_l = getMapRange(position, zaprange, probmap)    
    x_local,y_local = divmod(int(jnp.argmax(filteredMap)),filteredMap.shape[0])
    
    
    # Convert to global coordinates
    x_global = min(max(0, x_local + x_l), probmap.shape[0] - 1)
    y_global = min(max(0, y_local + y_l), probmap.shape[1] - 1)

    #Return target coordinates
    return (x_global,y_global),probmap[(x_global,y_global)]

def getZapCoordsOnly(x: int, y: int, zaprange: int, 
                    probmap: np.ndarray) -> Tuple[int, int]:
    """
    Get only the coordinates of the best zap target.
    
    Args:
        x (int): Current x position
        y (int): Current y position
        zaprange (int): Maximum zap range
        probmap (np.ndarray): Probability map
        
    Returns:
        Tuple[int, int]: Target coordinates
    """
    pos, probmap = getZapCoords((x,y), zaprange, probmap)
    return pos[0], pos[1]

def getfiles(mypath: Union[str, Path]) -> List[str]:
    """
    Get list of files in a directory.
    
    Args:
        mypath (Union[str, Path]): Directory path
        
    Returns:
        List[str]: List of file paths
    """
    return [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


#Running averages for training
class RunningAverages:
    """
    Class for maintaining running averages of different loss types.
    
    Attributes:
        mgrloss (deque): Manager loss history
        wrkloss (deque): Worker loss history
        wmloss (deque): World model loss history
        goalloss (deque): Goal loss history
    """
    def __init__(self, window_size: int = 10) -> None:
        """
        Initialize running averages with specified window size.
        
        Args:
            window_size (int): Size of the moving window (default: 10)
        """
        
        self.mgrloss = deque(maxlen=window_size)
        self.wrkloss = deque(maxlen=window_size)
        self.wmloss = deque(maxlen=window_size)
        self.goalloss = deque(maxlen=window_size)

    def append(self, losstype: str, value: float) -> None:
        """
        Append a new loss value to the specified loss type.
        
        Args:
            losstype (str): Type of loss ('mgrloss', 'wrkloss', 'wmloss', 'goalloss')
            value (float): Loss value to append
        """  

        if losstype == "mgrloss":
            self.mgrloss.append(value)
        elif losstype == "wrkloss":
            self.wrkloss.append(value)
        elif losstype == "wmloss":
            self.wmloss.append(value)
        elif losstype == "goalloss":
            self.goalloss.append(value)
    
    def __str__(self) -> str:
        """
        Get string representation of current averages.
        
        Returns:
            str: Formatted string of current averages
        """
        return f"Mgr loss: {np.mean(self.mgrloss)}\tWrk loss: {np.mean(self.wrkloss)}\tWmloss: {np.mean(self.wmloss)}\tGoalloss: {np.mean(self.goalloss)}"


def InitWorker(cfg: Dict[str, Any], fromfile: bool = True) -> MultiAgentPPO:
    """
    Initialize a worker agent.
    
    Args:
        cfg (Dict[str, Any]): Configuration dictionary
        fromfile (bool): Whether to load model from file (default: True)
        
    Returns:
        MultiAgentPPO: Initialized worker agent
    """

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

def InitManager(cfg: Dict[str, Any], fromfile: bool = True) -> MultiAgentPPO:
    """
    Initialize a manager agent.
    
    Args:
        cfg (Dict[str, Any]): Configuration dictionary
        fromfile (bool): Whether to load model from file (default: True)
        
    Returns:
        MultiAgentPPO: Initialized manager agent
    """

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