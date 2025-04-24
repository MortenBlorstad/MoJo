"""
Utility functions for the MoJo project.
This module contains various helper functions for data processing, file handling, and visualization.
"""

import json
from argparse import Namespace
import numpy as np
import jax.numpy as jnp
import os
from typing import Any, Dict, List, Tuple, Union, Optional

#Taken from kit
def from_json(state: Union[List, Dict, Any]) -> Union[jnp.ndarray, Dict, Any]:
    """
    Convert JSON-like data structure to JAX numpy arrays.
    
    Args:
        state: Input data structure (list, dict, or other)
        
    Returns:
        Converted data structure with lists replaced by JAX numpy arrays
    """
    if isinstance(state, list):
        return jnp.array(state)                                             #<-----------Change this to use numpy instead
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state 

#filter out missing values of observation
def filterObservation(obs: jnp.ndarray) -> jnp.ndarray:    
    """
    Filter out missing values from observation array.
    
    Args:
        obs: Input observation array
        
    Returns:
        Filtered array with only valid observations
    """
    return obs[jnp.where((obs[:,0] != -1) & (obs[:,1] != -1))]

#filter out missing values and swap axes
def swapAndFilterObservation(obs: jnp.ndarray) -> jnp.ndarray:    
    """
    Filter out missing values and swap x/y coordinates.
    
    Args:
        obs: Input observation array
        
    Returns:
        Filtered array with swapped coordinates
    """
    return filterObservation(obs)[:, [1, 0]]

#Handle symmetric inserts
def symmetric(el: List[int]) -> List[int]:
    """
    Calculate symmetric position across the main diagonal.
    
    Args:
        el: Input position [x, y]
        
    Returns:
        Symmetric position [y, x]
    """
    return[23-el[1],23-el[0]]

#Handle symmetric inserts for whole lists
def symmlist(l: List[List[int]]) -> List[List[int]]:
    """
    Generate symmetric list by adding symmetric positions.
    
    Args:
        l: Input list of positions
        
    Returns:
        List with original and symmetric positions
    """
    return l.copy() + [symmetric(el) for el in l]

#Returns jnp.array 'A' subtract 'B'
def reduce(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Compute set difference A - B.
    
    Args:
        A: First set of positions
        B: Second set of positions
        
    Returns:
        Positions in A but not in B
    """
    dims = jnp.maximum(B.max(0),A.max(0))+1
    return A[~jnp.isin(jnp.ravel_multi_index(A.T,dims),jnp.ravel_multi_index(B.T,dims))]

#Returns jnp.array 'A' subtract 'B' and points scored
def pointreduce(A: jnp.ndarray, B: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
    """
    Compute set difference and count points scored.
    
    Args:
        A: First set of positions
        B: Second set of positions
        
    Returns:
        Tuple of (positions in A but not in B, number of points scored)
    """
    dims = jnp.maximum(B.max(0),A.max(0))+1
    inb = jnp.isin(jnp.ravel_multi_index(A.T,dims),jnp.ravel_multi_index(B.T,dims))        
    return A[~inb],jnp.unique(A[inb],axis=0).shape[0]


#Agent main.py uses Namespace for parsing the json as observation. Let's do the same.
def getObsNamespace(file: str) -> Namespace:   
    """
    Parse observation file into Namespace object.
    
    Args:
        file: Path to observation file
        
    Returns:
        Namespace containing parsed observation data
    """
    # Open and read the (semi) JSON file        
    with open(file, 'r') as file:
        inputs = file.read()
        
    #Parse
    raw_input = json.loads(inputs)

    #Return all data as Namespace
    return Namespace(**dict(step=raw_input["step"], obs=raw_input["obs"], remainingOverageTime=raw_input["remainingOverageTime"], player=raw_input["player"], info=raw_input["info"]))

#Path to replay files
def getPath(seed: str, obs: int) -> str:
    """
    Get path to replay file.
    
    Args:
        seed: Game seed
        obs: Observation number
        
    Returns:
        Path to replay file
    """
    path1 = f"MoJo/world/replay/{seed}/obs_{obs}"
    path2 = f"../MoJo/world/replay/{seed}/obs_{obs}"
    # Check if the first path exists, else return the alternative path
    return path1 if os.path.exists(path1) else path2       

#Call this function to get an observation. Note that only first observations (step = 0) contains 'env_cfg'. Otherwise, empty dict is returned.
#'step' & 'player' is also contained in 'observation.obs' It is returned seperately to mimimc the behaviour of main.py:agent_fn(...) 
def getObservation(seed: str, step: int) -> Tuple[int, str, Dict, Dict, int]:
    """
    Get observation data for a specific game step.
    
    Args:
        seed: Game seed
        step: Step number
        
    Returns:
        Tuple containing (step, player, observation, config, remainingOverageTime)
    """
    observation = getObsNamespace(getPath(seed,step))    

    config = {}
    if(step == 0):
        config = observation.info['env_cfg']

    #Match parameters in Agent:main.py. We don't need all of this
    return observation.step, observation.player, from_json(observation.obs), config, observation.remainingOverageTime

#Unicode representation of Lux map (for example ship position probabilities)
def printmap(arr: np.ndarray, header: Optional[str] = None) -> None: 
    """
    Print a map using Unicode box-drawing characters.
    
    Args:
        arr: 2D array to visualize
        header: Optional header text
    """
    # Unicode map characters
    tl = '\u250C'  # Top left corner
    tr = '\u2510'  # Top right corner
    ue = '\u2500'  # Horizontal line
    bl = '\u2514'  # Bottom left corner
    br = '\u2518'  # Bottom right corner
    cr = '\u253C'  # Cross
    fl = '\u251C'  # Left fork
    fr = '\u2524'  # Right fork
    fb = '\u2534'  # Bottom fork
    ft = '\u252C'  # Top fork
    hl = '\u2502'  # Vertical line

    def mapstartstop(columns: int, istop: bool) -> List[str]:
        """Generate top or bottom border of map."""
        rc = tr if istop else br
        lc = tl if istop else bl
        f = ft if istop else fb
        x = []        
        for _ in range(columns):
            x.extend([f, ue, ue, ue])
        x[0] = lc
        x.append(rc)
        return x
            
    def mapline(columns: int) -> List[str]:
        """Generate horizontal divider line."""
        x = []        
        for _ in range(columns):
            x.extend([cr, ue, ue, ue])
        x[0] = fl
        x.append(fr)
        return x

    def getval(v: float) -> str:
        """Format value for display."""
        if v >= 10:                
            return str(round(v)).ljust(3)
        elif v >= 1:
            return '{:.1f}'.format(round(v, 1))
        else:
            return '{:.2f}'.format(round(v, 2))[1:]
            
    def mapvals(values: np.ndarray) -> List[str]:
        """Format a row of values for display."""
        x = []
        for v in values:
            x.append(hl)
            x.append(getval(v))
        x.append(hl)
        return x            

    shp = arr.shape   
    if header is not None:
        print(header)
    print(''.join(mapstartstop(shp[1], True)))
    for i in range(shp[0]):                        
        print(''.join(mapvals(arr[i])))
        if i != (shp[0]-1):
            print(''.join(mapline(shp[1])))            
    print(''.join(mapstartstop(shp[1], False)))

#Unicode representation of Lux map (for example ship position probabilities)
def prmap(arr: np.ndarray, header: Optional[str] = None) -> None: 
    """
    Print a binary map using Unicode box-drawing characters.
    
    Args:
        arr: 2D binary array to visualize
        header: Optional header text
    """
    # Unicode map characters (same as printmap)
    tl = '\u250C'
    tr = '\u2510'
    ue = '\u2500'
    bl = '\u2514'
    br = '\u2518'
    cr = '\u253C'
    fl = '\u251C'
    fr = '\u2524'
    fb = '\u2534'        
    ft = '\u252C'        
    hl = '\u2502'        

    def mapstartstop(columns: int, istop: bool) -> List[str]:
        rc = tr if istop else br
        lc = tl if istop else bl
        f = ft if istop else fb
        x = []        
        for _ in range(columns):
            x.extend([f, ue, ue, ue])
        x[0] = lc
        x.append(rc)
        return x
            
    def mapline(columns: int) -> List[str]:
        x = []        
        for _ in range(columns):
            x.extend([cr, ue, ue, ue])
        x[0] = fl
        x.append(fr)
        return x

    def getval(v: float) -> str:
        return '   ' if v == 0 else ' 1 '
                            
    def mapvals(values: np.ndarray) -> List[str]:
        x = []
        for v in values:
            x.append(hl)
            x.append(getval(v))
        x.append(hl)
        return x            

    shp = arr.shape   
    if header is not None:
        print(header)
    print(''.join(mapstartstop(shp[1], True)))
    for i in range(shp[0]):                        
        print(''.join(mapvals(arr[i])))
        if i != (shp[0]-1):
            print(''.join(mapline(shp[1])))            
    print(''.join(mapstartstop(shp[1], False)))

def get_symmetric_coordinates(indices: np.ndarray, nrows: int = 24, ncols: int = 24) -> Tuple[int, int]:
    """
    Get symmetric coordinates across the main diagonal.
    
    Args:
        indices: Input coordinates (i, j)
        nrows: Number of rows in grid
        ncols: Number of columns in grid
        
    Returns:
        Tuple of symmetric coordinates (j, i)
    """
    i, j = indices
    return ncols-j-1, nrows-i-1  # Swap i and j

