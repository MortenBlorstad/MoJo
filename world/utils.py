import json
from argparse import Namespace
import numpy as np
import jax.numpy as jnp
import os

#Taken from kit
def from_json(state):
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
def filterObservation(obs:jnp.ndarray):    
    return obs[jnp.where((obs[:,0] != -1) & (obs[:,1] != -1))]

#filter out missing values and swap axes
def swapAndFilterObservation(obs:jnp.ndarray):    
    return filterObservation(obs)[:, [1, 0]]

#Handle symmetric inserts
def symmetric(el):
    return[23-el[1],23-el[0]]

#Handle symmetric inserts for whole lists
def symmlist(l):
    return l.copy() + [symmetric(el) for el in l]

#Returns jnp.array 'A' subtract 'B'
def reduce(A,B):
    dims = jnp.maximum(B.max(0),A.max(0))+1
    return A[~jnp.isin(jnp.ravel_multi_index(A.T,dims),jnp.ravel_multi_index(B.T,dims))]

#Returns jnp.array 'A' subtract 'B' and points scored
def pointreduce(A,B):
    dims = jnp.maximum(B.max(0),A.max(0))+1
    inb = jnp.isin(jnp.ravel_multi_index(A.T,dims),jnp.ravel_multi_index(B.T,dims))        
    return A[~inb],jnp.unique(A[inb],axis=0).shape[0]


#Agent main.py uses Namespace for parsing the json as observation. Let's do the same.
def getObsNamespace(file):   

    # Open and read the (semi) JSON file        
    with open(file, 'r') as file:
        inputs = file.read()
        
    #Parse
    raw_input = json.loads(inputs)

    #Return all data as Namespace
    return Namespace(**dict(step=raw_input["step"], obs=raw_input["obs"], remainingOverageTime=raw_input["remainingOverageTime"], player=raw_input["player"], info=raw_input["info"]))

#Path to replay files
def getPath(seed, obs):
    path1 = f"MoJo/world/replay/{seed}/obs_{obs}"
    path2 = f"../MoJo/world/replay/{seed}/obs_{obs}"
    # Check if the first path exists, else return the alternative path
    return path1 if os.path.exists(path1) else path2       

#Call this function to get an observation. Note that only first observations (step = 0) contains 'env_cfg'. Otherwise, empty dict is returned.
#'step' & 'player' is also contained in 'observation.obs' It is returned seperately to mimimc the behaviour of main.py:agent_fn(...) 
def getObservation(seed,step):
    observation = getObsNamespace(getPath(seed,step))    

    config = {}
    if(step == 0):
        config = observation.info['env_cfg']

    #Match parameters in Agent:main.py. We don't need all of this
    return observation.step, observation.player, from_json(observation.obs), config, observation.remainingOverageTime

#Unicode representation of Lux map (for example ship position probabilities)
def printmap(arr, header = None): 

            #Unicode map characters
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

            #First / last line of map, depending on param isstop
            def mapstartstop(columns,istop):
                if(istop):
                    rc = tr #Right corner
                    lc = tl #Left corner
                    f = ft  #Fork
                else:
                    rc = br #Right corner
                    lc = bl #Left corner
                    f = fb  #Fork

                x = []        
                for i in range(columns):
                    x.extend([f,ue,ue,ue])
                x[0] = lc
                x.append(rc)
                return x
            
            #Map line
            def mapline(columns):
                x = []        
                for i in range(columns):
                    x.extend([cr,ue,ue,ue])
                x[0] = fl
                x.append(fr)
                return x

            #Format values
            def getval(v):
                if(v >= 10):                
                    return str(round(v)).ljust(3)
                elif(v >= 1):
                    return '{:.1f}'.format(round(v, 1))
                else:
                    return '{:.2f}'.format(round(v, 2))[1:]
            
            #Format a map line with values
            def mapvals(values):
                x = []
                for v in values:
                    x.append(hl)
                    x.append(getval(v))
                x.append(hl)
                return x            

            #------------------------------------------------
            #Begin processing map array
            #------------------------------------------------

            #shape of input
            shp = arr.shape   

            #Print header
            if(header != None):
                print(header)

            #Process map array
            print(''.join(mapstartstop(shp[1],True)))
            for i in range(shp[0]):                        
                print(''.join(mapvals(arr[i])))
                if(i != (shp[0]-1)):
                    print(''.join(mapline(shp[1])))            
            print(''.join(mapstartstop(shp[1],False))) 


#Unicode representation of Lux map (for example ship position probabilities)
def prmap(arr, header = None): 

            #Unicode map characters
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

            #First / last line of map, depending on param isstop
            def mapstartstop(columns,istop):
                if(istop):
                    rc = tr #Right corner
                    lc = tl #Left corner
                    f = ft  #Fork
                else:
                    rc = br #Right corner
                    lc = bl #Left corner
                    f = fb  #Fork

                x = []        
                for i in range(columns):
                    x.extend([f,ue,ue,ue])
                x[0] = lc
                x.append(rc)
                return x
            
            #Map line
            def mapline(columns):
                x = []        
                for i in range(columns):
                    x.extend([cr,ue,ue,ue])
                x[0] = fl
                x.append(fr)
                return x

            #Format values
            def getval(v):
                if(v == 0):
                    return '   '
                else:
                    return ' 1 '
                            
            #Format a map line with values
            def mapvals(values):
                x = []
                for v in values:
                    x.append(hl)
                    x.append(getval(v))
                x.append(hl)
                return x            

            #------------------------------------------------
            #Begin processing map array
            #------------------------------------------------

            #shape of input
            shp = arr.shape   

            #Print header
            if(header != None):
                print(header)

            #Process map array
            print(''.join(mapstartstop(shp[1],True)))
            for i in range(shp[0]):                        
                print(''.join(mapvals(arr[i])))
                if(i != (shp[0]-1)):
                    print(''.join(mapline(shp[1])))            
            print(''.join(mapstartstop(shp[1],False))) 



def get_symmetric_coordinates(indices:np.array, nrows = 24, ncols = 24):
    """
    Given coordinates (i, j), returns the symmetric coordinates (j, i)
    along the main diagonal of a square grid.
    
    Args:
        indices (np.ndarray): A tuple or array of row and column indices.
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
    
    Returns:
        np.ndarray: Array with swapped and transformed (j, i) coordinates.
    """
    i, j = indices
    return ncols-j-1, nrows-i-1  # Swap i and j