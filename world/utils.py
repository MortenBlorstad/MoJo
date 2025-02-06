import json
from argparse import Namespace
import numpy as np
import jax.numpy as jnp


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

#obs->array 
def fromObs(obs):
    return jnp.array(obs[0])

#obs->array, filter out missing values
def fromObsFiltered(obs):
    obs = fromObs(obs)
    return obs[jnp.where((obs[:,0] != -1) & (obs[:,1] != -1))]

#obs->array, filter out missing values, swap x/y
def fromObsFilteredSwap(obs):
    return fromObsFiltered(obs)[:, [1, 0]]

#Returns jnp.array of values that are in 'A' but not in 'B'
def reduce(A,B):
    dims = jnp.maximum(B.max(0),A.max(0))+1
    return A[~jnp.isin(jnp.ravel_multi_index(A.T,dims),jnp.ravel_multi_index(B.T,dims))]

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
    return "../MoJo/world/replay/{seed}/obs_{obs}".format(seed = seed, obs = obs)        

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