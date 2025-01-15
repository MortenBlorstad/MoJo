import jax.numpy as jnp

horizont = 2
mapshape = (5,5)
actions = 6
directions = jnp.array(
    [
        [0, 0],  # Do nothing
        [0, -1],  # Move up
        [1, 0],  # Move right
        [0, 1],  # Move down
        [-1, 0],  # Move left
    ],
    dtype=jnp.int16,
)

#Distribute probabilities
def probDistribute(lastprobmap, future):

    #Get indices of tiles that, with probability > 0, has a ship placed on it
    idx = jnp.where(lastprobmap > 0)
    vals = lastprobmap[idx]

    denominator = actions**(future+1)

    #Start with an empty space
    nw = jnp.zeros(mapshape)

    #For the x,y pairs of tiles containing a ship
    for x, y, v in zip(idx[0],idx[1], vals):        

        #Get possible new positions (Cardinal directions + no move)
        t = jnp.array([x,y]) + directions
        
        #Get rid of OOB map positions
        t = t[jnp.where((t[:,0] >= 0) & (t[:,1] >= 0) & (t[:,0] < mapshape[0]) & (t[:,1] < mapshape[1]))]

        #Using sum of probabilities. We could end up in a situation of p(ship in tile) > 1, but we don't care
        nw = jnp.add.at(nw, (t[:,0], t[:,1]), v/denominator, inplace=False)

    return nw

#Predict the future positions of ships. Return dim : MAPWIDTHxMAPHEIGHTSxHORIZON+1
def predictShipPositions(map):   #Current position of ships

    l = []
    l.append(map)
    for  i in range(horizont):
        map = probDistribute(map, i)
        l.append(map)
    return jnp.stack(l,axis = 0)
        
#Demo map
d = jnp.zeros(mapshape)
d = d.at[2,2].set(1)

r = predictShipPositions(d)
print('Shape of vector is',r.shape,'and it looks like this:\n\n',r)