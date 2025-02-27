import jax.numpy as jnp

#Filter out possible attacking positions based on zap range and the ships current location
def getMapRange(position, zaprange, probmap):

    x_lower = max(0,position[0]-zaprange)         #Avoid x pos < 0
    x_upper = min(size,position[0]+zaprange+1)    #Avoid x pos > map height
    y_lower = max(0,position[1]-zaprange)         #Avoid y pos < 0
    y_upper = min(size,position[1]+zaprange+1)    #Avoid y pos > map width

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


if __name__ == "__main__":
    #Example code (using small map)
    #----------------------------------------------------------------------
    pos = (4,4)                         #Ships position
    rng = 2                             #Zap range
    size = 8                            #Use a demo map of 8x8
    basemap = jnp.zeros((size,size))    #Create a demo map
    basemap = basemap.at[0,0].set(.1)   #Set some probabilities
    basemap = basemap.at[2,2].set(.1)   #Set some probabilities
    #Run the example
    #----------------------------------------------------------------------
    p,v = getZapCoords(pos, rng, basemap)
    print("Pick position",p,"because it has the optimal value of",v)