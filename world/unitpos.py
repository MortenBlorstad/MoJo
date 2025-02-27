from world.base_component import base_component
import jax.numpy as jnp

class Unitpos(base_component):


    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon        
        self.mapsize = (24,24)

        self.directions = jnp.array(
            [
                [0, 0],     # Don't move                
                [-1, 0],    # Move up
                [0, 1],     # Move right
                [1, 0],     # Move down
                [0, -1],    # Move left
            ],
            dtype=jnp.int16,
        )
    
    #Calculate probability of ending up in a surrounding tile
    def getProbs(self,possible,v,astroids):

        #Create the list of probabilities
        probs = [1/6]*len(possible)
        probs[0]+=1/6*(6-len(possible))        
        probs = jnp.array(probs)*v

        #Update values based on astroid probabilities on a given tile
        for idx, a in enumerate(astroids[1:]):
            if(a > 0):                  
                reduction = probs[idx+1]*a                  #How big is the reduction?                               
                probs = probs.at[0].add(reduction)          #Increment probability of not moving
                probs = probs.at[idx+1].subtract(reduction) #Decrement probability of moving into astroid tile
        

        return probs

    #Distribute probabilities
    def probDistribute(self, lastprobmap, astroids):

        #Get indices of tiles that, with probability > 0, has a ship placed on it
        idx = jnp.where(lastprobmap > 0)
        vals = lastprobmap[idx]

        #Start with an empty space
        nw = jnp.zeros(self.mapsize)

        #For the x,y pairs of tiles containing a ship
        for x, y, v in zip(idx[0],idx[1], vals):        

            #Get possible new positions (Cardinal directions + no move)
            t = jnp.array([x,y]) + self.directions            
            
            #Get rid of OOB map positions
            t = t[jnp.where((t[:,0] >= 0) & (t[:,1] >= 0) & (t[:,0] < self.mapsize[0]) & (t[:,1] < self.mapsize[1]))]

            #Using sum of probabilities. We could end up in a situation of p(ship in tile) > 1, but we don't care
            nw = jnp.add.at(nw, (t[:,0], t[:,1]), self.getProbs(t,v,astroids[(t[:,0], t[:,1])]), inplace=False)
        
        return nw

    def learn(self, shipPositions):             

        #Place ship at indices
        self.map =jnp.zeros(self.mapsize).at[shipPositions[:,0],shipPositions[:,1]].add(1)

    def predict(self, astroidPredictions):

        #In case we wan't to keep the current map in memory
        map = self.map.clone()

        l = []
        l.append(map)       

        for i in range(self.horizon):
            map = self.probDistribute(map,astroidPredictions[i])           
            l.append(map) 
        return jnp.array(l)
    


    