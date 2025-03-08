from world.base_component import base_component
import numpy as np

class Unitpos(base_component):

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon        
        self.mapsize = (24,24)

        self.directions = np.array(
            [
                [0, 0],     # Don't move                
                [-1, 0],    # Move up
                [0, 1],     # Move right
                [1, 0],     # Move down
                [0, -1],    # Move left
            ],
            dtype=np.int16,
        )

        self.probDict = {}
    
    #Calculate probability of ending up in a surrounding tile
    def getProbsInner(self,possible,v,astroids):

        #Create the list of probabilities
        probs = [1/6]*len(possible)
        probs[0]+=1/6*(6-len(possible))        
        probs = np.array(probs)*v

        #Update values based on astroid probabilities on a given tile
        for idx, a in enumerate(astroids[1:]):
            if(a > 0):                  
                reduction = probs[idx+1]*a  #How big is the reduction?                               
                probs[0]+=reduction         #Increment probability of not moving
                probs[idx+1]+=reduction     #Decrement probability of moving into astroid tile
        return probs

    #Lookup case in dictionary or create new entry   
    def getProbs(self,possible,v,astroids):

        key = (possible.tobytes(),v.tobytes(),astroids.tobytes())
        if key not in self.probDict:
            v = self.getProbsInner(possible,v,astroids)
            self.probDict[key] = v
            return v
        return self.probDict[key]

    #Distribute probabilities
    def probDistribute(self, lastprobmap, astroids):

        #Get indices of tiles that, with probability > 0, has a ship placed on it
        idx = np.where(lastprobmap > 0)
        vals = lastprobmap[idx]

        #Start with an empty space
        nw = np.zeros(self.mapsize)

        #For the x,y pairs of tiles containing a ship
        for x, y, v in zip(idx[0],idx[1], vals):

            #Get possible new positions (Cardinal directions + no move)
            t = np.array([x,y]) + self.directions            
            
            #Get rid of OOB map positions
            t = t[np.where((t[:,0] >= 0) & (t[:,1] >= 0) & (t[:,0] < self.mapsize[0]) & (t[:,1] < self.mapsize[1]))]

            #Using sum of probabilities. We could end up in a situation of p(ship in tile) > 1, but we don't care            
            nw[(t[:,0], t[:,1])]+=self.getProbs(t,v,astroids[(t[:,0], t[:,1])])
        
        return nw

    def learn(self, shipPositions):             

        #Place ship at indices
        self.map = np.zeros(self.mapsize)
        self.map[(shipPositions[:,0],shipPositions[:,1])]+=1

    def predict(self, astroidPredictions):

        #In case we wan't to keep the current map in memory
        map = np.copy(self.map)

        l = []
        l.append(map)       

        for i in range(self.horizon):
            map = self.probDistribute(map,astroidPredictions[i])           
            l.append(map) 
        return np.array(l)
    

