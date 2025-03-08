from world.base_component import base_component
from world.utils import reduce, symmetric, pointreduce
from world.relictools import EquationSet
import jax.numpy as jnp
import numpy as np

#Remove tiles marked as empty from the ship positions
def removeEmpty(positions:jnp.ndarray, values:jnp.ndarray):
    if len(values) == 0:
        return positions
    return reduce(positions, values)

#Compute intersection between ship positions and possible relic tile positions
#ToDo: Array of unique relic tile positions could be computed only @ relic node discovery time
def relicCandidates(positions:jnp.ndarray, values:list):
    l = []
    for r in values:
        l.append(
            positions[jnp.where(
                (positions[:,0] >= r[0]-2) & 
                (positions[:,0] <= r[0]+2) &
                (positions[:,1] >= r[1]-2) & 
                (positions[:,1] <= r[1]+2)
            )]
        )                
    return jnp.unique(jnp.concatenate(l),axis=0)

#Wrapper around list for handling symmetric insertion
class SymList():

    #Constructor
    def __init__(self):
        super().__init__()
        self.reset()


    #Implement a reset function for removing all values
    def reset(self):        
        #List of the stuff we are storing
        self.values = [] 

    #Process symmetric insert. Implemented as __call__ method
    def __call__(self, entries):
        
        #Keep track of changes
        changed = False

        #Convert to list
        if isinstance(entries, (jnp.ndarray, np.ndarray)):
            entries = entries.tolist()

        #Inserts
        for el in entries:                              #Iterate observations
            if not el in self.values:                   #Check if already present
                self.values.append(el)                  #Add observation to list
                self.values.append(symmetric(el))       #Add symetric observation to list
                changed = True
        return changed

    #Implement len(...)
    def __len__(self):
        return len(self.values)
    
    #Values as a jnp.array
    def tojnp(self):
        return jnp.array(self.values)
    
    #item in collection?
    def has(self,object):
        return object in self.values    


class Relics(base_component):

    def __init__(self):
        super().__init__()

        #Standard Lux stuff     
        self.mapsize = (24,24)

        #Relics time
        self.nodes = SymList()                  #Wrapper around a list of relic nodes that we have seen
        self.empty = SymList()                  #Wrapper around a list of tiles that are confirmed empty        
        self.relics = SymList()                 #Wrapper around a list of tiles that are confirmed relic tiles
        self.eqset = EquationSet(self.mapsize)  #Implementation of filter/solver of ambiguous observations

        #Initial map        
        self.map = self.eqset.compute(self.empty,self.relics)

    def learn(self, relicPositions, points, shipPos):
       
        points = int(points)

        flagCompute = False
        flagFilter = False

        #Update list of observed relicnodes : Filter empty values & swap y/x            
        if self.nodes(relicPositions):            
            self.empty.reset()
            self.eqset.reset()            
            flagCompute = True
        
        #If no points
        if points == 0:
            #We scored zero, so no relic tile was visited. Tiles are empty.  
            if self.empty(shipPos):        #Add to collection
                flagFilter = True               #Notify filtering is required                

        #Hey, we got some points
        else:
            #Any relic tiles included in this observation?
            if len(self.relics) > 0:

                #Remove these positions from the working list                
                shipPos,reduction = pointreduce(shipPos,self.relics.tojnp())
                points -= reduction

            #Have all points been accounted for?                
            if points > 0:

                #Remove (observed) empty tiles from the working list. (We know these are not causing points)                
                shipPos = removeEmpty(shipPos, self.empty.tojnp())
                
                #Have we observed any relic tiles?
                if len(self.nodes) > 0:
                    #Compute the intersection between possible relic tiles and current working list                    
                    shipPos = relicCandidates(shipPos, self.nodes.values)

                #If shipPos has a single tile we are certain. Add to list of relic tiles                    
                if(len(shipPos) == points):
                    if self.relics(shipPos):
                        flagFilter = True   #Notify filtering is required
                        flagCompute = True  #Notify recompute is required
                                
                #Number of possible tiles for the points scored > 1. Keep track of theese ambiguous observations                    
                else:
                    self.eqset.add([shipPos.tolist(),points])
                    flagCompute = True #Notify recompute is required
            else:
                #All points have been accounted for, and there are tiles left in the observation. These must be empty.
                if self.empty(shipPos):        #Add to collection
                    flagFilter = True               #Notify filtering is required

        #Check if we need to filter equations because of new relics or empty tiles 
        if flagFilter:
            flagCompute = self.eqset.filter(self.empty,self.relics) or flagCompute
        
        #To we need to recompute the map?
        if flagCompute:            
            self.map = self.eqset.compute(self.empty, self.relics)


    def predict(self):        
        return self.map[jnp.newaxis,:]