"""
Relic tracking and prediction module.
This module handles the tracking and prediction of relic positions on the game map.
"""
from universe.base_component import base_component
from universe.utils import reduce, symmetric, pointreduce
from universe.relictools import EquationSet
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Optional, Any, Union

#Remove tiles marked as empty from the ship positions
def removeEmpty(positions:jnp.ndarray, values:jnp.ndarray) -> jnp.ndarray:
    """
    Remove tiles marked as empty from ship positions.
    
    Args:
        positions: Array of ship positions
        values: Array of empty positions to remove
        
    Returns:
        Filtered array of ship positions
    """
    if len(values) == 0:
        return positions
    return reduce(positions, values)

#Compute intersection between ship positions and possible relic tile positions
#ToDo: Array of unique relic tile positions could be computed only @ relic node discovery time
def relicCandidates(positions: jnp.ndarray, values: List[List[int]]) -> jnp.ndarray:
    """
    Compute intersection between ship positions and possible relic tile positions.
    
    Args:
        positions: Array of ship positions
        values: List of possible relic positions
        
    Returns:
        Array of candidate relic positions
    """
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
    """
    Wrapper around list for handling symmetric insertion.
    
    This class maintains a list of positions and automatically adds
    their symmetric counterparts when new positions are added.
    """
    #Constructor
    def __init__(self):
        """Initialize the symmetric list."""
        super().__init__()
        self.reset()


    #Implement a reset function for removing all values
    def reset(self) -> None:        
        """Reset the list to empty state."""        
        #List of the stuff we are storing
        self.values = [] 

    #Process symmetric insert. Implemented as __call__ method
    def __call__(self, entries: Union[jnp.ndarray, np.ndarray, List[List[int]]]) -> bool:
        """
        Add entries to the list with their symmetric counterparts.
        
        Args:
            entries: Positions to add
            
        Returns:
            True if new entries were added, False otherwise
        """
        
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
    def __len__(self) -> int:
        """Get the number of values in the list."""
        return len(self.values)
    
    #Values as a jnp.array
    def tojnp(self) -> jnp.ndarray:
        """Convert the list to a JAX numpy array."""
        return jnp.array(self.values)
    
    #item in collection?
    def has(self, object: List[int]) -> bool:
        """Check if a position is in the list."""
        return object in self.values    


class Relics(base_component):
    """
    Relic tracking and prediction component.
    
    This class handles the tracking and prediction of relic positions
    on the game map, including probability calculations and position updates.
    """
    def __init__(self) -> None:
        """Initialize the relic tracker."""
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

    def learn(self, relicPositions: jnp.ndarray, points: int, shipPos: jnp.ndarray) -> None:
        """
        Update relic knowledge based on new observations.
        
        Args:
            relicPositions: Observed relic positions
            points: Points scored in this step
            shipPos: Current ship positions
        """
       
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
                    if len(shipPos) > 0:  # Only add equation if we have positions to consider
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


    def predict(self) -> jnp.ndarray:        
        """
        Get the current relic probability map.
        
        Returns:
            Probability map for relic positions
        """       
        return self.map[jnp.newaxis,:]