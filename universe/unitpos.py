"""
Unit position tracking and prediction module.
This module handles the tracking and prediction of unit positions on the game map.
"""
from universe.base_component import base_component
import numpy as np
from typing import Dict, List, Tuple, Optional
class Unitpos(base_component):
    """
    Unit position tracking and prediction component.
    
    This class handles the tracking and prediction of unit positions on the game map,
    including probability distribution calculations for unit movements.
    """
    def __init__(self, horizon):
        """
        Initialize the unit position tracker.
        
        Args:
            horizon: Number of steps to predict ahead
        """
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
    def getProbsInner(self,possible: np.ndarray, v: float, astroids: np.ndarray) -> np.ndarray:
        """
        Calculate probability distribution for unit movements.
        
        Args:
            possible: Possible movement positions
            v: Base probability value
            astroids: Asteroid probabilities for each position
            
        Returns:
            Array of movement probabilities
        """
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
    def getProbs(self,possible: np.ndarray, v: float, astroids: np.ndarray) -> np.ndarray:
        """
        Get cached or calculate new probability distribution.
        
        Args:
            possible: Possible movement positions
            v: Base probability value
            astroids: Asteroid probabilities for each position
            
        Returns:
            Array of movement probabilities
        """
        key = (possible.tobytes(),v.tobytes(),astroids.tobytes())
        if key not in self.probDict:
            v = self.getProbsInner(possible,v,astroids)
            self.probDict[key] = v
            return v
        return self.probDict[key]

    #Distribute probabilities
    def probDistribute(self, lastprobmap: np.ndarray, astroids: np.ndarray) -> np.ndarray:
        """
        Distribute probabilities across the map for the next step.
        
        Args:
            lastprobmap: Previous probability map
            astroids: Asteroid probabilities for each position
            
        Returns:
            New probability distribution map
        """
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

    def learn(self, shipPositions: np.ndarray) -> None:             
        """
        Update the current unit positions.
        
        Args:
            shipPositions: Array of current ship positions
        """
        #Place ship at indices
        self.map = np.zeros(self.mapsize)
        self.map[(shipPositions[:,0],shipPositions[:,1])]+=1

    def predict(self, astroidPredictions: np.ndarray) -> np.ndarray:
        """
        Predict unit positions for the next horizon steps.
        
        Args:
            astroidPredictions: Predicted asteroid positions for each step
            
        Returns:
            Array of probability maps for each future step
        """
        #In case we wan't to keep the current map in memory
        map = np.copy(self.map)

        l = []
        l.append(map)       

        for i in range(self.horizon):
            map = self.probDistribute(map,astroidPredictions[i])           
            l.append(map) 
        return np.array(l)
    

