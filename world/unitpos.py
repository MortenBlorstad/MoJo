"""
Unit position tracking and prediction module.
This module handles the tracking and prediction of unit positions on the game map.
"""

from world.base_component import base_component
import numpy as np
from typing import Dict, List, Tuple, Optional

class Unitpos(base_component):
    """
    Unit position tracking and prediction component.
    
    This class handles the tracking and prediction of unit positions on the game map,
    including probability distribution calculations for unit movements.
    """
    
    def __init__(self, horizon: int):
        """
        Initialize the unit position tracker.
        
        Args:
            horizon: Number of steps to predict ahead
        """
        super().__init__()
        self.horizon = horizon        
        self.mapsize = (24, 24)

        # Define possible movement directions (including staying in place)
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

        # Cache for probability calculations
        self.probDict = {}
    
    def getProbsInner(self, possible: np.ndarray, v: float, astroids: np.ndarray) -> np.ndarray:
        """
        Calculate probability distribution for unit movements.
        
        Args:
            possible: Possible movement positions
            v: Base probability value
            astroids: Asteroid probabilities for each position
            
        Returns:
            Array of movement probabilities
        """
        # Create initial uniform probability distribution
        probs = [1/6]*len(possible)
        probs[0] += 1/6*(6-len(possible))        
        probs = np.array(probs)*v

        # Adjust probabilities based on asteroid presence
        for idx, a in enumerate(astroids[1:]):
            if a > 0:                  
                reduction = probs[idx+1]*a  # Probability reduction due to asteroid
                probs[0] += reduction       # Add to probability of not moving
                probs[idx+1] -= reduction   # Remove from probability of moving into asteroid
        return probs

    def getProbs(self, possible: np.ndarray, v: float, astroids: np.ndarray) -> np.ndarray:
        """
        Get cached or calculate new probability distribution.
        
        Args:
            possible: Possible movement positions
            v: Base probability value
            astroids: Asteroid probabilities for each position
            
        Returns:
            Array of movement probabilities
        """
        key = (possible.tobytes(), v.tobytes(), astroids.tobytes())
        if key not in self.probDict:
            v = self.getProbsInner(possible, v, astroids)
            self.probDict[key] = v
            return v
        return self.probDict[key]

    def probDistribute(self, lastprobmap: np.ndarray, astroids: np.ndarray) -> np.ndarray:
        """
        Distribute probabilities across the map for the next step.
        
        Args:
            lastprobmap: Previous probability map
            astroids: Asteroid probabilities for each position
            
        Returns:
            New probability distribution map
        """
        # Get positions with non-zero probability
        idx = np.where(lastprobmap > 0)
        vals = lastprobmap[idx]

        # Initialize new probability map
        nw = np.zeros(self.mapsize)

        # Update probabilities for each position
        for x, y, v in zip(idx[0], idx[1], vals):
            # Get possible new positions
            t = np.array([x,y]) + self.directions            
            
            # Remove out-of-bounds positions
            t = t[np.where((t[:,0] >= 0) & (t[:,1] >= 0) & 
                          (t[:,0] < self.mapsize[0]) & (t[:,1] < self.mapsize[1]))]

            # Update probabilities using sum of probabilities
            nw[(t[:,0], t[:,1])] += self.getProbs(t, v, astroids[(t[:,0], t[:,1])])
        
        return nw

    def learn(self, shipPositions: np.ndarray) -> None:
        """
        Update the current unit positions.
        
        Args:
            shipPositions: Array of current ship positions
        """
        self.map = np.zeros(self.mapsize)
        self.map[(shipPositions[:,0], shipPositions[:,1])] += 1

    def predict(self, astroidPredictions: np.ndarray) -> np.ndarray:
        """
        Predict unit positions for the next horizon steps.
        
        Args:
            astroidPredictions: Predicted asteroid positions for each step
            
        Returns:
            Array of probability maps for each future step
        """
        # Keep current map in memory
        map = np.copy(self.map)

        l = []
        l.append(map)       

        # Predict for each step in horizon
        for i in range(self.horizon):
            map = self.probDistribute(map, astroidPredictions[i])           
            l.append(map) 
        return np.array(l)
    

