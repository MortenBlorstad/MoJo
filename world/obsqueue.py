"""
Observation queue module for managing historical observations.
This module provides a queue-like structure for storing and accessing past observations.
"""

from typing import List, Any, Union, Optional

class ObservationQueue:
    """
    Queue-like structure for managing historical observations.
    
    This class maintains a fixed-length queue of observations and provides
    methods for accessing them in various ways.
    """
    
    def __init__(self, length: int):
        """
        Initialize the observation queue.
        
        Args:
            length: Maximum number of observations to store
        """
        #History of observations
        self.obsQueueLen = length
        self.obsQueue = []

    #Add to observation queue. Implemented as __call__ method
    def __call__(self, observation: Any) -> None:
        """
        Add an observation to the queue.
        
        Args:
            observation: Observation to add
        """
        self.obsQueue.append(self.format(observation))
        if(len(self.obsQueue) > self.obsQueueLen):
            self.obsQueue.pop(0)

    #Put formatting of observation here
    def format(self, observation: Any) -> Any:
        """
        Format an observation before storing.
        
        Args:
            observation: Observation to format
            
        Returns:
            Formatted observation
        """
        #Format observation here if needed        
        return observation
    
    #Ugly code. Could probably be replaced with 'reduce(...)'
    def getitem(self, el: Any, keys: Union[str, List[str]]) -> Any:  
        """
        Get a value from an observation using nested keys.
        
        Args:
            el: Observation to access
            keys: Key or list of keys to access
            
        Returns:
            Value at the specified key path
        """
        if not isinstance(keys, list):
            return el[keys]
        else:           
            if(len(keys) == 1):
                return el[keys[0]]
            elif(len(keys) == 2):
                return el[keys[0]][keys[1]]
            elif(len(keys) == 3):
                return el[keys[0]][keys[1]][keys[2]]
            else:
                raise Exception("That many keys is not supported")    

    #Pull observations[keys] from queue:
    #Example:
        #To get the last 3 unit positions of team 0:
        #self.obsQueue.LastN(['units','position',0], 3)

    #Get all observations
    def All(self, keys: Union[str, List[str]]) -> List[Any]:
        """
        Get values from all observations.
        
        Args:
            keys: Key or list of keys to access
            
        Returns:
            List of values from all observations
        """
        return [self.getitem(el, keys) for el in self.obsQueue] 

    #Get the last observation
    def Last(self, keys: Union[str, List[str]]) -> List[Any]:
        """
        Get values from the most recent observation.
        
        Args:
            keys: Key or list of keys to access
            
        Returns:
            List of values from the last observation
        """
        return [self.getitem(el, keys) for el in self.obsQueue[-1:]]

    #Get the last N observations
    def LastN(self, keys: Union[str, List[str]], N: int) -> List[Any]:
        """
        Get values from the last N observations.
        
        Args:
            keys: Key or list of keys to access
            N: Number of observations to get
            
        Returns:
            List of values from the last N observations
        """
        return [self.getitem(el, keys) for el in self.obsQueue[-N:]]
