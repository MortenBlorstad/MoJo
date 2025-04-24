"""
Relic tools module for handling relic-related calculations and equations.
This module provides tools for managing relic equations and solving relic-related problems.
"""

import jax.numpy as jnp
from itertools import product
from world.utils import symmlist
from typing import Dict, List, Tuple, Optional, Any

class EquationSet:
    """
    Class for managing and solving sets of equations related to relic positions.
    
    This class handles the creation, filtering, and solving of equations that
    describe possible relic positions on the game map.
    """
    
    def __init__(self, mapsize: Tuple[int, int]):
        """
        Initialize the equation set.
        
        Args:
            mapsize: Size of the game map (height, width)
        """
        self.mapsize = mapsize
        self.reset()

    def reset(self) -> None:
        """Reset the equation set to empty state."""
        self.equations = []

    def add(self, eq: List[List[List[int]]]) -> None:
        """
        Add a new equation to the set.
        
        Args:
            eq: Equation to add, in the form [[positions], value]
        """
        # Sort the list to ensure unique insertions
        eq = [sorted(eq[0]), eq[1]]

        # Store observation if not already present
        if not eq in self.equations:
            self.equations.append(eq)

    def filter(self, empties: Any, relics: Any) -> bool:
        """
        Filter equations based on confirmed empty and relic positions.
        
        Args:
            empties: Object containing confirmed empty positions
            relics: Object containing confirmed relic positions
            
        Returns:
            True if equations were modified, False otherwise
        """
        # No need to filter if equation set is empty
        if len(self.equations) == 0:
            return False

        # Keep a copy of the equations to check for changes
        old = self.equations.copy()

        # Run filtering
        while True:
            # Keep track of relic/empty tiles changes in this iteration
            change = False

            for i in reversed(range(len(self.equations))):              
                for j in reversed(range(len(self.equations[i][0]))):    

                    if empties.has(self.equations[i][0][j]):              
                        self.equations[i][0].pop(j)                    
                    
                    elif relics.has(self.equations[i][0][j]):            
                        self.equations[i][0].pop(j)                     
                        self.equations[i][1] -= 1                       

                    # Remove equations with zero symbols
                    if len(self.equations[i][0]) == 0: 
                        self.equations.pop(i)

                    # If solution is 0, all symbols must be 0 (empty tiles)
                    elif self.equations[i][1] == 0:
                        change = change or empties(self.equations.pop(i)[0])
                        break

                    # Num symbols match solution - they must all be 1 (relic tiles)                    
                    elif len(self.equations[i][0]) == self.equations[i][1]:
                        change = change or relics(self.equations.pop(i)[0])
                        break

            # Stop if no new relic/empty tiles were discovered
            if not change:            
                return old != self.equations         

    def makeLookupDicts(self) -> None:
        """Create lookup dictionaries for position to index mapping."""
        self.indict = {}
        self.outdict = {}
        
        for eq in self.equations:
            [self.uniqueinsert(v) for v in eq[0]]

    def uniqueinsert(self, el_lst: List[int]) -> None:  
        """
        Insert a position into the lookup dictionaries.
        
        Args:
            el_lst: Position to insert [x, y]
        """
        # Use tuple instead of list so entries can be dict keys                            
        el_tpl = (el_lst[0], el_lst[1])
        if not el_tpl in self.indict:
            self.indict[el_tpl] = len(self.indict) 
            self.outdict[len(self.outdict)] = el_lst

    def row(self, v: List[List[int]]) -> List[int]:
        """
        Create a row vector for the equation matrix.
        
        Args:
            v: List of positions
            
        Returns:
            Binary row vector indicating presence of positions
        """
        r = [0]*len(self.indict)
        for el in v:
            r[self.indict[(el[0], el[1])]] = 1
        return r
    
    def eqSolve(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Solve the system of equations.
        
        Args:
            a: Coefficient matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector with probabilities for each position
        """
        # Get possible bitstrings
        mask = jnp.array([b for b in product([0, 1], repeat=a.shape[1])])
        
        # Compute row-wise sum of 'a AND b' and compare to b matrix
        res = jnp.sum(a[jnp.newaxis,:]&mask[:,jnp.newaxis,:],axis=2) == b

        # Return column-wise mean of valid solutions
        return jnp.mean(mask[jnp.where(jnp.all(res, axis=1))],axis=0)
    
    def compute(self, empties: Any, relics: Any) -> jnp.ndarray:
        """
        Compute the relic probability map.
        
        Args:
            empties: Object containing confirmed empty positions
            relics: Object containing confirmed relic positions
            
        Returns:
            Probability map for relic positions
        """
        # Create map
        relicmap = jnp.zeros(self.mapsize)

        # Set confirmed relics to probability 1
        if len(relics) > 0:
            indices = relics.tojnp()
            relicmap = relicmap.at[(indices[:,0],indices[:,1])].add(1) 

        # If there are equations to solve
        if self.equations:                
            # Create forward/backward lookup dictionaries
            self.makeLookupDicts()

            # Create the a and b matrices
            a = jnp.array([self.row(lst) for lst,_ in self.equations])
            b = jnp.array([val for _,val in self.equations])

            # Create the a and b matrices with proper types
            a = jnp.array([self.row(lst) for lst, _ in self.equations], dtype=jnp.int32)
            b = jnp.array([val for _, val in self.equations], dtype=jnp.int32)
            
            if a.size == 0 or b.size == 0:
                print("Quick fix! Either 'a' or 'b' is empty. Skipping equation solving.")
                if len(self.equations) == 0:
                    print("equations is empty", self.equations)
                if a.dtype != jnp.int32:
                    print("a", a)
                    print("equations", self.equations)
                if b.dtype != jnp.int32:
                    print("b", b)
                    print("equations", self.equations)
                return relicmap            

            # Solve equations
            res = self.eqSolve(a,b)

            # Update confirmed positions
            relics([self.outdict[x] for x in jnp.where(res == 1)[0].tolist()])
            empties([self.outdict[x] for x in jnp.where(res == 0)[0].tolist()])
                    
            # Update ambiguous tiles with calculated probability
            indices = jnp.array(symmlist([self.outdict[x] for x in range(len(res))]))
            relicmap = relicmap.at[(indices[:,0],indices[:,1])].set(jnp.concat([res,res]))
        
        return relicmap
    
    def pp(self) -> None:
        """Print the current equations for debugging."""
        print(self.equations)