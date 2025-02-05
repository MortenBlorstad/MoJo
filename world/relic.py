from world.base_component import base_component
from world.utils import printmap, fromObs, fromObsFiltered, fromObsFilteredSwap

#import jax
import jax.numpy as jnp

class TrackMap():

    def __init__(self, symmetric = False):
        super().__init__()
        self.symmetric = symmetric
        self.reset()

    def reset(self):
        self.values = jnp.empty((0,2),dtype=jnp.int16)

    #Add observation to jnp array. Implemented as __call__ method
    def __call__(self, obs1):

        if self.symmetric:
            obs2 = obs1.copy()
            obs2 = obs2.at[:,0].set(23 - obs1[:,1])
            obs2 = obs2.at[:,1].set(23 - obs1[:,0])

            #Keep jnp array of the relic positions seen so far
            self.values = jnp.unique(jnp.concatenate((self.values, obs1, obs2), axis=0),axis=0)
        else:
            self.values = jnp.unique(jnp.concatenate((self.values, obs1), axis=0),axis=0)

    #Returns jnp.array of values that are in 'positions' but not in 'self.values'
    def candidates(self,positions):
        dims = jnp.maximum(self.values.max(0),positions.max(0))+1
        return positions[~jnp.isin(jnp.ravel_multi_index(positions.T,dims),jnp.ravel_multi_index(self.values.T,dims))]


class Relics(base_component):

    def __init__(self, horizon, team_id):
        super().__init__()
        self.horizon = horizon        
        self.mapsize = (24,24)
        self.seenRelics = TrackMap(symmetric=True)
        self.emptyTiles = TrackMap()
        self.team_id = team_id
        self.numRelicNodes = 0


    def learn(self, t, relicPositions, relicMask, points, shipPos):        
       
        #Current time step
        t = t[0]

        #Update list of observed relicnodes : Filter empty values & swap y/x        
        self.seenRelics(fromObsFilteredSwap(relicPositions))

        #Keep track of how many visible relic nodes there are        
        relicCount = jnp.where(fromObs(relicMask))[0].shape[0]

        if relicCount > self.numRelicNodes:
            print("Wow, num relic nodes changed to ", relicCount, "at timestep",t)
            self.numRelicNodes = relicCount

            #Reset the self.emptyTiles TrackMap here...
            
        if self.numRelicNodes > 0:
            shipPos = fromObsFilteredSwap(shipPos)
            if points == 0:
                #We scored zero, so no relic tile was visited
                print("We have relics at time ",t,", but no points scored.",sep='')
                self.emptyTiles(shipPos)
            else:
                print("We have relics at time ",t,". We scored ",points," points",sep='')
                print()
                print("Empty tiles")
                print(self.emptyTiles.values)
                print()
                print("Ship positions")
                print(shipPos)
                print()
                print("Candidates (instersect)")
                cand = self.emptyTiles.candidates(shipPos)
                print(cand)
                print()

                #for r in self.seenRelics.values:
                r = self.seenRelics.values[0]
                print("Relic looks like this")
                print(r)
                fcand = cand[jnp.where(
                    (cand[:,0] >= r[0]-2) & 
                    (cand[:,0] <= r[0]+2) &
                    (cand[:,1] >= r[1]-2) & 
                    (cand[:,1] <= r[1]+2)
                )]
                print("filtered candidates")
                print(fcand)
                
        

            

    def predict(self):

        #Add probabilities around the relic node
        luxmap = jnp.zeros(self.mapsize)
        for r in self.seenRelics.values:
            luxmap = luxmap.at[r[0]-2:r[0]+3,r[1]-2:r[1]+3].add(1/25)        
        
        #luxmap = jnp.ones(self.mapsize) * 1/(24**2)
        #printmap(luxmap)
        #print(luxmap)

