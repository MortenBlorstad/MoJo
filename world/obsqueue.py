class ObservationQueue():

    def __init__(self, length):
        
        #History of observations
        self.obsQueueLen = length
        self.obsQueue = []

    #Add to observation queue. Implemented as __call__ method
    def __call__(self, observation):
        self.obsQueue.append(observation)
        if(len(self.obsQueue) > self.obsQueueLen):
            self.obsQueue.pop()

    def All(self,variable):
        return [el[variable] for el in self.obsQueue] 

    def Last(self, variable):
        return [el[variable] for el in self.obsQueue[-1:]]

    def LastN(self, variable, N):
        return [el[variable] for el in self.obsQueue[-N:]]


    #In case we choose to implement this as a EnvObs struct instead of dictionaries
    from flax import struct
    from luxai_s3.state import EnvObs
    
    def toEnvObs(firstObs):
        return EnvObs(
            firstObs['obs']['units'],
            firstObs['obs']['units_mask'],
            firstObs['obs']['sensor_mask'],
            firstObs['obs']['map_features'],
            firstObs['obs']['relic_nodes'],
            firstObs['obs']['relic_nodes_mask'],
            firstObs['obs']['team_points'],
            firstObs['obs']['team_wins'],
            firstObs['obs']['steps'],
            firstObs['obs']['match_steps']
        )