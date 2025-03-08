class ObservationQueue():

    def __init__(self, length):
        
        #History of observations
        self.obsQueueLen = length
        self.obsQueue = []

    #Add to observation queue. Implemented as __call__ method
    def __call__(self, observation):
        self.obsQueue.append(self.format(observation))
        if(len(self.obsQueue) > self.obsQueueLen):
            self.obsQueue.pop(0)

    #Put formatting of observation here
    def format(self,observation):
        #Format observation here if needed        
        return observation
    
    #Ugly code. Could probably be replaced with 'reduce(...)'
    def getitem(self,el,keys):  

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
    def All(self,keys):
        return [self.getitem(el,keys) for el in self.obsQueue] 

    #Get the last observation
    def Last(self, keys):
        return [self.getitem(el,keys) for el in self.obsQueue[-1:]]

    #Get the last N observations
    def LastN(self, keys, N):
        return [self.getitem(el,keys) for el in self.obsQueue[-N:]]
