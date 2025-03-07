import wandb

class WandbWrapper():

    def __init__(self, usewandb):

        self._WW_usewandb = usewandb

        if self._WW_usewandb:
            
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="Complete MoJo",

                # track hyperparameters and run metadata
                #config={        
                #    "some_param": "has_been_set"
                #}
            )
            self._WW_run = wandb.init()


    #Record loss value
    def record(self, propertyName, propertyValue):
        if self._WW_usewandb:
            x = getattr(self, propertyName)
            if isinstance(x,list):
                x.append(propertyValue)
            else:
                setattr(self, propertyName, propertyValue)


    #Clear everything
    def __clear(self):
        for (k, v) in self.__dict__.items():
            if(k[0:4] == "_WW_"):
                pass
            else:
                if isinstance(v, list):
                    del v[:]
                else:
                    setattr(self, k, 0)

    def report(self):
        if self._WW_usewandb:
            r = {}
            for (k, v) in self.__dict__.items():
                if(k[0:4] == "_WW_"):
                    pass
                else:
                    if isinstance(v, list):
                        if len(v) == 0:
                            r[k] = 0                    
                        else:
                            r[k] = sum(v) / len(v)
                    elif isinstance(v, dict):
                        r.update(v)
                    else:
                        r[k] = v

            self.__clear()        
            self._WW_run.log(r)
            
'''
#Example

ww = WandbWrapper(True)
ww.wmloss = 0
ww.goalloss = 0
ww.mgrloss = []
ww.wrkloss = []
ww.mstuff = 0


for i in range(3):

    ww.record("wrkloss",0)
    ww.record("wrkloss",24)
    ww.record("goalloss",15)
    ww.record("mgrloss",-1)
    ww.record("mgrloss",-2)
    ww.record("mgrloss",-3)


    d = {
        "val1" : i,
        "val2" : i+5
    }
    ww.record("mstuff",d)


    ww.report()

'''
