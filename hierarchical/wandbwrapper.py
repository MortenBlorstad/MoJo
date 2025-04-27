import wandb
from typing import Any, Dict, List, Union
class WandbWrapper():
    """
    A wrapper class for Weights & Biases (wandb) logging functionality.
    This class provides a simplified interface for logging metrics and tracking experiments.
    
    Attributes:
        _WW_usewandb (bool): Flag indicating whether wandb logging is enabled
        _WW_run (wandb.Run): The active wandb run instance
    """
    
    def __init__(self, usewandb: bool) -> None:
        """
        Initialize the WandbWrapper.
        
        Args:
            usewandb (bool): Whether to enable wandb logging
        """

        self._WW_usewandb = usewandb

        if self._WW_usewandb:
            
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="Complete MoJo",
            )
            self._WW_run = wandb.init()
            
    #Record loss value
    def record(self, propertyName: str, propertyValue: Any) -> None:
        """
        Record a metric value for wandb logging.
        
        Args:
            propertyName (str): Name of the metric to record
            propertyValue (Any): Value to record for the metric
        """
        if self._WW_usewandb:
            x = getattr(self, propertyName)
            if isinstance(x,list):
                x.append(propertyValue)
            else:
                setattr(self, propertyName, propertyValue)


    #Clear everything
    def __clear(self) -> None:
        """
        Clear all recorded metrics.
        Internal method used before logging a new batch of metrics.
        """
        for (k, v) in self.__dict__.items():
            if(k[0:4] == "_WW_"):
                pass
            else:
                if isinstance(v, list):
                    del v[:]
                else:
                    setattr(self, k, 0)

    def report(self) -> None:
        """
        Log all recorded metrics to wandb and clear the internal state.
        For list-type metrics, logs the average value.
        """
        if self._WW_usewandb:
            r:Dict[str, Union[float, int, Dict]] = {}
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

import numpy as np

ww = WandbWrapper(True)
ww.wmloss = 0
ww.goalloss = 0
ww.mgrloss = []
ww.wrkloss = []

d = {'model_loss': np.array(23.500671, dtype=np.float32), 'model_grad_norm': 29.497570037841797, 'reward_loss': np.array(3.4556196, dtype=np.float32), 'cont_loss': np.array(2.7174645e-05, dtype=np.float32), 'mbr_loss': np.array(6.026751e-06, dtype=np.float32), 'simsr_loss': np.array(16.212582, dtype=np.float32), 'kl_free': 1.0, 'dyn_scale': 0.5, 'rep_scale': 0.1, 'dyn_loss': np.array(6.387396, dtype=np.float32), 'rep_loss': np.array(6.387396, dtype=np.float32), 'kl': np.array(6.336668, dtype=np.float32), 'min_reward': np.array(-244.39638, dtype=np.float32), 'max_reward': np.array(-203.41843, dtype=np.float32), 'prior_ent': np.array(68.44803, dtype=np.float32), 'post_ent': np.array(66.625626, dtype=np.float32)}


for i in range(100):

    ww.record("wrkloss",0)
    ww.record("wrkloss",24)
    ww.record("goalloss",15)
    ww.record("mgrloss",-1)
    ww.record("mgrloss",-2)
    ww.record("mgrloss",-3)

    if i % 10 == 0:
        ww.record("wmloss",d)


    ww.report()

'''
