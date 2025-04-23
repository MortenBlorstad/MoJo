
import sys
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hierarchical.config import Config
from hierarchical.utils import InitWorker, InitManager

maspshape = (24,24)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config().Get("Director")
missioncontrol =    InitManager(cfg["Manager"])
fleet =             InitWorker(cfg["Worker"])

randstate = np.random.rand(1024)

def posToArray(x,y):
    map = np.zeros((24,24))
    map[x,y] = 1
    return map.flatten()



for t in range(1):
    for worker_id in range(1):    
        a = missioncontrol.select_action(randstate,worker_id)    
        #print(a)    
        missioncontrol.bufferList[worker_id].extrinsic_rewards.append(t%2)  
        missioncontrol.bufferList[worker_id].exploration_rewards.append(1)
        missioncontrol.bufferList[worker_id].is_terminals.append(0)


missioncontrol.update(0)
#missioncontrol.update(1)
'''

randstate = np.random.rand((1024+16)*2)

for t in range(3):

    for ship in range(1):    
        a = fleet.select_action(randstate,ship) 
        print(a)       
        fleet.bufferList[ship].rewards.append(t)    
        fleet.bufferList[ship].is_terminals.append(0)


fleet.update(0)
#fleet.update(1)
'''
