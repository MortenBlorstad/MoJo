import json
from typing import Dict
import sys
import os
from argparse import Namespace

import numpy as np

from agent import Agent
# from lux.config import EnvConfig
from utils.kit import from_json
### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict() # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()

def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    obs = observation.obs
    if type(obs) == str:
        obs = json.loads(obs)
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = Agent(player, configurations["env_cfg"])
    agent = agent_dict[player]
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    return dict(action=actions.tolist())
if __name__ == "__main__":

    def dumpfile(seed,obsNo,obs):
        path = "../world/replay/{seed}/obs_{obs}".format(seed = seed, obs = obsNo)        
        f = open(path, "w")
        f.write(obs)
        f.close()

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    step = 0
    player_id = 0
    env_cfg = None
    i = 0


    dir_path = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd()
    
    #print("dir_path",dir_path, file=sys.stderr)
    #print("cwd",cwd, file=sys.stderr)


    NUM_OBS = 75
    SEED = 223344

    while True:
        inputs = read_input()
        raw_input = json.loads(inputs)
        observation = Namespace(**dict(step=raw_input["step"], obs=raw_input["obs"], remainingOverageTime=raw_input["remainingOverageTime"], player=raw_input["player"], info=raw_input["info"]))
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        if i < NUM_OBS:
            if(player_id == 'player_0'):
                dumpfile(SEED,i,inputs)                
        i += 1

        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        # send actions to engine
        print(json.dumps(actions))