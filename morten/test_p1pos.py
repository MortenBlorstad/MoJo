import sys
import os
import json 
import numpy as np
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from morten.utils.plotting import plot_state_comparison
from agents.lux.kit import to_json,from_json
from world.unitpos import Unitpos
from world.obsqueue import ObservationQueue

seed = 223344
with open(f"MoJo/morten/test_dump_{seed}.json", "r") as infile:
    test_dump = from_json(json.load(infile))  

horizont = 3
obsQueue = ObservationQueue(10)
p1pos = Unitpos(horizont)

observation = test_dump["obs"]
predictions = test_dump["predictions"]
solutions = test_dump["solutions"] 

team_id = 1

obsQueue(observation)
p1pos.learn(obsQueue.Last(['units','position',team_id]))

nebula, astroid , pos = predictions

p = p1pos.predict(astroid[1:], debug=False)

observeable_tiles = np.zeros((24,24))
plot_state_comparison(18, solutions, predictions, observeable_tiles)

