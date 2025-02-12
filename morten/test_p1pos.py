import sys
import os
import json 
import numpy as np
import jax.numpy as jnp
mojo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, mojo_path)  # Ensure it is searched first

from morten.utils.plotting import plot_state_comparison
from agents.lux.kit import to_json,from_json
from world.unitpos import Unitpos
from world.obsqueue import ObservationQueue
from world.obs_to_state import State

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

#p1pos.map = jnp.array(State(observation,"player_1").player_units_count)

#print(p1pos.map == State(observation,"player_1").player_units_count.T)
#print(State(observation,"player_1").player_units_count)

nebula, astroid , pos = predictions
astroid_t = [a.T for a in astroid[1:]]
p = p1pos.predict(astroid[1:], debug=False)

# mask = np.where(astroid[-1]==1)

# print(pos[-1].T[mask])
# print(np.where((astroid[-1]==1) & (pos[-1].T>0)))
print(np.where((astroid[-1]==1) & (p[-1].T>0)))

import matplotlib.pyplot as plt
plots_dir = "MoJo/morten/plots"
os.makedirs(plots_dir, exist_ok=True)
fig, axs = plt.subplots(ncols=2)
axs[0].imshow(astroid[-1])
p[-1] = jnp.where(p[-1]==0,jnp.nan,p[-1])
axs[1].imshow(p[-1].T)
plt.tight_layout()
plot_filename = os.path.join(plots_dir, f"plot_{18}.png")
plt.savefig(plot_filename)
plt.close(fig) 
# print(solutions[-1][:,:,2])
# observeable_tiles = np.zeros((24,24))
# plot_state_comparison(18, solutions, predictions, observeable_tiles)

