import numpy as np
import pickle
from dataclasses import dataclass

infile =    './../MoJo/world/world.pkl'
outfile =   './../MoJo/world/npworld.pkl'

def loadworld(file):
    with open(file, 'rb') as f:
        return pickle.load(f)    

w = loadworld(infile)
print(w)

@dataclass
class MapTile:
    energy: int
    """Energy of the tile, generated via energy_nodes and energy_node_fns"""
    tile_type: int
    """Type of the tile"""


@dataclass
class NPState:

    relic_nodes: np.ndarray
    relic_nodes_mask: np.ndarray
    relic_node_configs: np.ndarray
    relic_nodes_map_weights: np.ndarray

    map_features_energy: np.ndarray
    map_features_tile_type: np.ndarray

    energy_nodes:  np.ndarray
    energy_node_fns: np.ndarray
    energy_nodes_mask: np.ndarray
   

npw = NPState(
    np.array(w.relic_nodes),
    np.array(w.relic_nodes_mask),
    np.array(w.relic_node_configs),
    np.array(w.relic_nodes_map_weights),

    np.array(w.map_features.energy),
    np.array(w.map_features.tile_type),

    np.array(w.energy_nodes),
    np.array(w.energy_node_fns),
    np.array(w.energy_nodes_mask)
)

with open(outfile, 'wb') as f:
    pickle.dump(npw, f)

print("Saved world")

print("relic_nodes")
print(w.relic_nodes)
#print("")

#print("relic_node_configs")
#print(w.relic_node_configs)
#print("")

#print("relic_nodes_mask")
#print(w.relic_nodes_mask)
#print("")

#print("relic_nodes_map_weights")
#print(w.relic_nodes_map_weights)
#print("")

#print(w.map_features)
#a = np.array(w.map_features)
#print(type(w.map_features))

#print(w.energy_node_fns)