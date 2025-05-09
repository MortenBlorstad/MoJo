"""
The Baseline PPO Agent. Standard PPO that uses representations from universe as input and outputs actions for the 16 ships.
Implementation of the PPO algorithm is based on the code from the github repo https://github.com/nikhilbarhate99/PPO-PyTorch and 
adopted to our use case.
"""
from universe.universe import Universe
import numpy as np
import jax.numpy as jnp
#Filter out possible attacking positions based on zap range and the ships current location
def getMapRange(position, zaprange, probmap):
    position = position.astype(int)
    x_lower = max(0,position[0]-zaprange)         #Avoid x pos < 0
    x_upper = min(23,position[0]+zaprange+1)    #Avoid x pos > map height
    y_lower = max(0,position[1]-zaprange)         #Avoid y pos < 0
    y_upper = min(23,position[1]+zaprange+1)    #Avoid y pos > map width

    #Filter out the probabilities of ships within zap-range
    return probmap[x_lower:x_upper,y_lower:y_upper], x_lower,y_lower
    
#Get the coordinates of the tile with the highest probability of enemy ship given a zap range and a ship location
def getZapCoords(position, zaprange, probmap):
    filteredMap, x_l, y_l = getMapRange(position, zaprange, probmap)    
    x,y = divmod(int(jnp.argmax(filteredMap)),filteredMap.shape[0])
    
    #Add back global indexing
    x+=x_l 
    y+=y_l
    x = min(x,23)
    y = min(y,23)	
    #Return target coordinates
    return (x,y),probmap[(x,y)]


class PPOAgent:
        
    def __init__(self, player, env_cfg, config):
        self.player = player
        self.universe = Universe(player, env_cfg, horizont=1)
        from .networks.ppo import PPO        
        self.ppo_agent = PPO(config)
        
    def append_to_buffer(self, done):
        reward = self.universe.reward
        self.ppo_agent.buffer.rewards.append(reward)
        self.ppo_agent.buffer.is_terminals.append(done)

    
    def handle_sap(self, actions, pos:np.ndarray):
        unit_sap_range = self.universe.unit_sap_range
        probmap = self.universe.zap_options
        sap_actions = np.where(actions[:,0] ==5)[0]
        if len(sap_actions) == 0:
            return actions
        sap_cords = np.zeros((16, 2))
        for sap_action in sap_actions:
            sap_cord, _ = getZapCoords(pos[sap_action], unit_sap_range, probmap )
            sap_cords[sap_action] = sap_cord
        actions[sap_actions,1:] = sap_cords[sap_actions]
        return actions

    def act(self, step:int, obs:dict, remainingOverageTime: int = 60)->np.ndarray:
        """
        Selects an action for the agent based on the current environment observation.
        Agent receives the observation which is processed by the universe and then mapped to an action using the actor network.

        args:
            step: The current step in the game.
            obs: The observation from the environment.
            remainingOverageTime: The remaining time for the agent to make a decision.
        
        Returns:
            np.ndarray: An array of selected actions for each active unit.
        """

        state = self.universe.predict(obs)
        one_hot_pos = np.zeros((1, 16, 24*24))
        coordinates = np.zeros((16, 2))
        for worker_idx in range(16):
            available, one_hot, pos  = self.universe.get_one_hot_pos(worker_idx)
            one_hot_pos[0,worker_idx] = one_hot
            coordinates[worker_idx] = pos
        
        actions = self.ppo_agent.select_action(state, one_hot_pos)
        units_inplay = self.universe.units_inplay
        actions[~units_inplay] = (0, 0, 0)
        actions = self.handle_sap(actions, coordinates)
        return actions
                
    
    def train(self):
        self.ppo_agent.training = True

    def evel(self):
        self.ppo_agent.training = False

    def learn(self):
        units_inplay = self.universe.units_inplay
        return self.ppo_agent.update(units_inplay)
    
    def save(self, checkpoint_path):
        self.ppo_agent.save(checkpoint_path)
    def load(self, checkpoint_path):
        self.ppo_agent.load(checkpoint_path)




        
