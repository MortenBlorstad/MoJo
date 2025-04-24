
import torch

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

        # eps_clip = config['eps_clip']          # clip parameter for PPO
        # gamma = config['gamma']            # discount factor

        # lr_actor = config['lr_actor']      # learning rate for actor network
        # lr_critic = config['lr_critic']     # learning rate for critic network

        
        
        # lr_actor = config['lr_actor']
        # lr_critic = config['lr_critic']
        # state_dim = config.get('state_dim', None)  # Can be inferred/set elsewhere
        # action_dim = config['action_dim']
        # K_epochs = config['K_epochs']
        # action_std = config['action_std']
        # has_continuous_action_space = config['has_continuous_action_space']
        # image_size = tuple(config['image_size'])
        #####################################################
        
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

    def act(self, step, obs, remainingOverageTime: int = 60):
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




        
