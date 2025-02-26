
import torch
from world.universe import Universe
import numpy as np
class PPOAgent:
        
    def __init__(self, player, env_cfg):
        self.player = player
        self.universe = Universe(player, env_cfg, horizont=3)
        from .networks.ppo import PPO

        ################ PPO hyperparameters ################
        


        eps_clip = 0.2          # clip parameter for PPO
        gamma = 0.99            # discount factor

        lr_actor = 0.00003       # learning rate for actor network
        lr_critic = 0.0001       # learning rate for critic network

        
        state_dim = None
        action_dim = 5 # exclude sap for now
        K_epochs = 4
        action_std = 0.5  # Initial action std
        has_continuous_action_space = False
        image_size = (26, 24, 24)
        #####################################################
        
        self.ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs,
                             eps_clip, has_continuous_action_space, action_std,
                            image_size=image_size)
        
    def append_to_buffer(self, done):
        reward = self.universe.reward
        self.ppo_agent.buffer.rewards.append(reward)
        self.ppo_agent.buffer.is_terminals.append(done)

    def act(self, step, obs, remainingOverageTime: int = 60):
        state = self.universe.predict(obs)
        actions = np.zeros((16, 3), dtype=int)
        one_hot_pos = []
        for worker_idx in range(16):
            available, one_hot,__annotations__  = self.universe.get_one_hot_pos(worker_idx)
            one_hot_pos.append(one_hot)
        
        one_hot_pos = torch.tensor(one_hot_pos).unsqueeze(0)
        actions = self.ppo_agent.act(state, one_hot_pos)
        units_inplay = self.universe.units_inplay
        actions[~units_inplay] = (0, 0, 0)
        return actions
    
    def select_action(self, step, obs, remainingOverageTime: int = 60):
        state = self.universe.predict(obs)
        
        one_hot_pos = np.zeros((1, 16, 24*24))
        for worker_idx in range(16):
            available, one_hot, _  = self.universe.get_one_hot_pos(worker_idx)
            one_hot_pos[0,worker_idx] = one_hot
        actions = self.ppo_agent.select_action(state, one_hot_pos)
        units_inplay = self.universe.units_inplay

        actions[~units_inplay] = (0, 0, 0)
        
        return actions
                
        # state = self.universe.predict(obs)
        # print(f"Feature extractor output shape: {state.shape}")
        # actions = self.ppo_agent.select_action(state)
        # units_inplay = self.universe.units_inplay
        # actions[~units_inplay] = (0, 0, 0)
        # return actions
    
    def learn(self):
        units_inplay = self.universe.units_inplay
        return self.ppo_agent.update(units_inplay)
    
    def save(self, checkpoint_path):
        self.ppo_agent.save(checkpoint_path)
    def load(self, checkpoint_path):
        self.ppo_agent.load(checkpoint_path)




        
