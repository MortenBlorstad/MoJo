
from world.universe import Universe

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
        image_size = (26,24,24)
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
        return self.ppo_agent.act(state)
    
    def select_action(self, step, obs, remainingOverageTime: int = 60):
        state = self.universe.predict(obs)
        print(f"Feature extractor output shape: {state.shape}")
        actions = self.ppo_agent.select_action(state)
        units_inplay = self.universe.units_inplay
        actions[~units_inplay] = (0,0,0)
        return actions
    
    def learn(self):
        self.ppo_agent.update()



        
