import yaml
import json
import sys
import os
from pathlib import Path
class Config():
    def __init__(self):
        self.parent_path = Path(__file__).parent.resolve()
        self.global_yaml_file = str(self.parent_path  / "config.yml")  
        
    def __Defaults(self):

        #Write the MoJo default configuration to file.
        #Change defaults here if needded and just call WriteDefaults()

        return {
            "Trainer" : {
                "logepisodes"       : True,                                             #Should we log episodes?                
                "episodelogdir": "MoJo/hierarchical/data/episodes/",  # Use Path.home()
                "modelSaveFrequency": 3,                                               #Save all models every modelSaveFrequency games
            },

            "Director" : {
                "usewandb"          : True,                                            #Use WANDB?
                "TimeSteps_K"       : 8,                                                #Director picks new goals every K time steps. Using same value as paper
                "TimeSteps_E"       : 16,                                               #Update  for PPO. Using same value as paper

                "Worldmodel" : {                    
                    "modelfile"     : "MoJo/hierarchical/weights/worldmodel.pt",
                    "image_size": [41, 24, 24],
                    "scalar_size": 6,
                    "latent_dim": 1024,                                                 # Deterministic hidden state size
                    "hidden_dim": 512,                                                  # Internal processing size
                    "stoch": 32,                                                        #  Stochastic latent state size
                    "num_actions": 6,                                                   # acton values 6  
                    "num_units": 16,                                                    # Number of units to make actions
                    "discrete_actions": 16,                                             # Discrete actions
                    "batch_size": 32,                                                    # Batch size
                    "model_lr": float(0.00001),                                                 # Learning rate
                    "memory_capacity": 1000,                                            # Replay buffer size
                    "memory_sequence_length": 4,                                        # sequence length in replay buffer
                    "step_embedding_dim": 64,                                           # Step embedding size
                    "scalar_dim": 6,                                                    # Scalar size
                    "shared": False,
                    "temp_post" : True,
                    "temp_post": True,
                    "std_act": 'sigmoid2',
                    "unimix_ratio": 0.01,
                    "value_head": 'symlog_disc',
                    "reward_head": 'symlog_disc',
                    "reward_layers": 3,
                    "units": 640,
                    "cont_layers": 3,
                    "value_layers": 3,
                    "actor_layers": 3,
                    "kl_free": 1.0,
                    "cont_scale": 1.0,
                    "dyn_scale": 0.5,
                    "rep_scale": 0.1,
                    "opt_eps": 1e-8,
                    "grad_clip": 1000,
                    "opt": 'adam',
                    "reward_scale": 1.0,
                    "weight_decay": 0.0,
                    "tau" : 0.005,
                    "precision": 32,
                    "cont_stoch_size": 32,
                    "grad_heads": ['reward', 'cont'],
                    "initial": 'learned',
                    "nomlr": False,
                    "nosimsr": False,

                    # MBR
                    "mask_ratio" : 0.5,
                    "patch_size": 10,
                    "block_size": 4,
                },

                "Goalmodel" : {
                    "datapath"      : str(self.parent_path / "data/goalmodel/"),
                    "modelfile"     : str(self.parent_path / "weights/goalmodel.pth"),
                    "input_dim"     : 1024,         #State is output from world model
                    "hid1_dim"      : 256,          #Hidden 1
                    "hid2_dim"      : 128,          #Hidden 2
                    "latent_dim"    : 8,            #Goalmodel latent space is 8 in the paper
                    "lr"            : 0.00001          #Learning rate for Goal VAE
                },                
                
                "Manager" : {
                    "modelfile"     :str( self.parent_path / "weights/manager.pth"),
                    "eps_clip"      : 0.2,          #Clip parameter for PPO
                    "gamma"         : 0.99,         #Discount factor
                    "lr_actor"      : 0.00003,      #Learning rate for actor network
                    "lr_critic"     : 0.0001,       #Learning rate for critic network
                    "state_dim"     : 1024,         #State is output from world model
                    "action_dim"    : 8,            #Manager selects a goals in 'goal latent space'. Must match Goalmodel.
                    "K_epochs"      : 4,            #PPO epochs
                    "action_std"    : 0.5,          #Initial action std                    
                    "cntns_actn_spc": True,         #Use contionous action space?
                    "behaviors"     : 16            #Number of behaviors workers. Should match max num ships
                },

                "Worker" : {
                    "modelfile"     : str(self.parent_path / "weights/worker.pth"),
                    "eps_clip"      : 0.2,          #Clip parameter for PPO
                    "gamma"         : 0.99,         #Discount factor
                    "lr_actor"      : 0.00003,      #Learning rate for actor network
                    "lr_critic"     : 0.0001,       #Learning rate for critic network
                    "state_dim"     : 2*1024 + 32,  #State is conatitnation of: WorldModel (1024) + Decoded(Goal latent) (1024) + Position/Energy as OneHot (16+16)
                    "action_dim"    : 6,            #Actions = [Still, Up, Right, Down, Left, Shoot]
                    "K_epochs"      : 4,            #PPO epochs
                    "action_std"    : 0.5,          #Initial action std
                    "cntns_actn_spc": False,        #Use contionous action space?
                    "behaviors"     : 16            #Number of behaviors workers. Should match max num ships
                    
                }       
            }
    }

    def WriteDefaults(self):
        with open(self.global_yaml_file, 'w') as json_file:
            json.dump(self.__Defaults(), json_file)

    def Get(self,section):
        with open(self.global_yaml_file, 'r') as file:
            cfg = yaml.safe_load(file)
        return cfg[section]
