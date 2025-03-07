import yaml
import json

class Config():
    def __init__(self):
        self.global_yaml_file = "/home/jorgen/MoJo/hierarchical/config.yml"
    

    def __Defaults(self):

        #Write the MoJo default configuration to file.
        #Change defaults here if needded and just call WriteDefaults()

        return {
<<<<<<< HEAD
            "Director" : {
=======
            "Trainer" : {
                "logepisodes"       : True,                                             #Should we log episodes?                
                "episodelogdir"     : "/home/jorgen/MoJo/hierarchical/data/episodes/",  #Directory for logging episodes
                "modelSaveFrequency": 10,                                               #Save all models every modelSaveFrequency games
            },

            "Director" : {
                "usewandb"          : False,        #Use WANDB?
                "TimeSteps_K"       : 8,            #Director picks new goals every K time steps. Using same value as paper
                "TimeSteps_E"       : 16,           #Update  for PPO. Using same value as paper

>>>>>>> 7515b2ceab11c37e9fed4d289e351ddc4a00fcd7
                "Worldmodel" : {
                    "datapath"      : "/home/jorgen/MoJo/hierarchical/data/worldmodel/",
                    "modelfile"     : "/home/jorgen/MoJo/hierarchical/weights/worldmodel.pth",
                    "input_dim"     : 25*24*24,
<<<<<<< HEAD
                    "hid1_dim"      : 512,
                    "hid2_dim"      : 256,
                    "latent_dim"    : 16,
                    "lr"            : 1e-3
                },
                
                "Manager" : {
                    "eps_clip"      : 0.2,      # clip parameter for PPO
                    "gamma"         : 0.99,     # discount factor
                    "lr_actor"      : 0.00003,  # learning rate for actor network
                    "lr_critic"     : 0.0001,   # learning rate for critic network
                    "state_dim"     : 25*24*24,
                    "action_dim"    : 6, 
                    "K_epochs"      : 4,         
                    "action_std"    : 0.5,      # Initial action std
                    "has_continuous_action_space" : False
                },

                "Worker" : {
                    "eps_clip"      : 0.2,      # clip parameter for PPO
                    "gamma"         : 0.99,     # discount factor
                    "lr_actor"      : 0.00003,  # learning rate for actor network
                    "lr_critic"     : 0.0001,   # learning rate for critic network
                    "state_dim"     : 25*24*24,
                    "action_dim"    : 6, 
                    "K_epochs"      : 4,         
                    "action_std"    : 0.5,      # Initial action std
                    "has_continuous_action_space" : False
                }       
            },
            "MortensStuffHere": {
                "A" : "This is A",
                "B" : "This is B"
=======
                    "hid1_dim"      : 4096,
                    "hid2_dim"      : 2048,
                    "latent_dim"    : 1024,
                    "lr"            : 1e-3
                },

                "Goalmodel" : {
                    "datapath"      : "/home/jorgen/MoJo/hierarchical/data/goalmodel/",
                    "modelfile"     : "/home/jorgen/MoJo/hierarchical/weights/goalmodel.pth",
                    "input_dim"     : 1024,         #State is output from world model
                    "hid1_dim"      : 256,          #Hidden 1
                    "hid2_dim"      : 128,          #Hidden 2
                    "latent_dim"    : 8,            #Goalmodel latent space is 8 in the paper
                    "lr"            : 1e-3          #Learning rate for Goal VAE
                },                
                
                "Manager" : {
                    "modelfile"     : "/home/jorgen/MoJo/hierarchical/weights/manager.pth",
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
                    "modelfile"     : "/home/jorgen/MoJo/hierarchical/weights/worker.pth",
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
>>>>>>> 7515b2ceab11c37e9fed4d289e351ddc4a00fcd7
            }
    }

    def WriteDefaults(self):
        with open(self.global_yaml_file, 'w') as json_file:
            json.dump(self.__Defaults(), json_file)

    def Get(self,section):
        with open(self.global_yaml_file, 'r') as file:
            cfg = yaml.safe_load(file)
        return cfg[section]