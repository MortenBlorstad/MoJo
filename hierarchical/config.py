import yaml
import json

class Config():
    def __init__(self):
        self.global_yaml_file = "/home/jorgen/MoJo/hierarchical/config.yml"
    

    def __Defaults(self):

        #Write the MoJo default configuration to file.
        #Change defaults here if needded and just call WriteDefaults()

        return {
            "Director" : {
                "Worldmodel" : {
                    "datapath"      : "/home/jorgen/MoJo/hierarchical/data/worldmodel/",
                    "modelfile"     : "/home/jorgen/MoJo/hierarchical/weights/worldmodel.pth",
                    "input_dim"     : 25*24*24,
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
            }
    }

    def WriteDefaults(self):
        with open(self.global_yaml_file, 'w') as json_file:
            json.dump(self.__Defaults(), json_file)

    def Get(self,section):
        with open(self.global_yaml_file, 'r') as file:
            cfg = yaml.safe_load(file)
        return cfg[section]