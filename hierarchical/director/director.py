import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from universe.universe import Universe
from hierarchical.config import Config
from hierarchical.director.utils import getZapCoordsOnly, InitWorker, InitManager, RunningAverages
from hierarchical.director.vae import VAE
from hierarchical.director.directorlossfuncs import MaxCosineNP as MaxCosine, ExplorationReward
from world_model.world_model import WorldModel
import torch

class Director():    
   
    def __init__(self, player: str, env_cfg, training = False) -> None:

        #Get variables
        self.player = player
        self.u:Universe = Universe(player, env_cfg, horizont=1)        
        self.cfg = Config().Get("Director")   
        self.clockTicks = self.cfg["TimeSteps_K"]
        self.updateFreq = self.cfg["TimeSteps_E"]
        self.numShips:int = 16
        self.training = training

        #Init WANDB for training
        if training:
            from hierarchical.wandbwrapper import WandbWrapper
            #------------------------------------------------------------------
            self.ww = WandbWrapper(self.cfg["usewandb"])
            self.ww.wmloss = 0       #Add worldmodel loss property
            self.ww.goalloss = 0     #Add goalmodel loss property
            self.ww.mgrloss = []     #Add manager loss list (use average)
            self.ww.wrkloss = []     #Add worker loss list (use average)     
           
            #------------------------------------------------------------------        

        #Keep track of running averages
        self.running_averages = RunningAverages()

        #Create a list of 'ShipActions' to keep track of when to update individual policies (per ship (16))  
        self.policies = [self.ShipActions(self,idx) for idx in range(self.numShips)] 
    
        #Keep track of the raw output from Universe so we can train world model
        self.universes = []

        #Load pretrained world model autoencoder
        self.worldmodel = WorldModel(self.cfg["Worldmodel"])
        if os.path.exists(self.cfg["Worldmodel"]["modelfile"]):
            self.worldmodel.load_model(self.cfg["Worldmodel"]["modelfile"])
        

        #Keep track of the output from world model so we can train goal autoencoder
        self.wmstates = []

        #init goal autoencoderf and load weights if exists
        self.goalmodel = VAE.Load(self.cfg["Goalmodel"])
        self.goalstates = []

        #Load the Muli Agent PPO with double critic - continous action space
        self.manager = InitManager(self.cfg["Manager"])

        #Load the Muli Agent PPO with single critic - discrete action space
        self.worker = InitWorker(self.cfg["Worker"])        

    #This function is called for all steps and should compute the ship actions
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        #Get x_(t:t+h) | o_t from universe. (Director paper uses x for the observation)
        x = self.u.predict(obs)
        #print("Director Step", step)
        # Get the current state of the game (state: (latent, action))
        is_first = step <= 1
        if is_first:
            with torch.no_grad():
                latent= self.worldmodel.dynamics.initial(batch_size=1)
                # x = {key: torch.tensor(np.array(x[key]),dtype=torch.float32).view(1, 1, *x[key].shape[1:]).to(self.worldmodel.device) for key in x}
                # print("First step", x.keys())
                action = torch.zeros((16, 1), dtype=torch.float32).to(self.worldmodel.device)
                self.state = (latent, action)


        
        #Get positions & energy
        
        unitpos =  np.array(obs["units"]["position"][self.u.team_id]) 
        unitene = np.array(obs["units"]["energy"][self.u.team_id])
        # if step <=1 or step >= 100:
        #     print(unitpos)
        #     print(unitene)

        
        
        # get state
        s, latent = self.worldmodel.predict(x, self.state, is_first) #s: latent representation

    
             
        
        #Keep track of states for training (still torch.tensor @ device)
        s = s.expand(16, -1)
        unit_info = torch.stack([torch.tensor(self.u.get_position_info(i), dtype = torch.float32) for i in range(16)]).to(self.worldmodel.device)
        one_hot = torch.stack([torch.tensor(self.GetOneHot(i,unitene[i]), dtype = torch.float32) for i in range(16)]).to(self.worldmodel.device)
        s = torch.cat([s, one_hot, unit_info], axis=1) # (16, 512+32+18)
        if self.training:
            self.wmstates.append(s)
            

        #We need state as numpy array
        self.s = s.detach().cpu().squeeze().numpy()   

        #Get action per ship
        action = np.array([self.policies[idx].act(p[1],p[0],e,step) for idx,(p,e) in enumerate(zip(unitpos,unitene))])

        #Check to see if we should update world model & goal model
        if self.training:
            self.worldmodel.add_to_memory(step, x, action, self.u.reward, step == 1, step == 100)
            if (step > 0 and step % self.updateFreq == 0):                
                self.update(step)

            #Send data to WANDB
            self.ww.report()
            
            
            

        #Concat into np array and return actions for env
        self.state = (latent, action[:,0])
        return action
    
    def update(self, step):

        # #Update world model here 
        if step > 0 and step % 64 == 0:  
            world_model_matrics = self.worldmodel.train()
            self.ww.record("wmloss",world_model_matrics)    #<--- Merges Metrics dictionary with other metrics for WANDB
        
        #Update goal model
        goal_loss = self.goalmodel.backwardFromList(self.wmstates)        
        self.ww.record("goalloss",goal_loss)
        self.running_averages.append("goalloss",goal_loss)    

        del self.universes[:]
        del self.wmstates[:]        
    
    def save(self):
        self.worldmodel.saveDescriptive(self.cfg["Worldmodel"]["modelfile"],"Worldmodel")
        self.goalmodel.saveDescriptive(self.cfg["Goalmodel"]["modelfile"],"Goalmodel")
        self.manager.saveDescriptive(self.cfg["Manager"]["modelfile"],"Manager")
        self.worker.saveDescriptive(self.cfg["Worker"]["modelfile"],"Worker")
       

    
    def GetOneHot(self,shipIndex, energy):

        map = np.zeros(32)
        map[shipIndex] = 1
        map[shipIndex+self.numShips] = energy/100

        return map
    
    class ShipActions():
    
        def __init__(self,parent,shipIndex):

            self.parent:Director = parent
            self.shipIndex = shipIndex
            self.reset()

        def reset(self):
            self.activelast = False
            
            self.goal = None

        def missionComplete(self):
            if self.parent.training:
            
                #Compute extrinsic & exploration rewards
                r_extr = self.cumuativeExtrinsic
                r_expl = ExplorationReward(self.goal,self.parent.s)

                #Update manager with rewards
                self.parent.manager.bufferList[self.shipIndex].reward(r_extr, r_expl)
                self.parent.manager.bufferList[self.shipIndex].is_terminals.append(0)

                #<Insert Manager dreaming here>

                #Update mission control
                mgr_loss = self.parent.manager.update(self.shipIndex)
                self.parent.ww.record("mgrloss", mgr_loss)
                self.parent.running_averages.append("mgrloss",mgr_loss)  
                self.parent.ww._WW_run.log({"Manager loss": mgr_loss})              
            else:
                self.parent.manager.bufferList[self.shipIndex].clear()

        
        def setGoal(self):

            #Let manager pick a goal for this ship
            z = self.parent.manager.select_action(self.parent.s[self.shipIndex],self.shipIndex)

            #reset cumuative extrinsic reward            
            self.cumuativeExtrinsic = 0

            #Decode latent goal into goal vector (1024) using goal VAE
            gt = self.parent.goalmodel.npdecode(z)
            self.goal = gt.detach().cpu().numpy()

            #Reset clock
            self.goalclock = self.parent.clockTicks            
        
        def GetOneHot(self,e):

            map = np.zeros(32)
            map[self.shipIndex] = 1
            map[self.shipIndex+self.parent.numShips] = e/100
            return map


      
        
        def rewardShip(self,terminal):

            if self.parent.training:

                #Update worker with reward            
                r = MaxCosine(self.goal,self.parent.s)
                self.parent.worker.bufferList[self.shipIndex].reward(r)  
                self.parent.worker.bufferList[self.shipIndex].is_terminals.append(terminal)          

        def pickShipAction(self,x,y,e,step):

            #Update (extrinsic) cumulative reward
            self.cumuativeExtrinsic += self.parent.u.reward[0,self.shipIndex] # this unit's reward from universe
                        
            #Decrement goal timer
            self.goalclock-=1

            #If expired: End mission & pick a new goal
            if self.goalclock == 0:
                self.missionComplete()
                self.setGoal()

            #Create the state, as seen for a single ship
           
            shipstate = np.concat([
                self.parent.s[self.shipIndex],
                self.goal,
                self.GetOneHot(e),
                self.parent.u.get_position_info(self.shipIndex)
            ])
            
            
            #Get the current buffer length for this ship
            l = self.parent.worker.bufferList[self.shipIndex].length()  

            if self.parent.training:              

                #Check if it is update o'clock: timeout or last step in match
                if (l > 0 and l % self.parent.updateFreq == 0) or (step == 100 and l > 0):

                    #<Insert Worker dreaming here>

                    #Update worker PPO
                    wkr_loss = self.parent.worker.update(self.shipIndex)
                    self.parent.ww.record("wrkloss",wkr_loss)
                    self.parent.running_averages.append("wrkloss",wkr_loss)        

                if step == 100:
                    self.missionComplete()
                    self.reset()
            else:
                self.parent.worker.bufferList[self.shipIndex].clear()

            #Get action from worker PPO
            action = self.parent.worker.select_action(shipstate, self.shipIndex)
            zapX,zapY = 0,0

            #Was that an attempt at shooting?
            if action == 5:
                zapX,zapY = getZapCoordsOnly(x, y, self.parent.u.unit_sap_range, self.parent.u.zap_options)

            return (action, zapX, zapY)            


        def is_valid_step(self,step):
            return (step - 1) % 3 == 0 and 1 <= step <= 1 + 3 * 15  # Ensure within range
        
        def act(self, x, y, e, step):
                #Is ship in play this time step?              
                active = x != -1 and y != -1 and e > 0 and self.is_valid_step(step)               
                
                #If ship was removed from map... 
                if (self.activelast and not active) or step == 101:

                    #...we end the mission
                    self.missionComplete()

                    #and reward ship with notification that this was terminal
                    self.rewardShip(1)

                #Ship was spawned.
                elif not self.activelast and active:                    
                    self.setGoal()

                #Alive and kicking....
                elif self.activelast and active:
                    
                    #Reward ship notifying it is not terminal
                    self.rewardShip(0)

                #Active ships must take an action
                if active:

                    action = self.pickShipAction(x,y,e.item(),step)
                
                #For inactive ships we return zeros
                else:
                    action = (0,0,0)       
                
                #Update active state
                if step != 100:
                    self.activelast = active
                else:
                    self.parent.worker.bufferList[self.shipIndex].clear()
                    
                
                #Return action picked by ship
                return action
