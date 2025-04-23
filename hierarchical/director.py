import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from world.universe import Universe
from hierarchical.config import Config
from hierarchical.utils import getZapCoordsOnly, InitWorker, InitManager, RunningAverages
from hierarchical.vae import VAE
from hierarchical.directorlossfuncs import MaxCosineNP as MaxCosine, ExplorationReward
import torch
class Director():    
   
    def __init__(self, player: str, env_cfg, training = False) -> None:

        #Get variables
        self.player = player
        self.u = Universe(player, env_cfg, horizont=3)        
        self.cfg = Config().Get("Director")   
        self.clockTicks = self.cfg["TimeSteps_K"]
        self.updateFreq = self.cfg["TimeSteps_E"]
        self.numShips = 16
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

        #Load the Multi Agent PPO with double critic - continous action space
        self.manager = InitManager(self.cfg["Manager"], fromfile=False)

        #Load the Multi Agent PPO with single critic - discrete action space
        self.worker = InitWorker(self.cfg["Worker"], fromfile=False)        

        #Create a list of 'ShipActions' to keep track of when to update individual policies (per ship (16))  
        self.policies = [self.ShipActions(self,idx) for idx in range(self.numShips)] 
    
        #Keep track of the raw output from Universe so we can train worldmodel VAE
        self.universes = []

        #Load pretrained world model autoencoder
        #self.worldmodel = VAE.Load(self.cfg["Worldmodel"])

        #Create a new world model autoencoder
        self.worldmodel = VAE.Create(self.cfg["Worldmodel"])

        #Keep track of the output from world model so we can train goal VAE
        self.wmstates = []

        #Load pretrained goal autoencoder
        #self.goalmodel = VAE.Load(self.cfg["Goalmodel"])

        #Create a new goal autoencoder
        self.goalmodel = VAE.Create(self.cfg["Goalmodel"]) 

    def flattened_universe(self, x):
        flattened = np.concatenate([x["image"].flatten(), x["step_embedding"].flatten(), x["scalars"].flatten(), x["one_hot_unit_id"].flatten(), x["one_hot_unit_energy"].flatten()])
        flattened = torch.Tensor(flattened).to(self.worldmodel.device)
        return flattened

    #This function is called for all steps and should compute the ship actions
    def act(self, step: int, obs, remainingOverageTime: int = 60):

        step = obs['steps']

        #Get x_(t:t+h) | o_t from universe. (Director paper uses x for the observation)
        x = self.flattened_universe(self.u.predict(obs))

        #Keep track of the universe for training
        self.universes.append(x)

        #Encode the universe into a latent space        
        s = self.worldmodel.inference(x)

        #Get positions & energy
        unitpos =  np.array(obs["units"]["position"][self.u.team_id]) 
        unitene = np.array(obs["units"]["energy"][self.u.team_id])                  
        
        #Keep track of states for training (still torch.tensor @ device)
        if self.training:
            self.wmstates.append(s)

        #We need state as numpy array
        self.s = s.detach().cpu().squeeze().numpy()   

        #Get action per ship
        action = np.array([self.policies[idx].act(p[1],p[0],e,step) for idx,(p,e) in enumerate(zip(unitpos,unitene))])

        #Check to see if we should update world model & goal model
        if self.training:

            #self.worldmodel.add_to_memory(step, x, action, self.u.reward, step == 1, step ==100)
            if (step > 0 and step % self.updateFreq == 0):                
                self.update()

            #Send data to WANDB
            self.ww.report()            

        return action
    
    def update(self):

        #Update world model here       
        #world_model_matrics = self.worldmodel.train()
        #self.ww.record("wmloss",world_model_matrics)    #<--- Merges Metrics dictionary with other metrics for WANDB
        
        #Update world model
        wmloss = self.worldmodel.backwardFromList(self.universes)
        self.running_averages.append("wmloss",wmloss)

        #Update goal model
        goal_loss = self.goalmodel.backwardFromList(self.wmstates) 
        self.running_averages.append("goalloss",goal_loss)

        del self.universes[:]
        del self.wmstates[:]    
    
    def save(self):
        self.worldmodel.saveDescriptive(self.cfg["Worldmodel"]["modelfile"],"Worldmodel")
        self.goalmodel.saveDescriptive(self.cfg["Goalmodel"]["modelfile"],"Goalmodel")
        self.manager.saveDescriptive(self.cfg["Manager"]["modelfile"],"Manager")
        self.worker.saveDescriptive(self.cfg["Worker"]["modelfile"],"Worker")

    
    class ShipActions():  

        def __init__(self,parent,shipIndex):

            self.parent = parent
            self.shipIndex = shipIndex
            self.reset()

        def dbg(self,step,*args):            
            if self.shipIndex == 0:                
                msg = " ".join(args)
                #print(f"Step {step}: {msg}")             

        def reset(self):
            self.activelast = False
            self.goal = None            

        def missionComplete(self):

            self.dbg("X","Mission complete") 

            if self.parent.training:
            
                #Compute extrinsic & exploration rewards
                r_extr = self.cumuativeExtrinsic
                r_expl = ExplorationReward(self.goal,self.parent.s)

                #Update manager with rewards
                self.parent.manager.bufferList[self.shipIndex].reward(r_extr, r_expl)
                self.parent.manager.bufferList[self.shipIndex].is_terminals.append(0)

                #<Insert Manager dreaming here>

                #Update mission control
                ml = self.parent.manager.update(self.shipIndex)
                self.parent.ww.record("mgrloss",ml)                
                self.parent.running_averages.append("mgrloss",ml)
            else:
                self.parent.manager.bufferList[self.shipIndex].clear()

        def setGoal(self):

            #Let manager pick a goal for this ship
            z = self.parent.manager.select_action(self.parent.s,self.shipIndex)

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
            self.cumuativeExtrinsic += self.parent.u.thiscore # <- this is the reward from the universe: the number of relics collected in this timestep
                        
            #Decrement goal timer
            self.goalclock-=1

            #If expired: End mission & pick a new goal
            if self.goalclock == 0:
                self.missionComplete()
                self.setGoal()

            #Create the state, as seen for a single ship
            shipstate = np.concat([
                self.parent.s,
                self.goal,
                self.GetOneHot(e)
            ])
            
            #Get the current buffer length for this ship
            l = self.parent.worker.bufferList[self.shipIndex].length()

            if self.parent.training:              

                #Check if it is update o'clock: timeout or last step in match
                if l > 0 and (l % self.parent.updateFreq == 0 or step == 100):

                    #<Insert Worker dreaming here>
                    
                    #Update worker PPO
                    wl = self.parent.worker.update(self.shipIndex)
                    self.parent.ww.record("wrkloss",wl)
                    self.parent.running_averages.append("wrkloss",wl)

                    self.dbg(step,"Worker updated")

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

        def act(self,x,y,e,step):
                                
                tmp = self.parent.worker.bufferList[self.shipIndex].tmplength()
                self.dbg(step,tmp)

                #Is ship in play this time step?
                active = x != -1 and y != -1
                #if self.shipIndex == 0 and step == 23:
                #    active = False                    
                #    self.dbg(step,"OVERRIDE")
                self.dbg(step,"Active last =",str(self.activelast),"Active =",str(active))

                #If ship was removed from map... 
                if self.activelast and not active:                   

                    #...we end the mission
                    self.missionComplete()                    

                    #and reward ship with notification that this was terminal
                    self.rewardShip(1)
                    self.dbg(step,"Rewarded with terminal")                    

                #Ship was spawned.
                elif not self.activelast and active:                   
                    self.setGoal()
                    self.dbg(step,"New goal picked")

                #Alive and kicking....
                elif self.activelast and active:
                    
                    #Reward ship notifying it is not terminal
                    self.rewardShip(0)
                    self.dbg(step,"Rewarded with non-terminal")
                #Active ships must take an action
                if active:
                    action = self.pickShipAction(x,y,e.item(),step)
                    self.dbg(step,"Action picked")
                #For inactive ships we return zeros
                else:
                    action = (0,0,0)
                
                #Update active state
                if step != 100:
                    self.activelast = active
                else:
                    self.parent.worker.bufferList[self.shipIndex].clear()
                    
                tmp = self.parent.worker.bufferList[self.shipIndex].tmplength()
                self.dbg(step,tmp)
                self.dbg(step,"")
                



                #Return action picked by ship
                return action
