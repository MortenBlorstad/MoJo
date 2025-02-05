from world.base_component import base_component
from jax import jit, lax
import jax.numpy as jnp
import numpy as np
from world.nebula import Nebula

import copy



class NebulaAstroid(base_component):

    def __init__(self, horizon):
        super().__init__()
        self.horizon:int = horizon
        self.nebula_tile_drift_speed:float = 0.0
        self.change_steps:list = [7, 10, 14, 20, 27, 30, 34, 40, 47, 50, 54, 60, 67, 70, 74, 80, 87, 90, 94, 100]
        self.change_steps_set:set = set(self.change_steps)
        self.previous_observed_change:int = 0
        self.change_rate:float = 0
        self.map = jnp.ones((2,24,24))
        self.nebula = Nebula(horizon,"nebula")
        self.astroid = Nebula(horizon, "astroid")
        self.direction:float = np.inf



    def round_down_to_nearest_100(self,step:int)->int:
        return (step // 100) * 100
    
    def should_check(self,step:int)->bool:
        return (step - self.round_down_to_nearest_100(step)) in self.change_steps_set

    def update_change_rate(self,new_rate:float)->None:
        if self.previous_observed_change ==0:
            self.change_rate = new_rate
        else:
            self.change_rate = np.round(0.7*self.change_rate + 0.3*new_rate,4)

    def closest_change_rate(self,change_rate:float, possible_rates:list=[0.15, 0.1, 0.05, 0.025])->float:
        """
        Selects the closest value from the list of possible change rates to the given change_rate.
        
        Args:
            change_rate (float): The estimated change rate.
            possible_rates (list of float): The predefined list of possible change rates.
        
        Returns:
            float: The closest change rate from the list.
        """
        return min(possible_rates, key=lambda x: abs(x - change_rate))

    # def set_values(from_info:Nebula, to_info:Nebula):
    
    
    


    def learn(self, nebulas,astroids, observable, current_step):
        print("nebula")
        observed_change_nebula = self.nebula.learn(nebulas,observable,current_step-1, self.previous_observed_change)
        
        print("astroid")
        observed_change_astroid =  self.astroid.learn(astroids, observable, current_step-1, self.previous_observed_change)
        
        
        # update them based on which find the solution first.
        nebula_drift = float(self.nebula.nebula_tile_drift_speed)
        astroid_drift = float(self.astroid.nebula_tile_drift_speed)
    
        nebula_dir = float(self.nebula.direction)
        astroid_dir = float(self.astroid.direction)

        nebula_change_rate = float(self.nebula.change_rate)
        astroid_change_rate = float(self.astroid.change_rate)

        if observed_change_nebula or observed_change_astroid:
           
            self.previous_observed_change = current_step-1
            
            weight = 1/float(observed_change_nebula + observed_change_astroid)
            print("dir",weight,  (observed_change_nebula*nebula_dir), observed_change_astroid*astroid_dir)
            self.direction = jnp.sign(weight*(observed_change_nebula*nebula_dir) + weight*(observed_change_astroid*astroid_dir)).item()
            print(current_step-1,self.direction,observed_change_nebula,observed_change_astroid,(observed_change_nebula * nebula_change_rate),(observed_change_astroid*astroid_change_rate))
            new_rate = weight*(observed_change_nebula * nebula_change_rate) + weight*(observed_change_astroid*astroid_change_rate)
            self.nebula.change_rate = new_rate
            self.astroid.change_rate = new_rate
            
            if self.change_rate ==0:
                self.change_rate = new_rate
            else: 
                self.change_rate = self.change_rate*0.7 + new_rate*0.3

            if self.nebula.found_unique:
                self.nebula_tile_drift_speed = nebula_drift
                self.astroid.change_rate = nebula_change_rate
                self.astroid.direction = nebula_dir
            elif self.astroid.found_unique:
                self.nebula_tile_drift_speed = astroid_drift
                self.nebula.change_rate = astroid_change_rate
                self.nebula.direction = astroid_dir

            else:
                print("change_rate", self.change_rate, "direction", self.direction, self.closest_change_rate(self.change_rate))
                self.nebula_tile_drift_speed = self.direction*self.closest_change_rate(self.change_rate)
        
        
        print("after",self.nebula_tile_drift_speed,self.nebula.nebula_tile_drift_speed, self.astroid.nebula_tile_drift_speed,
               self.nebula.change_rate, self.astroid.change_rate,
                 self.nebula.previous_observed_change,self.astroid.previous_observed_change, self.previous_observed_change,
                 self.nebula.direction , self.astroid.direction, self.direction 
                   )


           
  
        
    def predict(self,nebulas,astroids, observable,current_step):
        # update nebula_tile_drift_speed
        self.nebula.nebula_tile_drift_speed = self.nebula_tile_drift_speed
        self.astroid.nebula_tile_drift_speed = self.nebula_tile_drift_speed

        nebula_predictions = self.nebula.predict(nebulas,observable, current_step-1)
        astroid_predictions = self.astroid.predict(astroids,observable, current_step-1)
        return nebula_predictions, astroid_predictions

