from abc import ABC, abstractmethod


class base_component(ABC):
    
    

    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass


    
    