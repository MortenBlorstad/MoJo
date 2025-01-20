from abc import ABC, abstractmethod


class base_component(ABC):
    
    def __init__(self):
        super().__init__()
        self.env_params_ranges = dict(
            # map_type=[1],
            unit_move_cost=list(range(1, 6)),
            unit_sensor_range=list(range(2, 5)),
            nebula_tile_vision_reduction=list(range(0, 4)),
            nebula_tile_energy_reduction=[0, 0, 10, 25],
            unit_sap_cost=list(range(30, 51)),
            unit_sap_range=list(range(3, 8)),
            unit_sap_dropoff_factor=[0.25, 0.5, 1],
            unit_energy_void_factor=[0.0625, 0.125, 0.25, 0.375],
            # map randomizations
            nebula_tile_drift_speed=[-0.05, -0.025, 0.025, 0.05],
            energy_node_drift_speed=[0.01, 0.02, 0.03, 0.04, 0.05],
            energy_node_drift_magnitude=list(range(3, 6)),
        )


    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass


    
    