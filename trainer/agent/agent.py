
from world.universe import Universe
from worker import Worker
from world_model import WorldModel
from replay_memory import ReplayMemory
from mission_control import MissionControl

class Agent:
    def __init__(self, player, env_cfg):
        self.player = player
        self.universe = Universe(player, obs, env_cfg, horizont=3)
        self.worker = Worker()
        self.world_model = WorldModel()
        self.replay_memory = ReplayMemory(1000)
        self.mission_control = MissionControl()


    def learn(self):
        transitions = self.replay_memory.sample(32)
        state, latent_state, action, reward, next_state, next_latent_state, done = transitions
        
        #update world model
        #update worker
        #update mission control (goal autoencoder and mgr)
        
    def push_memory(self, transitions):
        state, action, reward, next_state, done = transitions
        
        latent_state = self.world_model(state)
        latent_next_state = self.world_model(next_state)

        self.replay_memory.push(state, latent_state, action, reward, next_state, latent_next_state, done)

    def act(self, step, obs, remainingOverageTime):
        state = self.universe.predict(obs)
        latent_state = self.world_model(state)
        mission = self.mission_control.act(latent_state) # mission is a 16 x 24 x 24 matrix
        # filter workers that are not alive
        actions = self.worker(latent_state, mission)
        
        return actions


