import torch
import torch.nn as nn
from world_model.world_model import WorldModel, ImagBehavior
import numpy as np
import os
from pathlib import Path

class Dreamer(nn.Module):
    def __init__(self, config, logger=None):
        super(Dreamer, self).__init__()
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize World Model
        self.world_model = WorldModel(config).to(self.device)

        # Initialize Imaginary Policy Behavior
        self.imag_behavior = ImagBehavior(config, self.world_model, stop_grad_actor=True).to(self.device)

        # Load existing model (if available)
        self.model_path = Path(os.path.abspath(os.path.dirname(__file__))) / "world_model" / "world_model.pt"
        if self.model_path.exists():
            print(f"Loading model from {self.model_path}")
            self.world_model.load_model(self.model_path)

        # Initialize replay memory & counters
        self.replay_memory = []
        self.step_count = 0
        self.training = True

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """ 
        Given an observation, decide on an action using the trained world model 
        and the actor policy learned from imagined trajectories.
        """
        with torch.no_grad():
            # Preprocess observation and convert to latent space
            obs = {key: torch.tensor(np.array(value), dtype=torch.float32).to(self.device).unsqueeze(0) for key, value in obs.items()}
            is_first = step <= 1
            if is_first:
                batch_size = 1
                latent= self.worldmodel.dynamics.initial(batch_size)
                action = torch.zeros((16, 1), dtype=torch.float32).to(self.worldmodel.device)
            else:
                latent, action = self.state

            # Encode current observation
            embed = self.world_model.encoder(obs)
            
            # Perform a step in the world model's latent space
            latent, _ = self.world_model.dynamics.obs_step(latent, action, embed, torch.tensor([0], dtype=torch.int8).to(self.device))
            feat = self.world_model.dynamics.get_feat(latent)

            # Choose action based on training or exploration mode
            if self.training:
                if np.random.rand() < self.config["exploration_prob"]:
                    action = self.imag_behavior.actor(feat).sample()
                else:
                    action = self.imag_behavior.actor(feat).mode()
            else:
                action = self.imag_behavior.actor(feat).mode()

            action = action.detach().cpu().numpy()[0]

            self.state = (latent, action[:,0])
            return action

    def train(self):
        """
        Train both the world model and the imaginary behavior model.
        """
        if len(self.world_model.memory) < self.config["batch_size"]:
            return {}

        # Sample batch from replay memory
        sequences = self.world_model.memory.sample(self.config["batch_size"])
        batch, actions, rewards, is_first, done = self.world_model.convert_sequence_to_tensor(sequences)
        batch["reward"] = rewards
        batch["cont"] = torch.Tensor(1.0 - done).to(self.device)

        # Train world model
        true_post, context, metrics = self.world_model._train(batch, actions, is_first)

        # Train the imaginary behavior model
        imagined_states = true_post
        reward_fn = lambda f, s, a: self.world_model.heads["reward"](self.world_model.dynamics.get_feat(s)).mode()
        self.imag_behavior._train(imagined_states, reward_fn)

        return metrics

    def save_model(self):
        """Save trained world model."""
        self.world_model.save_model(self.model_path)
        print("Dreamer Model Saved!")

    def update_memory(self, step, state, action, reward, is_first, done):
        """Store transitions in memory for training."""
        self.world_model.add_to_memory(step, state, action, reward, is_first, done)