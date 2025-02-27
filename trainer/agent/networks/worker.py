import torch
from torch import nn
from networks.ppo import BackboneNetwork

class Worker(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout, 
                 discount_factor, optimizer):
        super().__init__()
        self.actor = BackboneNetwork(latent_dim, hidden_dim, output_dim, dropout)
        self.critic = BackboneNetwork(latent_dim, hidden_dim, 1, dropout)
        self.discount_factor = discount_factor
        self.optimizer = optimizer



    def forward(self, state, goal):
        action_pred = self.actor(state, goal)
        value_pred = self.critic(state)
        return action_pred, value_pred
    

    def learn(self, transitions):
        states, actions, rewards, next_states, dones = transitions

        self.optimizer.zero_grad()
        action_pred, value_pred = self(states, next_states)
        action_prob = torch.softmax(action_pred, dim=-1)
        action_log_prob = torch.log(action_prob)
        next_action_prob = torch.softmax(action_pred, dim=-1)
        next_action_log_prob = torch.log(next_action_prob)
        next_value_pred = self.critic(next_states)
        td_target = rewards + self.discount_factor * next_value_pred * (1 - dones)
        td_error = td_target - value_pred

        actor_loss = -action_log_prob * td_error.detach()
        critic_loss = td_error ** 2

        loss = actor_loss + critic_loss
        loss.mean().backward()
        self.optimizer.step()
        return loss.mean().item()

    


    








