import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hierarchical.multiagentutils import Critic, EMSDNormalizer, MgrRolloutBuffer as RolloutBuffer

################################## set device ##################################

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()



class ActorCriticBase(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCriticBase, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 512),
                            nn.Tanh(),
                            nn.Linear(512, 256),
                            nn.Tanh(),
                            nn.Linear(256, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 512),
                            nn.Tanh(),
                            nn.Linear(512, 256),
                            nn.Tanh(),
                            nn.Linear(256, action_dim),
                            nn.Softmax(dim=-1)
                        )
        #Use double critics : extrinsic & exploration
        self.critic_extr = Critic(state_dim)
        self.critic_expl = Critic(state_dim)
            
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

class BehaviourAC(ActorCriticBase):
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)       

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val_extr = self.critic_extr(state)
        state_val_expl = self.critic_expl(state)

        return action.detach(), action_logprob.detach(), state_val_extr.detach(), state_val_expl.detach()


class CommonAC(ActorCriticBase):
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)       

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()        
        state_val_extr = self.critic_extr(state)
        state_val_expl = self.critic_expl(state)        
        
        return action_logprobs, state_val_extr,state_val_expl, dist_entropy



class MultiAgentManagerPPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, num_workers=2):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.num_workers = num_workers
        
        self.bufferList = []
        for _ in range(num_workers):
            self.bufferList.append(RolloutBuffer())
        
        self.commonac = CommonAC(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.commonac.actor.parameters(), 'lr': lr_actor},
            {'params': self.commonac.critic_extr.parameters(), 'lr': lr_critic},
            {'params': self.commonac.critic_expl.parameters(), 'lr': lr_critic}
        ])

        self.workerACs = []
        for _ in range(num_workers):
            newworkerAc = BehaviourAC(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
            newworkerAc.load_state_dict(self.commonac.state_dict())
            self.workerACs.append(newworkerAc)
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.commonac.set_action_std(new_action_std)
            for i in range(self.num_workers):
                self.workerACs[i].set_action_std(new_action_std)


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std

            self.set_action_std(self.action_std)

    def select_action(self, state, worker_id):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val_extr, state_val_expl = self.workerACs[worker_id].act(state)                

        self.bufferList[worker_id].states.append(state)
        self.bufferList[worker_id].actions.append(action)
        self.bufferList[worker_id].logprobs.append(action_logprob)
        self.bufferList[worker_id].state_values_extr.append(state_val_extr)
        self.bufferList[worker_id].state_values_expl.append(state_val_expl)          

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def update(self, worker_id):

        def conditional_squeeze(tensor):
            return tensor.squeeze() if tensor.ndim > 1 else tensor

        # Initialize EMSD normalizers
        extrinsic_normalizer =      EMSDNormalizer(alpha=0.1)
        exploration_normalizer =    EMSDNormalizer(alpha=0.1)

        # Monte Carlo estimate of returns for extrinsic and exploration rewards
        extrinsic_rewards = self.bufferList[worker_id].extrinsic_rewards  # List of extrinsic rewards
        exploration_rewards = self.bufferList[worker_id].exploration_rewards  # List of exploration rewards

        discounted_extrinsic = []
        discounted_exploration = []
        G_extr, G_expl = 0, 0

        for r_extr, r_expl, is_terminal in zip(
            reversed(extrinsic_rewards), reversed(exploration_rewards), reversed(self.bufferList[worker_id].is_terminals)
        ):
            if is_terminal:
                G_extr, G_expl = 0, 0
            G_extr = r_extr + self.gamma * G_extr
            G_expl = r_expl + self.gamma * G_expl
            discounted_extrinsic.insert(0, G_extr)
            discounted_exploration.insert(0, G_expl)

        # Normalize returns using EMSD        
        normalized_extrinsic =   extrinsic_normalizer.normalize(discounted_extrinsic).to(device)
        normalized_exploration = exploration_normalizer.normalize(discounted_exploration).to(device)

        # Compute final return with weighted sum
        w_extr, w_expl = 1.0, 0.1
        rewards = w_extr * normalized_extrinsic + w_expl * normalized_exploration
        
        #Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.bufferList[worker_id].states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.bufferList[worker_id].actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.bufferList[worker_id].logprobs, dim=0)).detach().to(device)   

        #Extract old state values from both critics
        old_state_values_extr = torch.squeeze(torch.stack(self.bufferList[worker_id].state_values_extr, dim=0)).detach().to(device)
        old_state_values_expl = torch.squeeze(torch.stack(self.bufferList[worker_id].state_values_expl, dim=0)).detach().to(device)

        #Compute total state value using weighted sum
        old_state_values = w_extr * old_state_values_extr + w_expl * old_state_values_expl

        #Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()        

        #Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluate old actions using current policy
            logprobs, state_values_extr, state_values_expl, dist_entropy = self.commonac.evaluate(old_states, old_actions)

            # Compute total state value estimate
            state_values = w_extr * state_values_extr + w_expl * state_values_expl
            state_values = conditional_squeeze(state_values)  # Match dimensions with rewards

            # Compute PPO ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Compute PPO surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Compute final PPO loss (with value loss from both critics)
            loss = (
                -torch.min(surr1, surr2) 
                + 0.25 * self.MseLoss(state_values_extr, normalized_extrinsic)
                + 0.25 * self.MseLoss(state_values_expl, normalized_exploration)
                + 0.5 * self.MseLoss(state_values, rewards)  # Total value loss
                - 0.01 * dist_entropy  # Entropy regularization
            ).mean()

            #Take gradient step
            self.optimizer.zero_grad()
            #loss.mean().backward()
            loss.backward()
            self.optimizer.step()
            
        #Copy new weights into old policy
        self.workerACs[worker_id].load_state_dict(self.commonac.state_dict())

        #Clear buffer
        self.bufferList[worker_id].clear()
        return loss
    
    def save(self, checkpoint_path):
        torch.save(self.commonac.state_dict(), checkpoint_path)

    def saveDescriptive(self, path, name):        
        self.save(path)
        print("Saved",name,"to file",path)
   
    def load(self, checkpoint_path):    
        dictionary = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.commonac.load_state_dict(dictionary)
        for worker in self.workerACs:
            worker.load_state_dict(dictionary)