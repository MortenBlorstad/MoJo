import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.one_hot_pos = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.one_hot_pos[:]


def compute_conv_output_size(input_size, kernel_size, stride, padding=0, dilation=1):
    """
    Compute the output size (height/width) after a convolutional layer.

    Args:
        input_size (int): Input height or width.
        kernel_size (int): Kernel/filter size.
        stride (int): Stride size.
        padding (int, optional): Padding applied. Default: 0.
        dilation (int, optional): Dilation rate. Default: 1.

    Returns:
        int: Output height or width after applying convolution.
    """
    return ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

class Actor(nn.Module):
    def __init__(self, feature_dim, one_hot_pos_dim, action_dim, has_continuous_action_space):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space
        self.num_units = 16
        
        if has_continuous_action_space :
            self.network = nn.Sequential(
                            nn.Linear(feature_dim + one_hot_pos_dim, 64),
                            nn.SiLU(),
                            nn.Linear(64, 64),
                            nn.SiLU(),
                            nn.Linear(64, action_dim),
                            nn.SiLU()
                        )
        else:
            self.network = nn.Sequential(
                            nn.Linear(feature_dim + one_hot_pos_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
            
    def forward(self, x, one_hot_pos):
        batch_size = x.size(0)
        actions = []
        for i in range(16):
            state = torch.cat((x, one_hot_pos[:, i]), dim=1)
            action = self.network(state)
            actions.append(action)
        actions = torch.stack(actions, dim=0)
        actions = actions.view(batch_size, self.num_units, self.action_dim)
        return actions
    
class Critic(nn.Module):
    def __init__(self, feature_dim, one_hot_pos_dim):
        super(Critic, self).__init__()
        self.num_units = 16
        self.network = nn.Sequential(
                        nn.Linear(feature_dim + one_hot_pos_dim, 64),
                        nn.SiLU(),
                        nn.Linear(64, 64),
                        nn.SiLU(),
                        nn.Linear(64, 1)
                    )

    def forward(self, x, one_hot_pos):
        batch_size = x.size(0)
        values = []
        for i in range(16):
            state = torch.cat((x, one_hot_pos[:, i]), dim=1)
            action = self.network(state)
            values.append(action)
        values = torch.stack(values, dim=0)
        values = values.view(batch_size, self.num_units, 1)
        return values
    

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space,
                  action_std_init, image_size=None):
        super(ActorCritic, self).__init__()
        self.num_units = 16
        self.has_continuous_action_space = has_continuous_action_space
        self.image_size = image_size # (channel, height, width)
        one_hot_pos_dim = 24*24
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # Feature Extractor for Images (CNN)
        if self.image_size is not None:
            c_size, w_size, h_size = self.image_size
            channel_sizes = [64, 64, 64]
            kernel_sizes = [3, 3, 3]
            strides = [2, 2, 1]  
            paddings = [(k - 1) // 2 for k in kernel_sizes]  

            self.feature_extractor = nn.Sequential(
                nn.Conv2d(c_size, channel_sizes[0], kernel_size=kernel_sizes[0],
                          padding=paddings[0], stride=strides[0], padding_mode="zeros"), 
                nn.Tanh(),
                nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=kernel_sizes[1],
                          padding=paddings[1], stride=strides[1], padding_mode="zeros"),
                nn.Tanh(),
                nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=kernel_sizes[2],
                          padding=paddings[2], stride=strides[2], padding_mode="zeros"),
                nn.Tanh(),
                nn.Flatten()
            )

            
            for i in range(3):
                w_size = compute_conv_output_size(w_size, kernel_sizes[i], strides[i], paddings[i])
            feature_dim = w_size * w_size * 64
            
        else:
            feature_dim = state_dim  # MLP directly takes state_dim as input
        
        # actor
        self.actor = Actor(feature_dim, one_hot_pos_dim, action_dim, has_continuous_action_space)
        # critic
        self.critic = Critic(feature_dim, one_hot_pos_dim)
        
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self, state, one_hot_pos):
        if self.image_size is not None:
            #state = state / 255.0  # Normalize image
            state = self.feature_extractor(state)

        return self.actor(state, one_hot_pos), self.critic(state, one_hot_pos)
    
    def act(self, state, one_hot_pos):
        if self.image_size is not None:
            #state = state / 255.0  # Normalize image
            state = self.feature_extractor(state)

        
        if self.has_continuous_action_space:
            action_mean = self.actor(state, one_hot_pos)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state, one_hot_pos)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state, one_hot_pos)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, one_hot_pos, action):
       
        if self.image_size is not None:
            #state = state / 255.0  # Normalize
            state = self.feature_extractor(state)
        if self.has_continuous_action_space:
            action_mean = self.actor(state, one_hot_pos)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
         
            action_probs = self.actor(state, one_hot_pos)
     
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state, one_hot_pos)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma,
                  K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6,
                  image_size = None):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space,
                                   action_std_init, image_size).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space,
                                       action_std_init, image_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, one_hot_pos):
        
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                one_hot_pos = torch.FloatTensor(one_hot_pos).to(device)
                action, action_logprob, state_val = self.policy_old.act(state, one_hot_pos)

            self.buffer.states.append(state)
            self.buffer.one_hot_pos.append(one_hot_pos)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                one_hot_pos = torch.FloatTensor(one_hot_pos).to(device)
                action, action_logprob, state_val = self.policy_old.act(state, one_hot_pos)
            
            self.buffer.states.append(state)
            self.buffer.one_hot_pos.append(one_hot_pos)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            action = action.squeeze(0).detach().cpu().numpy()
            
            valid_action_form = np.zeros((16, 3), dtype=int)
            #TODO: the last two dims are for zapping and attacking,
            #  we need to add functionality to add these

            # Fill the first column with original values
            valid_action_form[:, 0] = action.squeeze()
            return valid_action_form
            

    def update(self, units_inplay):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = rewards.unsqueeze(-1).expand(-1, self.policy_old.num_units) # Expand rewards if it was (batch,) to (batch, num_units)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0),dim=1).detach().to(device)
        old_one_hot_pos = torch.squeeze(torch.stack(self.buffer.one_hot_pos, dim=0),dim=1).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        # Optimize policy for K epochs
        total_loss = 0.0
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_one_hot_pos, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = (ratios * advantages)
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # final loss of clipped objective PPO
            loss = (-torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy)
            loss = loss[:,units_inplay].mean() # mask away unavailable units
            total_loss += loss.item()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return total_loss / self.K_epochs
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       

    
