import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from typing import List, Tuple, Optional, Union, Dict, Any
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hierarchical.multiagentutils import Critic, WrkrRolloutBuffer, MgrRolloutBuffer, Normalize

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


class ActorCriticBase(nn.Module):
    """
    Base class for actor-critic networks in PPO.
    Implements common functionality for both continuous and discrete action spaces.
    
    Attributes:
        has_continuous_action_space (bool): Whether the action space is continuous
        action_dim (int): Dimension of the action space
        action_var (torch.Tensor): Variance for continuous actions
        actor (nn.Sequential): Actor network
        critic (Critic): Critic network
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 has_continuous_action_space: bool, action_std_init: float) -> None:
        """
        Initialize the actor-critic base network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            has_continuous_action_space (bool): Whether the action space is continuous
            action_std_init (float): Initial standard deviation for continuous actions
        """
        super(ActorCriticBase, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            
        # actor network
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
            
        # critic network
        self.critic = Critic(state_dim)
        
    def set_action_std(self, new_action_std: float) -> None:
        """
        Update the standard deviation for continuous actions.
        
        Args:
            new_action_std (float): New standard deviation value
        """
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)


class BehaviourAC(ActorCriticBase):
    """
    Behavior actor-critic network for action selection.
    Extends ActorCriticBase with action sampling functionality.
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:        
        super().__init__(*args, **kwargs)       

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select an action based on the current state.
        
        Args:
            state (torch.Tensor): Current state observation
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Selected action
                - Action log probability
                - State value estimate
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()


class CommonAC(ActorCriticBase):
    """
    Common actor-critic network for policy evaluation.
    Extends ActorCriticBase with policy evaluation functionality.
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:        
        super().__init__(*args, **kwargs)       

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the policy for given states and actions.
        
        Args:
            state (torch.Tensor): Batch of states
            action (torch.Tensor): Batch of actions
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Action log probabilities
                - State values
                - Distribution entropy
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class MultiAgentPPO:
    """
    Multi-Agent Proximal Policy Optimization implementation.
    Supports both worker and manager agents in a hierarchical RL setting.
    
    Attributes:
        has_continuous_action_space (bool): Whether the action space is continuous
        action_std (float): Standard deviation for continuous actions
        gamma (float): Discount factor
        eps_clip (float): PPO clipping parameter
        K_epochs (int): Number of optimization epochs
        num_workers (int): Number of worker agents
        isWorker (bool): Whether this is a worker agent
        bufferList (List[Union[WrkrRolloutBuffer, MgrRolloutBuffer]]): List of rollout buffers
        commonac (CommonAC): Common actor-critic network
        workerACs (List[BehaviourAC]): List of behavior actor-critic networks
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr_actor: float, lr_critic: float,
                 gamma: float, K_epochs: int, eps_clip: float, has_continuous_action_space: bool,
                 action_std_init: float = 0.6, num_workers: int = 2, isWorker: bool = True) -> None:
        """
        Initialize the Multi-Agent PPO system.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            lr_actor (float): Learning rate for the actor network
            lr_critic (float): Learning rate for the critic network
            gamma (float): Discount factor
            K_epochs (int): Number of optimization epochs
            eps_clip (float): PPO clipping parameter
            has_continuous_action_space (bool): Whether the action space is continuous
            action_std_init (float): Initial standard deviation for continuous actions
            num_workers (int): Number of worker agents
            isWorker (bool): Whether this is a worker agent
        """
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.num_workers = num_workers
        self.isWorker = isWorker
        
        self.bufferList = []

        if self.isWorker:
            for _ in range(num_workers):
                # Initialize worker buffer
                self.bufferList.append(WrkrRolloutBuffer())
                # Initialize normalizer  
                self.normalize = Normalize()      
        else:
            for _ in range(num_workers):
                # Initialize manager buffer
                self.bufferList.append(MgrRolloutBuffer())
                # Initialize normalizers        
                self.extrinsic_normalize = Normalize()
                self.exploration_normalize = Normalize()
        
        self.commonac = CommonAC(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.commonac.actor.parameters(), 'lr': lr_actor},
            {'params': self.commonac.critic.parameters(), 'lr': lr_critic}
        ])

        self.workerACs = []
        for _ in range(num_workers):
            newworkerAc = BehaviourAC(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
            newworkerAc.load_state_dict(self.commonac.state_dict())
            self.workerACs.append(newworkerAc)
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std: float) -> None:
        """
        Update the standard deviation for continuous actions.
        
        Args:
            new_action_std (float): New standard deviation value
        """
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.commonac.set_action_std(new_action_std)
            for i in range(self.num_workers):
                self.workerACs[i].set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate: float, min_action_std: float) -> None:
        """
        Decay the action standard deviation over time.
        
        Args:
            action_std_decay_rate (float): Rate at which to decay the standard deviation
            min_action_std (float): Minimum allowed standard deviation
        """
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std

            self.set_action_std(self.action_std)

    def select_action(self, state: np.ndarray, worker_id: int) -> Union[np.ndarray, int]:
        """
        Select an action for a given worker based on the current state.
        
        Args:
            state (np.ndarray): Current state observation
            worker_id (int): ID of the worker agent
            
        Returns:
            Union[np.ndarray, int]: Selected action
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.workerACs[worker_id].act(state)

        self.bufferList[worker_id].states.append(state)
        self.bufferList[worker_id].actions.append(action)
        self.bufferList[worker_id].logprobs.append(action_logprob)
        self.bufferList[worker_id].state_values.append(state_val)            
           
        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def update(self, worker_id: int) -> float:
        """
        Update the policy for a given worker using PPO.
        
        Args:
            worker_id (int): ID of the worker agent to update
            
        Returns:
            float: Loss value from the update
        """
        def conditional_squeeze(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.squeeze() if tensor.ndim > 1 else tensor

        if self.isWorker:
            # Normalize rewards using normalizer from paper's implementation
            rewards = torch.tensor(self.bufferList[worker_id].rewards, dtype=torch.float32).to(device)
            rewards = self.normalize(rewards)
        else:
            # Normalize extrinsic and exploration rewards using normalizer from papers implementation
            extrinsic_rewards = torch.tensor(self.bufferList[worker_id].extrinsic_rewards, dtype=torch.float32).to(device)
            extrinsic_rewards = self.extrinsic_normalize(extrinsic_rewards)

            exploration_rewards = torch.tensor(self.bufferList[worker_id].exploration_rewards, dtype=torch.float32).to(device)
            exploration_rewards = self.exploration_normalize(exploration_rewards)

            # Compute final return with weighted sum
            w_extr, w_expl = 1.0, 0.1   # Use same weights as the paper
            rewards = w_extr * exploration_rewards + w_expl * exploration_rewards
        
        # Convert lists to tensors
        old_states = torch.squeeze(torch.stack(self.bufferList[worker_id].states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.bufferList[worker_id].actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.bufferList[worker_id].logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.bufferList[worker_id].state_values, dim=0)).detach().to(device)

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.commonac.evaluate(old_states, old_actions)

            # Match state_values tensor dimensions with rewards tensor
            state_values = conditional_squeeze(state_values)            
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = (-torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy).mean()

            # Take gradient step        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.workerACs[worker_id].load_state_dict(self.commonac.state_dict())

        # Clear buffer
        self.bufferList[worker_id].clear()

        return loss.detach().cpu().numpy()
    
    def save(self, checkpoint_path: str) -> None:
        """
        Save the model weights to a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to save the checkpoint
        """
        torch.save(self.commonac.state_dict(), checkpoint_path)

    def saveDescriptive(self, path: str, name: str) -> None:
        """
        Save the model with a descriptive message.
        
        Args:
            path (str): Path to save the model
            name (str): Name of the model
        """
        self.save(path)
        print("Saved", name, "to file", path)
   
    def load(self, checkpoint_path: str) -> None:
        """
        Load model weights from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        dictionary = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.commonac.load_state_dict(dictionary)
        for worker in self.workerACs:
            worker.load_state_dict(dictionary)