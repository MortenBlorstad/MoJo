import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import numpy as np


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) implementation with beta-VAE support.
    This class implements a VAE with configurable architecture and beta parameter for beta-VAE.
    
    Attributes:
        view_dim (int): Input dimension size
        device (torch.device): Device to run the model on (CPU/GPU)
        latent_dim (int): Dimension of the latent space
        beta (float): Beta parameter for beta-VAE
        encoder (nn.Sequential): Encoder network
        mean_layer (nn.Linear): Layer for mean of latent distribution
        logvar_layer (nn.Linear): Layer for log variance of latent distribution
        decoder (nn.Sequential): Decoder network
        optimizer (torch.optim.Adam): Optimizer for training
    """

    def __init__(self, device: torch.device, input_dim: int, hid1_dim: int, 
                 hid2_dim: int, latent_dim: int, lr: float, beta: float = 1.0) -> None:
        """
        Initialize the VAE.
        
        Args:
            device (torch.device): Device to run the model on
            input_dim (int): Dimension of input data
            hid1_dim (int): Dimension of first hidden layer
            hid2_dim (int): Dimension of second hidden layer
            latent_dim (int): Dimension of latent space
            lr (float): Learning rate for optimizer
            beta (float): Beta parameter for beta-VAE (default: 1.0)
        """
        super(VAE, self).__init__()

        self.view_dim = input_dim
        self.device = device
        self.latent_dim = latent_dim
        self.beta = beta  # Beta parameter for beta-VAE
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hid1_dim),            
            torch.nn.PReLU(),
            nn.Linear(hid1_dim, hid2_dim),            
            torch.nn.PReLU()
        )
        
        # Latent mean and variance layers
        self.mean_layer = nn.Linear(hid2_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(hid2_dim, self.latent_dim)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hid2_dim),            
            torch.nn.PReLU(),
            nn.Linear(hid2_dim, hid1_dim),            
            torch.nn.PReLU(),
            nn.Linear(hid1_dim, input_dim),
            nn.Sigmoid()            
        )
        
        # Initialize optimizer
        self.optimizer = Adam(self.parameters(), lr=lr)
     
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input data into latent space.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance of latent distribution
        """
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent space.
        
        Args:
            mean (torch.Tensor): Mean of latent distribution
            var (torch.Tensor): Variance of latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector back to data space.
        
        Args:
            x (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Reconstructed data
        """
        return self.decoder(x)
    
    def npdecode(self, x: Union[np.ndarray, List[float]]) -> torch.Tensor:
        """
        Decode numpy array or list to data space.
        
        Args:
            x (Union[np.ndarray, List[float]]): Input data in numpy array or list format
            
        Returns:
            torch.Tensor: Reconstructed data
        """
        return self.decoder(torch.Tensor(x).to(self.device))

    def npencode(self, x: Union[np.ndarray, List[float]]) -> torch.Tensor:
        """
        Encode numpy array or list into latent space.
        
        Args:
            x (Union[np.ndarray, List[float]]): Input data in numpy array or list format
            
        Returns:
            torch.Tensor: Mean of latent distribution
        """
        with torch.no_grad():
            mean, _ = self.encode(torch.Tensor(x).to(self.device))
            return mean
        
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on input data.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Mean of latent distribution
        """
        with torch.no_grad():
            mean, _ = self.encode(x)
            return mean
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Reconstructed data
                - Mean of latent distribution
                - Log variance of latent distribution
        """
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.decode(z)  
        return x_hat, mean, log_var    

    def goal_loss(self, x: torch.Tensor, x_hat: torch.Tensor, 
                 mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute the VAE loss function.
        
        Args:
            x (torch.Tensor): Original input data
            x_hat (torch.Tensor): Reconstructed data
            mean (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution
            
        Returns:
            torch.Tensor: Total loss value
        """
        # Reconstruction loss
        mse_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        
        # KL divergence with beta scaling
        kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kld_loss = self.beta * kld_loss  # Scale KL divergence by beta
        
        return mse_loss + kld_loss

    def backwardFromList(self, x: List[torch.Tensor]) -> float:
        """
        Perform backward pass on a list of tensors.
        
        Args:
            x (List[torch.Tensor]): List of input tensors
            
        Returns:
            float: Loss value
        """
        return self.backward(torch.stack(x, dim=0))
    
    def backward(self, x: torch.Tensor) -> float:
        """
        Perform backward pass and update model parameters.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            float: Loss value
        """
        # Reshape input
        x = x.view(-1, self.view_dim).to(self.device)

        self.optimizer.zero_grad()

        x_hat, mean, log_var = self(x)        
        loss = self.goal_loss(x, x_hat, mean, log_var)
        
        rval = loss.item()
        
        loss.backward()
        self.optimizer.step()

        return rval
        
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model state to file.
        
        Args:
            path (Union[str, Path]): Path to save model state
        """
        torch.save(self.state_dict(), path)
    
    def saveDescriptive(self, path: Union[str, Path], name: str) -> None:
        """
        Save model state to file with descriptive message.
        
        Args:
            path (Union[str, Path]): Path to save model state
            name (str): Descriptive name for the model
        """
        self.save(path)
        print("Saved", name, "to file", path)

    @staticmethod
    def __Device() -> torch.device:
        """
        Get the appropriate device (CPU/GPU) for the model.
        
        Returns:
            torch.device: Device to run the model on
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def Create(cfg: Dict[str, Union[int, float, str]]) -> 'VAE':
        """
        Create a new VAE instance from configuration.
        
        Args:
            cfg (Dict[str, Union[int, float, str]]): Configuration dictionary
            
        Returns:
            VAE: New VAE instance
        """
        device = VAE.__Device()

        return VAE(
            device,
            cfg['input_dim'],
            cfg['hid1_dim'],
            cfg['hid2_dim'],
            cfg['latent_dim'],
            cfg['lr'],
            beta=cfg['beta']        
        ).to(device)

    @staticmethod
    def Load(cfg: Dict[str, Union[int, float, str]], EVAL: bool = False) -> 'VAE':
        """
        Load a VAE model from file.
        
        Args:
            cfg (Dict[str, Union[int, float, str]]): Configuration dictionary
            EVAL (bool): Whether to set model to evaluation mode (default: False)
            
        Returns:
            VAE: Loaded VAE instance
        """
        model = VAE.Create(cfg)
        model.load_state_dict(torch.load(cfg['modelfile'], weights_only=True))
        if EVAL:
            model.eval()
        return model