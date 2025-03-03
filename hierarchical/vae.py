import torch
import torch.nn as nn
from torch.optim import Adam


class VAE(nn.Module):

    def __init__(self, device, input_dim, hid1_dim, hid2_dim, latent_dim, lr):
        super(VAE, self).__init__()

        self.view_dim = input_dim
        self.device = device
        self.latent_dim = latent_dim
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hid1_dim),            
            torch.nn.PReLU(),
            nn.Linear(hid1_dim, hid2_dim),            
            torch.nn.PReLU()
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(hid2_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(hid2_dim, self.latent_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hid2_dim),            
            torch.nn.PReLU(),
            nn.Linear(hid2_dim, hid1_dim),            
            torch.nn.PReLU(),
            nn.Linear(hid1_dim, input_dim),
            nn.Sigmoid()            
        )
        
        #Use Adam
        self.optimizer = Adam(self.parameters(), lr=lr)
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.decode(z)  
        return x_hat, mean, log_var    

    def goal_loss(self, x, x_hat, mean, log_var):
        
        mse_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        kld_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return mse_loss + kld_loss


    def loss_function(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')        
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD


    def backward(self,x):
            
        #View [BATCH,1,25,24,24] as [BATCH, 14400]
        x = x.view(-1, self.view_dim).to(self.device)

        self.optimizer.zero_grad()

        x_hat, mean, log_var = self(x)
        #loss = self.loss_function(x, x_hat, mean, log_var)
        loss = self.goal_loss(x, x_hat, mean, log_var)
        
        rval = loss.item()
        
        loss.backward()
        self.optimizer.step()

        return rval
        
    
    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def __Device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def Create():
        device = VAE.__Device()
        model = VAE(device).to(device)
        return model

    @staticmethod
    def Load(PATH, EVAL = False):        
        device = VAE.__Device()
        model = VAE(device).to(device)        
        model.load_state_dict(torch.load(PATH, weights_only=True))
        if EVAL:
            model.eval()
        return model