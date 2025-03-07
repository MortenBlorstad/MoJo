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
    
    def npdecode(self, x):
        return self.decoder(torch.Tensor(x).to(self.device))

    def npencode(self, x):
        with torch.no_grad():
            mean, _ = self.encode(torch.Tensor(x).to(self.device))
            return mean
        
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.decode(z)  
        return x_hat, mean, log_var    


    def goal_loss(self, x, x_hat, mean, log_var):
        
        mse_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        kld_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        
        return mse_loss + kld_loss

    
    def backwardFromList(self,x):
        return self.backward(torch.stack(x, dim=0))
    

    def backward(self,x):
            
        #View [BATCH,1,25,24,24] as [BATCH, 14400]
        x = x.view(-1, self.view_dim).to(self.device)

        self.optimizer.zero_grad()

        x_hat, mean, log_var = self(x)        
        loss = self.goal_loss(x, x_hat, mean, log_var)
        
        rval = loss.item()
        
        loss.backward()
        self.optimizer.step()

        return rval
        
    
    def save(self, path):        
        torch.save(self.state_dict(), path)
    
    def saveDescriptive(self, path, name):        
        self.save(path)
        print("Saved",name,"to file",path)

    @staticmethod
    def __Device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def Create(cfg):

        device = VAE.__Device()

        return VAE(
            device,
            cfg['input_dim'],
            cfg['hid1_dim'],
            cfg['hid2_dim'],
            cfg['latent_dim'],
            cfg['lr']        
        ).to(device)

    @staticmethod
    def Load(cfg, EVAL = False):  

        model = VAE.Create(cfg)
        model.load_state_dict(torch.load(cfg['modelfile'], weights_only=True))
        if EVAL:
            model.eval()
        return model