import sys
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hierarchical.config import Config
from hierarchical.utils import InitVAE, getfiles


cfg = Config().Get("Director")["Worldmodel"]
wm = InitVAE(cfg)

#Load universe states from np files
x = []
for path in getfiles(cfg['datapath']):
    with open(path, 'rb') as f:
        a = np.load(f)
        x.append(a)
        #print("Loaded",a.shape,"from",path)    

#Concat into one big array
x = np.nan_to_num(np.concat(x))

#Create dummy y's (it's a VAE we're training)
y = np.ones((x.shape[0],1))

#create dataset & loader
batch_size = 100
state_dataset = TensorDataset(torch.Tensor(x),torch.Tensor(y))
train_loader = DataLoader(dataset=state_dataset, batch_size=batch_size, shuffle=True)

#training routine
def train(model, epochs):
    model.train()
    #for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            overall_loss += model.backward(x)
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
    return overall_loss     

#Execute

print("Starting training. Shape is",x.shape," Bat size is",batch_size)

train(wm, epochs=200)

#Save model & weights
wm.save(cfg['modelfile'])