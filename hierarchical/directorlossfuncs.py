
import torch
import torch.nn.functional as F
import numpy as np

#Implementation of max-cosine reward
#https://arxiv.org/pdf/2206.04114
def MaxCosine(goal, state):

    eps=torch.tensor(1e-8)
    
    #Compute vector norms
    goal_normed = torch.sqrt(torch.sum(goal**2))
    state_normed = torch.sqrt(torch.sum(state**2))

    #Compute max of the two norms and assure not null using epsilon
    m = torch.max(goal_normed,state_normed)
    m = torch.max(m,eps)

    #return g/m * s/m
    return torch.sum(goal/m) * torch.sum(state/m) 

#NP version of the same function
def MaxCosineNP(goal, state):

    eps=1e-8
    
    #Compute vector norms
    goal_normed = np.sqrt(np.sum(goal**2))
    state_normed = np.sqrt(np.sum(state**2))

    #Compute max of the two norms and assure not null using epsilon
    m = max(goal_normed,state_normed)
    m = max(m,eps)

    #return g/m * s/m
    return np.sum(goal/m) * np.sum(state/m) 

'''
#Example:

#With this implementation the vectors could differ in dimensions:

goal = torch.tensor([.1,.2,.3,.4,.5])
state = torch.tensor([.1,.2,.3,.4,.5,.6,.7,.8,.9])
print(MaxCosine(goal, state))
'''

#Implementation of exploration reward from paper
#https://arxiv.org/pdf/2206.04114

#Goal is the output of a reconstruction dec_φ(z) 
#State is the observed s_t+1
#NP version....
def ExplorationReward(goal, state):
   
    #ax = 1 if goal.ndim > 1 else 0
    #return ((goal-state)**2).sum(axis = ax)
    return ((goal-state)**2).sum().item()
