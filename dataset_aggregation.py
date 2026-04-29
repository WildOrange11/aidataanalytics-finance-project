from cloning_network import Policy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

#setting up
model = Policy()
#load the policy RELATIVE path into the the model
model.eval()
expert_data = #load it!
#need to do more for setting everything up; not completely done yet!

#carring out DAgger operation
def run_DAgger(x):
    global expert_labels #need to make a previous instance of expert_labels in this file, too

    for cycle in range(x):
        states_new=[]
    
        num_episodes=20
        for episode in range (num_episodes):
            #i want to reset the env--so i need to make an env for it first, higher up in the file
            terminated = False # need to make a condition for terminates, as well

            while terminated == False:
                tensor_state = torch.tensor(state, dtype = torch.float32).unsqueeze(dim = 0)
                
