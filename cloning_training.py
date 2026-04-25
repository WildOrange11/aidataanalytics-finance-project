import torch
import torch.nn as nn
import torch.nn.functional as function
import torch.optim as optim
import numpy as np
import cloning_network
from cloning_network import Policy

#THIS IS FOR BASIC BEHAVIOURAL CLONING, NOT DAGGER YET!

#loading the expert data
#need to create a file which is the expert data, takes our current data and then assigns labels, using those labels, assigns an output accordingly
LABELS_PATH = #need relative path to that file here
expert_labels = torch.load(LABELS_PATH)
expert_labels = expert_labels.float() # so that you can do calculations with this

#model obj
model = Policy()

#loss function -- you want to minimize this!
loss = nn.CrossEntropyLoss()

#optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.003)

#training loop
epoch = 0
num_epoch = 100
while epoch<=num_epoch:
    optimizer.zero_grad()
    logits = model(expert_labels)
    loss_value = loss(logits)
    loss_value.backward()
    optimizer.step()
    if epoch %2 ==0:
        print(loss_value.item())
    epoch+=1

torch.save(model.state_dict(), "bc_policy")
