import torch
import torch.nn as nn
import torch.nn.functional as function

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(17, 3) #i want this to output major+year, gender, a financial score / 100
            #havent finished that code yet tho, working on it
        self.layer2 = nn.Linear(3,32) #making sure that those 3 things are done really well right now...
        self.layer3 = nn.Linear(32,32) # i want to make sure that's working really well
        self.layer4 = nn.Linear(32,2) # i want to compress it to 2 things, which are going to be a financial score/100, and what to work on, represented by a number
            #havent finished making that code yet either

    def forward(self, pass_):
        pass_ = function.relu(self.layer1(pass_))
        pass_ = function.relu(self.layer2(pass_))
        pass_ = function.relu(self.later3(pass_))
        pass_=self.layer4(pass_)
        return pass_