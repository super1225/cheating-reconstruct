import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

class NET(nn.Module):
    def __init__(self,input_size=2,hidden_size=64,output_size=5,num_layer=2):
        super(NET,self).__init__()
        self.rnn=nn.LSTM(input_size,hidden_size,num_layer)
        self.out=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        out,_=self.rnn(x)
        #print(out)
        out=self.out(out[:,-1,:])
        return out

