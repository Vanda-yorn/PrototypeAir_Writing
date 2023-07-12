import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim



class SimpleNet(nn.Module):
  def __init__(self, layers_sz, in_sz, out_sz):
    super(SimpleNet, self).__init__()
    layers = []
    for sz in layers_sz:
      layers.append(nn.Linear(in_sz, sz))
      in_sz = sz
    self.linears = nn.ModuleList(layers)
    self.out = nn.Linear(layers_sz[-1], out_sz)
    self.act_func = nn.LeakyReLU()
  

  def forward(self, x):
    for l in self.linears:
      x = self.act_func(l(x))
    return self.out(x)
