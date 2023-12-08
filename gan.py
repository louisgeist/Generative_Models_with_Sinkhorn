import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, dimensions, dropout_prob=0.2):
        super(Discriminator, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(dim_in, dim_out) 
            for dim_in, dim_out in dimensions
        ])
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            # Apply ReLU activation except for the last layer
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)    
            # Apply dropout between layers
            x = self.dropout(x)
        return F.sigmoid(x)

class Generator(nn.Module):
    def __init__(self, dimensions, dropout_prob=0.2):
        super(Generator, self).__init__()
        self.fc_layers = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(dimensions):
            if i < len(dimensions) - 1:
                layer = nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.BatchNorm1d(dim_out),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_prob)
                )
            else: # no batch norm on last layer
                layer = nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.Tanh()  
                )
            self.fc_layers.append(layer)

    def forward(self, x):
      for layer in self.fc_layers:
          x = layer(x)
      return x