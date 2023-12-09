import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


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
    
    def visualize_random_sample(self, data_name):
        sample = self.forward(torch.rand(1, 2))

        if data_name == "MNIST" or data_name == "FashionMNIST" :
            sample = sample.view(28, 28).detach().numpy()

            plt.figure()
            plt.imshow(sample, cmap = 'gray_r')
            plt.show()

        elif data_name == "CIFAR10" :
            sample = sample.view(-1, 32,32).detach().numpy()
            sample = sample/2+0.5
            sample = np.transpose(sample, (1,2,0))

            plt.figure()
            plt.imshow(sample)
            plt.show()


    def display_manifold(self, data_name):
        """
        Displays on Z the image of g_theta

        Two assumptions for that :
            - Z is of dimension 2
            - the law zeta on Z is U([0,1])

        """
        sample_per_axis = 20

        z_grid = torch.linspace(0,1,sample_per_axis)

        fig, axes = plt.subplots(sample_per_axis, sample_per_axis, figsize = (10,10))

        z_list = [0.9 for _ in range(2)]
        for i in range(sample_per_axis):
            for j in range(sample_per_axis):

                z_list[0],z_list[1] = z_grid[i],z_grid[j]
                z = torch.tensor(z_list)
                sample = self.forward(torch.unsqueeze(z, 0))

                if data_name == 'MNIST' or data_name == 'FashionMNIST':
                    sample = sample.view(28,28).detach().numpy()

                    axes[i, j].imshow(sample, cmap='gray_r')
                    axes[i, j].axis('off')


                else :
                    sample = sample.view(-1, 32,32).detach().numpy()
                    sample = sample/2+0.5
                    sample = np.transpose(sample, (1,2,0))

                    axes[i, j].imshow(sample)
                    axes[i, j].axis('off')


        plt.subplots_adjust(wspace = 0, hspace = 0)

        plt.show()