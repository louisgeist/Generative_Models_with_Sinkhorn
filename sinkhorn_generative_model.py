import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss_sinkhorn import sinkhorn_loss

import matplotlib.pyplot as plt
import numpy as np

import time

import matplotlib.pyplot as plt

#simple fully connected generator and learnable cost function
class FC_net(nn.Module):
    def __init__(self, dimensions, dropout_prob=0.2):
        super(FC_net, self).__init__()
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
    

class Model(nn.Module):
    def __init__(self, generator_dim, learned_cost_dim, batch_size, lr, epsilon, learnable_cost = False, device = "cpu", data_name = None):
        super(Model, self).__init__()
        self.device = device
        self.data_name = data_name

        self.batch_size = batch_size
        self.criterion = sinkhorn_loss(learnable_cost, epsilon, device = self.device)

        self.generator = FC_net(generator_dim).to(self.device)
        self.optimizer = optim.Adam(self.generator.parameters(), lr)
        self.sample_dim = generator_dim[0][0]

        self.learnable_cost = learnable_cost # Boolean           
        if self.learnable_cost:
            self.learned_cost = FC_net(learned_cost_dim).to(self.device)
            self.cost_optimizer = optim.Adam(self.learned_cost.parameters(), lr, maximize = True)

        self.training_logs = []



    def forward(self):
        z = torch.rand((1,self.sample_dim), device = self.device)
        x = self.generator(z)
        return x

    def forward_batch(self):
        z = torch.rand(self.batch_size, self.sample_dim, device = self.device)
        x = self.generator(z)
        return x

    def deterministic_foward(self,z):
        z = z.unsqueeze(0)
        x = self.generator(z).to(self.device)
        return x



    ### --- Training methods ---

    def train_1epoch(self, training_loader):
        """
        One epoch training of the generative model.
        If we learn the cost function (self.learnable_cost = True), then 
        each epoch of the generative model is first composed of n_c (fixed in self. train_1epoch_cost) 
        training steps of the cost.

        """
        running_loss = 0

        for k, data in enumerate(training_loader):
            start = time.time()
            if self.learnable_cost:
                opposite_loss = self.train_1epoch_cost(training_loader)
                end = time.time()
                #print(f"Iteration {k} ({round(end-start,2)} s): opposite loss = {opposite_loss}")
                start = end
                                
            x = self.forward_batch()
            y = data[0].to(self.device)

            if self.learnable_cost:
                x = self.learned_cost(x)
                y = self.learned_cost(y)
            

            self.optimizer.zero_grad()
            loss = 2 * self.criterion(x,y) - self.criterion(x,x) - self.criterion(y,y)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return running_loss

    def train_1epoch_cost(self,training_loader):
        n_c = 20

        for k, data in enumerate(training_loader):
            if k >=n_c:
                break

            self.cost_optimizer.zero_grad()

            x = self.forward_batch()
            y = data[0].to(self.device)

            x = self.learned_cost(x)
            y = self.learned_cost(y)

            loss =  2 * self.criterion(x,y) - self.criterion(x,x) - self.criterion(y,y)
            loss.backward()
            #print(loss.item())
            self.cost_optimizer.step()

            for param in self.learned_cost.parameters():
                param.data = torch.clip(param.data, min=-10, max=10)

        return loss


    ### --- display results ---
    def set_training_logs(self, training_logs):
        self.training_logs = training_logs

    def get_training_logs(self):
        return self.training_logs

    def plot_training_loss(self):
        plt.figure()
        plot = plt.plot(self.get_training_logs()[:,0],self.get_training_logs()[:,1])
        plt.show()

    def display_manifold(self):
        """
        Displays on Z the image of g_theta

        Two assumptions for that :
            - Z is of dimension 2
            - the law zeta on Z is U([0,1])

        """
        sample_per_axis = 20

        z_grid = torch.linspace(0,1,sample_per_axis)

        fig, axes = plt.subplots(sample_per_axis, sample_per_axis, figsize = (10,10))

        for i in range(sample_per_axis):
            for j in range(sample_per_axis):

                z = torch.tensor([z_grid[i],z_grid[j]], device = self.device)
                sample = self.deterministic_foward(z).to(self.device)

                if self.data_name == 'MNIST' or self.data_name == 'FashionMNIST':
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





        



