import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss_sinkhorn import sinkhorn_loss

#simple fully connected generator and learnable cost function
class FC_net(nn.Module):
    def __init__(self, dimensions, dropout_prob=0.2):
        super(FC_net, self).__init__()
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
        return x
    

class Model(nn.Module):
    def __init__(self, generator_dim, learned_cost_dim, batch_size, lr, epsilon, learnable_cost = False):
        super(Model, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generator = FC_net(generator_dim)
        self.learnable_cost = learnable_cost
        if self.learnable_cost:
            self.learned_cost = FC_net(learned_cost_dim)
        self.sample_dim = generator_dim[0][0]
        self.batch_size = batch_size
        self.criterion = sinkhorn_loss(learnable_cost, epsilon).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr)

        if learnable_cost:
            self.sinkhorn_loss_optimizer = optim.Adam(self.criterion.parameters(), lr)

        

    def forward(self):
        z = torch.randn(self.sample_dim) # N(0, I_d)
        x = self.generator(z)
        return x

    def forward_batch(self):
        z = torch.randn(self.batch_size, self.sample_dim)
        x = self.generator(z)
        return x
    
    def train_1epoch(self, training_loader):
        running_loss = 0

        for k in range(100):
        #for i, data in enumerate(training_loader):

            if self.learnable_cost: #self.criterion = sinkhorn_loss
                f_x = self.learned_cost(x)
                f_y = self.learned_cost(y)

                n_c = 10 
                for t in range(n_c):
                    self.sinkhorn_loss_optimizer.zero_grad()

                    x = self.forward_batch().to(self.device)

                    dataiter = iter(training_loader)
                    y, _ = next(dataiter)
                    y = y.to(self.device)

                    opposite_loss =  -(2 * self.criterion(x,y) - self.criterion(x,x) - self.criterion(y,y))
                    
                    opposite_loss.backward()
                    self.sinkhorn_loss_optimizer.step()

                    self.criterion.parameters = torch.clip(self.criterion.parameters, min = - 10, max = 10) 
            
            x = self.forward_batch().to(self.device)

            dataiter = iter(training_loader)
            y, _ = next(dataiter)
            y = y.to(self.device)
            

            self.optimizer.zero_grad()
            loss = 2 * self.criterion(x,y) - self.criterion(x,x) - self.criterion(y,y)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            if k==99: 
                print("End of the epoch due to the end of the for loop on k.")

        return running_loss




        





