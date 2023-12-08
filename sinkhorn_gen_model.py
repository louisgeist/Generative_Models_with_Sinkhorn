import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    
"""
attention le param is_learned_cost est mit a False par defaut
"""
class Model(nn.Module):
    def __init__(self, generator_dim, learned_cost_dim, batch_size, criterion, lr, is_learned_cost = False):
        super(Model, self).__init__()
        self.generator = FC_net(generator_dim)
        self.is_learned_cost = is_learned_cost
        if self.is_learned_cost:
            self.learned_cost = FC_net(learned_cost_dim)
        self.sample_dim = generator_dim[0][0]
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optim.Adam(self.parameters(), lr)


    def forward(self):
        z = torch.randn(self.batch_size, self.sample_dim) # N(0, I_d)
        x = self.generator(z)
        return x
    
    def train_1epoch(self, training_loader):
        running_loss = 0
        for i, data in enumerate(training_loader):
            y, _ = data #we don't care about label

            self.optimizer.zero_grad()

            #sample z and generate x
            x = self()

            #if needed compute the learned cost
            if self.is_learned_cost:
                f_x = self.learned_cost(x)
                f_y = self.learned_cost(y)
            """
            la normalement tu devrais avoir a faire qqch
            """

            # Calculate loss
            """
            ici j'ai pris une loss pas adapté pour tester si le model run
            normalement tu dois remplacer cette partie 
            """
            loss = self.criterion(x, y)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            return running_loss




        





