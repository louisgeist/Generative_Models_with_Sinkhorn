import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss


class sinkhorn_loss(torch.nn.modules.loss._Loss):

    def __init__(self, fixed_cost : bool , f = None): # size_average = None, reduce = None, reduction : str = 'mean'
        super().__init__(size_average = None, reduce = None, reduction = None)

        self.fixed_cost = fixed_cost
        self.L = 10

        if not self.fixed_cost : 
            self.f = f


    def forward(self, input : torch.Tensor, target :  torch.Tensor, epsilon : float):
        # input and target batch sizes are supposed the same


        batch_size = input.shape[0]
        c = torch.zeros((batch_size,batch_size))

        
            # X = input
            # Y = target

            # X_expanded = X.unsqueeze(1)
            # Y_expanded = Y.unsqueeze(0) 

            # mse_tensor = F.mse_loss(X_expanded, Y_expanded, reduction='none')


        for i in range(batch_size):
            for j in range(batch_size):

                if self.fixed_cost :
                    c[i,j] = F.mse_loss(input[i], target[j], reduction = 'sum')
                else :
                    c[i,j] = F.mse_loss(self.f(input[i]), self.f(target[j]), reduction = 'sum')

        
        K = torch.exp(- c/epsilon)

        one_vector = torch.ones(batch_size)
        b = torch.ones(batch_size)

        for l in range(self.L):
            a = torch.mul(one_vector,1/(K @ b ))
            b = torch.mul(one_vector,1/(K.T @ a))

        return torch.sum((K * c)@ b *a)




