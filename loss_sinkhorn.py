import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss


class sinkhorn_loss(torch.nn.modules.loss._Loss):

    def __init__(self, learnable_cost : bool, epsilon : float, f = None): # size_average = None, reduce = None, reduction : str = 'mean'
        super().__init__(size_average = None, reduce = None, reduction = None)

        self.learnable_cost = learnable_cost
        self.epsilon = epsilon
        self.L = 10


        if self.learnable_cost : 
            self.f = f



    def forward(self, input : torch.Tensor, target :  torch.Tensor):
        # input and target batch sizes are supposed the same
        if (input.shape != target.shape):
            print("Error : x shape ", input.shape," should be the same as y shape ", target.shape)


        batch_size = input.shape[0]
        c = torch.zeros((batch_size,batch_size))


        for i in range(batch_size):
            for j in range(batch_size):

                if not self.learnable_cost :
                    c[i,j] = F.mse_loss(input[i], target[j], reduction = 'mean')
                else :
                    c[i,j] = F.mse_loss(self.f(input[i]), self.f(target[j]), reduction = 'mean')

        #c = torch.clip(c/self.epsilon, min = 0, max = 100)
        K = torch.exp(- c/self.epsilon)

        one_vector = torch.ones(batch_size)
        b = torch.ones(batch_size)

        for l in range(self.L):
            a = torch.mul(one_vector,1/(K @ b ))
            b = torch.mul(one_vector,1/(K.T @ a))

        return torch.sum((K * c)@ b *a)




