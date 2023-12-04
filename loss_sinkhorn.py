import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss


class sinkhorn_loss(torch.nn.modules.loss._Loss):

    def __init__(self, learnable_cost : bool, epsilon : float, f = None, device = torch.device("cpu")): # size_average = None, reduce = None, reduction : str = 'mean'
        super().__init__(size_average = None, reduce = None, reduction = None)

        self.device = device
        self.epsilon = epsilon
        self.L = 10

    def forward(self, input : torch.Tensor, target :  torch.Tensor):
        input_batch_size = input.shape[0]
        target_batch_size = target.shape[0]

        # use of |x-y|^2 = |x|^2 - 2<x,y> + |y|^2
        dots = input @ target.T
        input_norm = (input**2).sum(dim = 1)
        target_norm = (target**2).sum(dim = 1)

        c = input_norm.view(input_batch_size,-1) - 2*dots + target_norm.view(1,target_batch_size)
        c = torch.clip(c,min = 1e-16)
        c = c**(1/2)

        K = torch.exp(- c/self.epsilon)

        b = torch.ones(target_batch_size, device = self.device)/target_batch_size

        for l in range(self.L):
            a = torch.ones(input_batch_size, device = self.device)/input_batch_size * 1/(K @ b )
            b = torch.ones(target_batch_size, device = self.device)/target_batch_size * 1/(K.T @ a)


        return torch.sum((K * c)@ b *a)




