import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gen_model_with_sinkhorn import Model

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

batch_size = 32
epochs = 10
generator_dim = [[2,32], [32,128], [128, 784]] #last one should be [_,784]
learned_cost_dim = [[784, 128], [128, 128]] #first one should be [784, _]
lr = 0.01
learnable_cost = False
epsilon = 0.1


Model = Model(generator_dim, learned_cost_dim, batch_size, lr, epsilon, learnable_cost)
Model = torch.load('basic_model.pt')

sample = Model()

def visualize(sample : torch.Tensor):
	sample = sample.view(28, 28).detach().numpy()


	plt.figure()
	plt.imshow(sample, cmap = 'gray')
	plt.show()

visualize(sample)