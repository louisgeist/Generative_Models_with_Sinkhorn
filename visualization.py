import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sinkhorn_generative_model import Model

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

Model = torch.load('trained_models/CIFAR10/sinkhorn_eps100_20epoch_.pt').to(device = "cpu") # "mps:0"

Model.eval()
sample = Model()


Model.visualize_random_sample()
Model.plot_training_loss()
Model.display_manifold()

data_name = 'MNIST'
list_epsilon = [1, 10, 100]

for epsilon in list_epsilon :
	Model = torch.load(f'trained_models/{data_name}/sinkhorn_eps{epsilon}_40epoch_.pt').to(device = "cpu")
	Model.eval()
	Model.display_manifold()

