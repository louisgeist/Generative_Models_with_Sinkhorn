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


Model = torch.load('basic_model.pt')

Model.eval()
sample = Model()

def visualize(sample : torch.Tensor):
	sample = sample.view(28, 28).detach().numpy()


	plt.figure()
	plt.imshow(sample, cmap = 'gray')
	plt.show()

visualize(sample)