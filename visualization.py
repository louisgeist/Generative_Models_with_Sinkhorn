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

Model = torch.load('trained_models/basic_model.pt').to(device = "cpu") # "mps:0"

Model.eval()
sample = Model()

# plt.figure()
# plt.plot(Model.get_training_logs()[:,0],Model.get_training_logs()[:,1])
# plt.show()

Model.plot_training_loss()

#Model.display_manifold()

def visualize_random_sample(sample : torch.Tensor):
	sample = sample.view(28, 28).detach().numpy()

	plt.figure()
	plt.imshow(sample, cmap = 'gray')
	plt.show()

#visualize_random_sample(sample)

def visualize_random_sample_cifar10(sample : torch.Tensor):
	sample = sample.view(-1, 32,32).detach().numpy()
	sample = sample/2+0.5
	sample = np.transpose(sample, (1,2,0))

	plt.figure()
	plt.imshow(sample)
	plt.show()

# CIFAR 10 - to see images of the training set
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
# ])
# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=True)

# for i,(images, labels) in enumerate(test_dataloader):
# 	if i>=0 : break
# 	visualize_random_sample_cifar10(images[0])

# sample = Model()
# visualize_random_sample_cifar10(sample)




