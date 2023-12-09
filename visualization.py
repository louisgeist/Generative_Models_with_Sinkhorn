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

Model = torch.load('trained_models/MNIST/sinkhorn_eps1_20epoch_.pt').to(device = "cpu") # "mps:0"

Model.eval()
sample = Model()


Model.visualize_random_sample()
Model.plot_training_loss()
Model.display_manifold()



