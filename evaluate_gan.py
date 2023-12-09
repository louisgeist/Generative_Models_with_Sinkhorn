import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sinkhorn_generative_model import Model
from gan import Generator, Discriminator
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

#### PARAMETERS #####
batch_size = 200
model_name  = "GAN_MNIST_10epochs" #to save the model weights



############# Load trained generator #############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = torch.load(f'./trained_models/{model_name}.pt').to(device)
Model.eval()
print("visualize random sample")
Model.visualize_random_sample("MNIST")
print("visualize manifold")
Model.display_manifold("MNIST")