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


##### to fill ####
model_name = "AGAN_MNIST_40epochs"
data_name = "MNIST"
###################
Model = torch.load(f'./trained_models/{data_name}/{model_name}.pt').to(device = "cpu")

if model_name.split('_')[0] == "GAN":
    print(f"visualize {model_name}")
    Model.eval()
    print("visualize random sample")
    Model.visualize_random_sample(data_name)
    print("visualize manifold")
    Model.display_manifold(data_name)
else: #visualization functions of GANs and Sinkhorn based models are slightly different
    Model.eval()
    sample = Model()
    Model.visualize_random_sample()
    Model.plot_training_loss()
    Model.display_manifold()



