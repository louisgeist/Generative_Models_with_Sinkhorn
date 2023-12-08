import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sinkhorn_generative_model import Model

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(device)


#PARAMETERS
model_name  = "GAN_MNIST_10_epochs"
batch_size = 200
epochs = 40
#generator_dim = [[2,32], [32,256], [256, 784]] 
generator_dim = [[2, 256], [256, 512], [512, 1024], [1024, 784]] #last one should be [_,784]
learned_cost_dim = [[784, 128], [128, 128]] #first one should be [784, _]
lr = 0.01
learnable_cost = False
epsilon = 1

model = Model(generator_dim, learned_cost_dim, batch_size, lr, epsilon, learnable_cost, device)

#model = torch.load('basic_model.pt') # in order to continue the training

#to use normalized version of MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

# Download and load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)


# Flatten 28x28 image into a vector= of 784 fetures
flatten = transforms.Lambda(lambda x: x.view(-1))

# Apply the flatten transformation to the dataset
train_dataset.transform = transforms.Compose([transform, flatten])
test_dataset.transform = transforms.Compose([transform, flatten])

# Create dataloaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#training 
start = time.time()

training_logs = np.zeros((epochs,3))
for epoch in range(1,epochs+1):
    model.train(True)
    loss = model.train_1epoch(train_dataloader)
    end = time.time()
    epoch_duration = round(end-start,2)
    print(f"Epoch {epoch} ({epoch_duration} s): loss = {loss}")
    start = end

    training_logs[epoch-1,:] = np.array([epoch,loss,epoch_duration])

model.set_training_logs(training_logs)

torch.save(model, "trained_models/basic_model.pt")
