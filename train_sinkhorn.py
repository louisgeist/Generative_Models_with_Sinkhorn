import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sinkhorn_generative_model import Model

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(device)


######## PARAMETERS ###########
model_name  = "basic_model_912"
batch_size = 200
epochs = 3
lr = 0.001
learnable_cost = False
epsilon = 1

data_name = "MNIST" #to adapt

if data_name == "CIFAR10":
    output_dim = 3072
elif data_name == "MNIST":
    output_dim = 784
elif data_name == "FashionMNIST":
    output_dim = 784
else :
    print("not correct dataset name")

generator_dim = [[2, 256], [256, 512], [512, 1024], [1024, output_dim]] #last one should be [_,784]
learned_cost_dim = [[output_dim, 128], [128, 128]] #first one should be [784, _]



model = Model(generator_dim, learned_cost_dim, batch_size, lr, epsilon, learnable_cost, device)
#model = torch.load('basic_model.pt') # in order to continue the training

######## Load data  ########
trans2D = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    ])

trans3D = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
])

flatten = transforms.Lambda(lambda x: x.view(-1))

if data_name == "MNIST":
    train_dataset = datasets.MNIST(root='./data', train=True, transform=trans2D, download=True)
    train_dataset.transform = transforms.Compose([trans2D, flatten])
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
if data_name == "FashionMNIST":
    train_dataset = datasets.MNIST(root='./data', train=True, transform=trans2D, download=True)
    train_dataset.transform = transforms.Compose([trans2D, flatten])
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
if data_name == "CIFAR10":
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=trans3D, download=True)
    train_dataset.transform = transforms.Compose([trans3D, flatten])
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


####### Training ##########
training_logs = np.zeros((epochs,3))
start = time.time()
for epoch in range(1,epochs+1):
    model.train(True)
    loss = model.train_1epoch(dataloader)
    end = time.time()
    epoch_duration = round(end-start,2)
    print(f"Epoch {epoch} ({epoch_duration} s): loss = {loss}")
    start = end

    training_logs[epoch-1,:] = np.array([epoch,loss,epoch_duration])


model.set_training_logs(training_logs)
torch.save(model, f"trained_models/{model_name}.pt")
