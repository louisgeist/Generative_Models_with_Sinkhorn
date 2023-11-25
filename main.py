import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gen_model_with_sinkhorn import Model

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#PARAMETERS
batch_size = 64
epochs = 20
generator_dim = [[2,32], [32,128], [128, 784]] #last one should be [_,784]
learned_cost_dim = [[784, 128], [128, 128]] #first one should be [784, _]
lr = 0.01
learnable_cost = False
epsilon = 0.1

model = Model(generator_dim, learned_cost_dim, batch_size, lr, epsilon, learnable_cost)

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
for epoch in range(1,epochs+1):
    model.train(True)
    loss = model.train_1epoch(train_dataloader)
    print(f"epoch{epoch}: loss = {loss}\n")

torch.save(model, "basic_model.pt")

"""
j'ai pas fait de boucle de test car il faut choisir comment on evalue nos model
il me reste a faire le GAN, je pense faire ca demain.
"""