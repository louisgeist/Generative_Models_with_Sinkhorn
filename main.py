import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sinkhorn_gen_model import Model



import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#PARAMETERS
batch_size = 64
epochs = 5
generator_dim = [[2,32], [32,128], [128, 784]] #last one should be [_,784]
learned_cost_dim = [[784, 128], [128, 128]] #first one should be [784, _]
criterion = nn.CrossEntropyLoss()
lr = 0.001
"""
ducoup la tu dois supprimer criterion je pense (dans Model() aussi)
"""
model = Model(generator_dim, learned_cost_dim, batch_size, criterion, lr)

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
for epoch in range(epochs):
    model.train(True)
    loss = model.train_1epoch(train_dataloader)
    print(f"epoch{epoch}: loss = {loss}\n")

"""
c'est normal si la loss ne decroit pas car utilisé la crossentropy a aucun sens
mais au moin on sait que le model fonctionne!

j'ai pas fait de boucle de test car il faut choisir comment on evalue nos model
il me reste a faire le GAN, je pense faire ca demain.
"""

