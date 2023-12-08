import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gan import Generator, Discriminator
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#### PARAMETERS #####
batch_size = 64
lr = 0.0002
num_epochs = 10
model_name  = "GAN_MNIST_10_epochs" #to save the model weights
generator_dim = [[2, 256], [256, 512], [512, 1024], [1024, 784]] # first one should be [sample_dim, _] last one should be [_, 784]
discriminator_dim = [[784, 1024], [1024, 512], [512, 256], [256, 1]] #first one should be [784, _] last one should be [_, 1]
sample_dim = generator_dim[0][0]


#####Load MNIST#####
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    ])

train_dataset = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
flatten = transforms.Lambda(lambda x: x.view(-1))
train_dataset.transform = transforms.Compose([trans, flatten])
dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


############# Create discriminator and generator #############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator = Discriminator(discriminator_dim).to(device)
generator = Generator(generator_dim).to(device)
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()


############# Training #################
true_label = torch.ones(batch_size, 1).to(device)
false_label = torch.zeros(batch_size, 1).to(device)
discriminator_loss_history = []
generator_loss_history = []

for epoch in range(1, num_epochs+1):
    discriminator_batch_loss = 0.0
    generator_batch_loss = 0.0
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]
        discriminator.zero_grad()

        #Train discriminator on real data
        d_real_predict = discriminator(data.view(data.shape[0], -1))
        d_real_loss = criterion(d_real_predict, true_label[:batch_size])

        # Train discriminator on fake data from generator
        d_fake_noise = torch.rand(batch_size, sample_dim).to(device)
        d_fake_input = generator(d_fake_noise).detach() #avoid training the generator
        d_fake_predict = discriminator(d_fake_input)
        d_fake_loss = criterion(d_fake_predict, false_label[:batch_size])

        discriminator_loss = d_real_loss + d_fake_loss
        discriminator_batch_loss += discriminator_loss.item()
        discriminator_loss.backward()
        optimizerD.step()


        # Train generator
        g_fake_noise = torch.rand(batch_size, sample_dim).to(device)
        g_fake_input = generator(g_fake_noise)
        generator.zero_grad()
        g_fake_predict = discriminator(g_fake_input)
        generator_loss = criterion(g_fake_predict, true_label[:batch_size])
        generator_batch_loss += generator_loss.item()
        generator_loss.backward()
        optimizerG.step()

        # print loss every 100 batches
        if (batch_idx + 1) % 100 == 0:

            print(f'Epoch [{epoch}/{num_epochs}]  Batch {batch_idx + 1}/{len(dataloader)} \
                    Loss D: {discriminator_loss:.4f}, Loss G: {generator_loss:.4f}')


    discriminator_loss_history.append(discriminator_batch_loss / (batch_idx + 1))
    generator_loss_history.append(generator_batch_loss / (batch_idx + 1))

torch.save(generator, f"./trained_model/{model_name}.pt")
