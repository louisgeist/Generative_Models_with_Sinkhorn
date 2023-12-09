import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sinkhorn_generative_model import Model

import matplotlib.pyplot as plt

from geomloss import SamplesLoss

batch_size = 200

data_name = "MNIST"

def evaluate_with_sinkhorn(data_name, models : list):
	"""
	For each model in models contained in the folder trained_modes/{data_name}/,
	computes the sinkhorn loss between 200 samples generated by the model and 
	200 samples of the test set.

	Returns the result in a list.

	"""

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
	    test_dataset = datasets.MNIST(root='./data', train=False, transform=trans2D, download=True)
	    test_dataset.transform = transforms.Compose([trans2D, flatten])
	    dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
	elif data_name == "FashionMNIST":
	    test_dataset = datasets.FashionMNIST(root='./data', train=True, transform=trans2D, download=True)
	    test_dataset.transform = transforms.Compose([trans2D, flatten])
	    dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


	Loss = SamplesLoss("sinkhorn", p = 2, blur = 0.05, scaling = 0.8, backend="tensorized")

	list_loss = []

	for model in models:
		Model = torch.load(f'trained_models/{data_name}/{model}.pt')

		Wass_xy = 0
		for i,data in enumerate(dataloader):
			Y,_ = data
			X = Model.forward_batch()

			Wass_xy += Loss(X, Y).item()


		Wass_xy = Wass_xy/(i+1)
		list_loss.append(Wass_xy)

	return list_loss


models = ['sinkhorn_eps1_40epoch_','sinkhorn_eps10_40epoch_','sinkhorn_eps100_40epoch_']
list_loss = evaluate_with_sinkhorn("MNIST", models)
print(list_loss)

