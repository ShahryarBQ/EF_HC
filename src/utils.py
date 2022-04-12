import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms

from LeNet5 import LeNet5


def aux_info(dataset_name, model_name):
	if dataset_name in ["MNIST", "FMNIST"]:
		num_classes = 10

	if model_name == "SVM":
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
			transforms.Lambda(lambda x: torch.flatten(x))
		])
	if model_name == "LeNet5":
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
			transforms.Resize((32, 32))
		])

	return num_classes, transform


def dataset_info(dataset_name, transform):
	if dataset_name == "MNIST":
		trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
		testset = datasets.MNIST('../data', train=False, download=True, transform=transform)

	if dataset_name == "FMNIST":
		trainset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
		testset = datasets.FashionMNIST('../data', train=False, download=True, transform=transform)

	input_dim = calculate_input_dim(trainset[0][0].shape)
	return list(trainset), list(testset), input_dim


def model_info(model_name, input_dim, num_classes):
	if model_name == "SVM":
		model = torch.nn.Linear(input_dim, num_classes)
		criterion = torch.nn.MultiMarginLoss()

	if model_name == "LeNet5":
		model = LeNet5(num_classes)
		criterion = torch.nn.CrossEntropyLoss()

	model_dim = calculate_model_dim(model.parameters())
	return model, criterion, model_dim


def calculate_input_dim(shape):
	dim = 1
	for ax in shape:
		dim *= ax
	return dim


def calculate_model_dim(model_params):
	model_dim = 0
	for param in model_params:
		model_dim += len(param.flatten())
	return model_dim


def calculate_learning_rate(learning_rate_type, iteration, epoch, batch_size):
	if learning_rate_type == "constant":
		lr = 0.01
	if learning_rate_type == "iter_decay":
		lr = 1 / np.sqrt(1 + iteration)
	if learning_rate_type == "epoch_decay":
		lr = 0.01 / (1 + epoch)
	if learning_rate_type == "data_decay":
		lr = 1 / np.sqrt(1 + iteration*batch_size)

	return lr


def calculate_accuracy(model, testset):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model.eval()
	test_loader = torch.utils.data.DataLoader(
		testset,
		batch_size=32,
		shuffle=True
	)

	correct = 0
	for dataX, dataY in iter(test_loader):
		dataX, dataY = dataX.to(device), dataY.to(device)
		output = model(dataX)
		pred = output.argmax(dim=1)
		correct += (pred == dataY).int().sum().item()
		
	return correct / len(testset)


def moving_average(x, y, window=32):
	if len(x) <= window:
		return x, y

	output_x = x[window-1:]

	val = 0
	for i in range(window):
		val += y[i] / window
	output_y = [val]
	for i in range(window, len(y)):
		val += (y[i] - y[i - window]) / window
		output_y.append(val)

	return output_x, output_y


def moving_average_df(dfs_dict, index="iters", window=32):
	dfs_dict_ret = {}

	for threshold_type, df in dfs_dict.items():
		dfs_dict_ret[threshold_type] = pd.DataFrame()

		for column_name in df:
			if column_name == index:
				x, _ = moving_average(df[index], df[index], window)
				dfs_dict_ret[threshold_type][index] = x

			_, y = moving_average(df[index], df[column_name], window)
			dfs_dict_ret[threshold_type][column_name] = y

	return dfs_dict_ret