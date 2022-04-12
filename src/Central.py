import torch
import torch.optim as optim

import utils


class Central():
	"""docstring for Central"""
	def __init__(self,
				model_name			=	"SVM",
				dataset_name		=	"MNIST",
				batch_size			=	1,
				num_epochs			=	1,
				learning_rate_type	=	"iter_decay"
	):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		num_classes, transform = utils.aux_info(dataset_name, model_name)
		self.trainset, self.testset, input_dim = utils.dataset_info(dataset_name, transform)
		self.model, self.criterion, _ = utils.model_info(model_name, input_dim, num_classes)

		self.model = self.model.to(self.device)

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate_type = learning_rate_type


	def run(self):
		train_loader = torch.utils.data.DataLoader(
			self.trainset,
			batch_size=self.batch_size,
			shuffle=True
		)

		iters, loss_per_iter, accuracy_per_epoch = [], [], []
		epochs = [i for i in range(self.num_epochs)]
		iteration = 0
		for epoch in range(self.num_epochs):
			for dataX, dataY in iter(train_loader):
				lr = utils.calculate_learning_rate(self.learning_rate_type, iteration, epoch, self.batch_size)
				loss = self.gradient_descent(dataX, dataY, lr)
				loss_per_iter.append(float(loss))

				iters.append(iteration)
				iteration += 1

			accuracy = utils.calculate_accuracy(self.model, self.testset)
			accuracy_per_epoch.append(accuracy)

		return {"iters"				:	iters,
				"epochs"			:	epochs,
				"loss_per_iter"		:	loss_per_iter,
				"accuracy_per_epoch":	accuracy_per_epoch}


	def gradient_descent(self, dataX, dataY, learning_rate):
		self.model.train()
		dataX, dataY = dataX.to(self.device), dataY.to(self.device)

		optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
		optimizer.zero_grad()
		output = self.model(dataX)
		loss = self.criterion(output, dataY)
		loss.backward()
		optimizer.step()

		return loss


if __name__ == '__main__':
	simulation = Central(
		model_name			=	"LeNet5",
		dataset_name		=	"FMNIST",
		batch_size			=	1,
		num_epochs			=	3,
		learning_rate_type	=	"iter_decay"
	)
	log = simulation.run()

	print(log["accuracy_per_epoch"])