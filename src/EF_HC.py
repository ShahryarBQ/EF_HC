import networkx as nx
from copy import deepcopy
from threading import Thread
from random import sample, randint, choices

import utils
from Agent import Agent


class EF_HC():
	"""docstring for EF_HC"""
	def __init__(self,
				model_name,
				dataset_name,
				num_epochs					=	1,
				num_agents					=	10,
				graph_connectivity			=	0.2,
				system_bandwidth			=	10*5000,
				system_bandwidth_type		=	"two_slice",
				system_bandwidth_parameter	=	0.8,
				data_distribution			=	"iid",
				labels_per_agent			=	None,
				batch_size					=	1,
				learning_rate_type			=	"iter_decay",
				r							=	1
	):
		# Agent-level properties
		self.model_name = model_name
		self.num_classes, transform = utils.aux_info(dataset_name, model_name)
		self.trainset, testset, self.input_dim = utils.dataset_info(dataset_name, transform)
		# model, criterion, self.model_dim = utils.model_info(model_name, self.input_dim, self.num_classes)

		self.num_epochs = num_epochs

		# System-level properties
		self.num_agents = num_agents
		self.graph_connectivity = graph_connectivity
		self.system_bandwidth = system_bandwidth
		self.system_bandwidth_type = system_bandwidth_type
		self.system_bandwidth_parameter = system_bandwidth_parameter
		self.data_distribution = data_distribution
		self.labels_per_agent = labels_per_agent

		graph = self.generate_graph(graph_connectivity)
		bandwidths = self.generate_bandwidths(system_bandwidth_parameter)
		trainsets = self.generate_trainsets(data_distribution, labels_per_agent)
		models, criterion, self.model_dim = self.generate_models(model_name)

		self.agents = self.generate_agents(graph, bandwidths, models, criterion, trainsets,
				testset, batch_size, learning_rate_type, r)


	def generate_graph(self, graph_connectivity):
		while True:
		    graph = nx.random_geometric_graph(self.num_agents, graph_connectivity)
		    if nx.is_k_edge_connected(graph, 1) == True:
		        break

		return graph


	def generate_bandwidths(self, system_bandwidth_parameter):
		# system_bandwidth_parameter is "k" if system_bandwidth_type is "random"
		# system_bandwidth_parameter is "weak_ratio" if system_bandwidth_type is "two_slice"

		if self.system_bandwidth_type == "random":
			bandwidths = self.device_data_random(system_bandwidth_parameter)
		if self.system_bandwidth_type == "two_slice":
			bandwidths = self.device_data_two_slice(system_bandwidth_parameter)

		return bandwidths


	def generate_trainsets(self, data_distribution, labels_per_agent=None):
		shuffled = sample(self.trainset, k=len(self.trainset))

		if data_distribution == "iid":
			trainsets = self.separate_data_iid(shuffled)
		if data_distribution == "non_iid":
			trainsets = self.separate_data_non_iid(shuffled)
		if data_distribution == "labels_per_agent":
			trainsets = self.separate_data(shuffled, labels_per_agent)

		return trainsets


	def generate_models(self, model_name):
		models = []
		for _ in range(self.num_agents):
			model, criterion, model_dim = utils.model_info(model_name, self.input_dim, self.num_classes)
			models.append(model)
		return models, criterion, model_dim


	def generate_agents(self, graph, bandwidths, models, criterion, trainsets,
		testset, batch_size, learning_rate_type, r):
		agents = []
		for i in range(self.num_agents):
			agent_i = Agent(
				ID								=	i,
				bandwidth						=	bandwidths[i],
				model							=	models[i],
				criterion						=	criterion,
				trainset						=	trainsets[i],
				testset							=	testset,
				model_dim						=	self.model_dim,
				batch_size						=	batch_size,
				learning_rate_type				=	learning_rate_type,
				r								=	r,
				threshold_type					=	None,
				randomized_gossip_probability	=	None
			)
			agents.append(agent_i)

		for i in range(self.num_agents):
			for j in list(graph.adj[i]):
				agents[i].add_neighbor(agents[j])

		return agents


	def device_data_random(self, k):
		# Assuming k >= 1
		# Variance ~ (1 - 1/k)^2

		medium_bandwidth = self.system_bandwidth / self.num_agents
		weak_bandwidth = medium_bandwidth // k
		powerful_bandwidth = medium_bandwidth * (2*k - 1) // k

		bandwidths = [randint(weak_bandwidth, powerful_bandwidth) for _ in range(self.num_agents)]
		return bandwidths


	def device_data_two_slice(self, weak_ratio):
		weak_count = int(weak_ratio * self.num_agents)
		powerful_count = self.num_agents - weak_count

		if weak_count != self.num_agents:
			weak_bandwidth = 1000
		else:
			weak_bandwidth = self.system_bandwidth // weak_count
		if powerful_count != 0:
			powerful_bandwidth = (self.system_bandwidth - weak_count*weak_bandwidth) // powerful_count
		else:
			powerful_bandwidth = 0

		bandwidths = [weak_bandwidth for _ in range(weak_count)]
		bandwidths.extend([powerful_bandwidth for _ in range(powerful_count)])
		return bandwidths


	def separate_data_iid(self, shuffled):
		# return self.separate_data(shuffled, self.num_classes)

		div = len(shuffled) // self.num_agents
		separated = [shuffled[i*div : (i+1)*div] for i in range(self.num_agents-1)]
		separated.append(shuffled[(self.num_agents-1)*div : len(shuffled)])
		return separated


	def separate_data_non_iid(self, shuffled):
		# return self.separate_data(shuffled, 1)

		separated_by_output = [[data for data in shuffled if data[1] == j] for j in range(self.num_classes)]

		each_class_div = [self.num_agents // self.num_classes for _ in range(self.num_classes-1)]
		each_class_div.append(self.num_agents - self.num_agents//self.num_classes*(self.num_classes-1))
		separated = []
		for j in range(self.num_classes):
			div = len(separated_by_output[j]) // (each_class_div[j])
			separated.extend([separated_by_output[j][i*div : (i+1)*div] for i in range(each_class_div[j]-1)])
			separated.append(separated_by_output[j][(each_class_div[j]-1)*div : len(separated_by_output[j])])

		separated_shuffled = sample(separated, k=len(separated))
		return separated_shuffled


	def separate_data(self, shuffled, labels_per_agent):
		separated_by_output = {j : [data for data in shuffled if data[1] == j] for j in range(self.num_classes)}

		total_data_splits_count = self.num_agents * labels_per_agent
		data_splits_per_class = total_data_splits_count // self.num_classes
		each_class_div = [data_splits_per_class for _ in range(self.num_classes-1)]
		each_class_div.append(total_data_splits_count - data_splits_per_class*(self.num_classes-1))
		available_splits = {j : each_class_div[j] for j in range(self.num_classes)}

		data_splits = {j : [] for j in range(self.num_classes)}
		for j in range(self.num_classes):
			div = len(separated_by_output[j]) // (each_class_div[j])
			data_splits[j].extend([separated_by_output[j][i*div : (i+1)*div] for i in range(each_class_div[j]-1)])
			data_splits[j].append(separated_by_output[j][(each_class_div[j]-1)*div : len(separated_by_output[j])])

		separated = [[] for _ in range(self.num_agents)]
		for i in range(self.num_agents):
			available_splits_temp = deepcopy(available_splits)
			chosen_splits = []
			for j in range(labels_per_agent):
				chosen_splits.extend(choices(list(available_splits_temp.keys()),
					weights=list(available_splits_temp.values()), k=1))
				del available_splits_temp[chosen_splits[-1]]

			for j in chosen_splits:
				separated[i].extend(data_splits[j][0])
				available_splits[j] -= 1

		separated_shuffled = [sample(separated[i], k=len(separated[i])) for i in range(len(separated))]
		return separated_shuffled


	def run(self):
		# num_iters = 2		# comment this line (this was used for testing)
		num_iters = len(self.trainset) // self.num_agents
		threshold_types = ["heterogeneous", "constant", "randomized_gossip", "zero"]
		results = {}

		for threshold_type in threshold_types:
			for i in range(self.num_agents):
				model, _, _ = utils.model_info(self.model_name, self.input_dim, self.num_classes)

				if threshold_type == "randomized_gossip":
					self.agents[i].reset(model, threshold_type=threshold_type,
						randomized_gossip_probability=1/self.num_agents)
				else:
					self.agents[i].reset(model, threshold_type=threshold_type)

			log = self.simulate(num_iters)
			results[threshold_type] = log

		return results


	def simulate(self, num_iters):
		iters, iters_sampled = [], []
		losses, accuracies = [], []
		cpu_utilizations = []
		bandwidth_useds, bandwidth_useds_cumsum, bandwidth_utilizations = [], [], []
		transmission_time_useds, transmission_time_useds_cumsum, transmission_time_utilizations = [], [], []

		threads = [None for _ in range(self.num_agents)]

		total_iter = 0
		tx_iter = 0
		tx_sample = self.model_dim/(self.system_bandwidth//self.num_agents)

		for k in range(self.num_epochs):
			print(f"epoch: {k}")
			
			for i in range(num_iters):
				total_iter = k*num_iters + i
				# print(f"epoch: {k}, iter: {i}, total_iter={total_iter}")

				loss = 0
				cpu_used, max_cpu_usable = 0, 0
				bandwidth_used, max_bandwidth_usable = 0, 0
				transmission_time_used, max_transmission_time_usable = 0, 0

				for j in range(self.num_agents):
					threads[j] = Thread(target=self.agents[j].run())
					threads[j].start()

				for j in range(self.num_agents):
					threads[j].join()

					loss += float(self.agents[j].get_loss())
					cpu_used += self.agents[j].cpu_used()
					max_cpu_usable += self.agents[j].max_cpu_usable()
					bandwidth_used += self.agents[j].bandwidth_used()
					max_bandwidth_usable += self.agents[j].max_bandwidth_usable()
					transmission_time_used += self.agents[j].transmission_time_used()
					max_transmission_time_usable += self.agents[j].max_transmission_time_usable()

				for j in range(self.num_agents):
					threads[j] = Thread(target=self.agents[j].post_run_update())
					threads[j].start()

				iters.append(total_iter)
				losses.append(loss / self.num_agents)
				cpu_utilizations.append(cpu_used / max_cpu_usable)
				bandwidth_utilizations.append(bandwidth_used / max_bandwidth_usable)
				transmission_time_utilizations.append(transmission_time_used / max_transmission_time_usable)
				bandwidth_useds.append(bandwidth_used / self.num_agents)
				transmission_time_useds.append(transmission_time_used / self.num_agents)
				if total_iter == 0:
					bandwidth_useds_cumsum.append(bandwidth_useds[-1])
					transmission_time_useds_cumsum.append(transmission_time_useds[-1])
				else:
					bandwidth_useds_cumsum.append(bandwidth_useds_cumsum[-1] + bandwidth_useds[-1])
					transmission_time_useds_cumsum.append(transmission_time_useds_cumsum[-1] + transmission_time_useds[-1])

				for j in range(self.num_agents):
					threads[j].join()

				######################## (a) For the Line plots ########################
				# cond1 = total_iter % 10**(1 + (len(str(total_iter)) - 1)//3) == 0
				# cond2 = transmission_time_useds_cumsum[-1] > tx_sample*tx_iter
				# if cond1 or cond2:
				# 	accuracy = 0
					
				# 	for j in range(self.num_agents):
				# 		threads[j] = Thread(target=self.agents[j].calculate_accuracy())
				# 		threads[j].start()
				# 	for j in range(self.num_agents):
				# 		threads[j].join()
				# 		accuracy += self.agents[j].get_accuracy()

				# 	accuracies.append(accuracy / self.num_agents)
				# 	iters_sampled.append(total_iter)

				# if cond2:
				# 	tx_iter += 10**(1 + (len(str(tx_iter)) - 1)//3)
				########################################################################

			########################### (b) For the Bar plots ##########################
				cond3 = transmission_time_useds_cumsum[-1] > 1e2*2*tx_sample
				if cond3:
					accuracy = 0
					
					for j in range(self.num_agents):
						threads[j] = Thread(target=self.agents[j].calculate_accuracy())
						threads[j].start()
					for j in range(self.num_agents):
						threads[j].join()
						accuracy += self.agents[j].get_accuracy()

					accuracies.append(accuracy / self.num_agents)
					iters_sampled.append(total_iter)
					break
			if cond3:
				break
			#############################################################################

		log1 = {"iters"								:	iters,
				"losses"							:	losses,
				"cpu_utilizations"					:	cpu_utilizations,
				"bandwidth_useds"					:	bandwidth_useds,
				"bandwidth_useds_cumsum"			:	bandwidth_useds_cumsum,
				"bandwidth_utilizations"			:	bandwidth_utilizations,
				"transmission_time_useds"			:	transmission_time_useds,
				"transmission_time_useds_cumsum"	:	transmission_time_useds_cumsum,
				"transmission_time_utilizations"	:	transmission_time_utilizations}

		log2 = {"iters_sampled"						:	iters_sampled,
				"accuracies"						:	accuracies}
		return [log1, log2]


	def reset(self, system_bandwidth_parameter=0.8,
			graph_connectivity=0.4,
			data_distribution="iid",
			labels_per_agent=None):
		if system_bandwidth_parameter != self.system_bandwidth_parameter:
			self.reset_bandwidths(system_bandwidth_parameter)
		if graph_connectivity != self.graph_connectivity:
			self.reset_graph(graph_connectivity)
		if data_distribution != self.data_distribution or labels_per_agent != self.labels_per_agent:
			self.reset_trainsets(data_distribution, labels_per_agent)


	def reset_bandwidths(self, system_bandwidth_parameter):
		bandwidths = self.generate_bandwidths(system_bandwidth_parameter)
		for i in range(self.num_agents):
			self.agents[i].set_bandwidth(bandwidths[i])


	def reset_graph(self, graph_connectivity):
		graph = self.generate_graph(graph_connectivity)
		for i in range(self.num_agents):
			self.agents[i].clear_neighbors()
			for j in list(graph.adj[i]):
				self.agents[i].add_neighbor(self.agents[j])


	def reset_trainsets(self, data_distribution, labels_per_agent):
		trainsets = self.generate_trainsets(data_distribution, labels_per_agent=labels_per_agent)
		for i in range(self.num_agents):
			self.agents[i].set_trainset(trainsets[i])


if __name__ == '__main__':
	simulation = EF_HC(
		model_name					=	"SVM",
		dataset_name				=	"MNIST",
		num_epochs					=	1,
		num_agents					=	10,
		graph_connectivity			=	0.2,
		system_bandwidth			=	10*5000,
		system_bandwidth_type		=	"two_slice",
		system_bandwidth_parameter	=	0.8,
		data_distribution			=	"iid",
		labels_per_agent			=	None,
		batch_size					=	1,
		learning_rate_type			=	"iter_decay",
		r							=	5000*1e-1
	)
	results = simulation.run()

	for threshold_type, log in results.items():
		log1, log2 = log
		print(threshold_type)
		print(log2["accuracies"])