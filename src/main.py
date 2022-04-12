import os
import pandas as pd

from EF_HC import EF_HC


class Simulations():
	"""docstring for Main"""
	def __init__(self):
		self.avg_bandwidth = 5000
		is_barplots = True

		# Simulation parameters
		self.model_name = "SVM"
		self.dataset_name = "FMNIST"
		self.num_epochs = 2
		self.num_agents = 10
		self.graph_connectivities = [2*i/10 for i in range(1,6)]
		self.system_bandwidth_parameters = [i/10 for i in range(10)]
		# self.data_distributions = ["iid", "non_iid", "labels_per_agent"]
		self.data_distributions = ["non_iid", "labels_per_agent"]
		self.labels_per_agents = [i for i in range(1, 11)]
		self.r = self.avg_bandwidth*1e-2

		lpa = 1 if "labels_per_agent" in self.data_distributions else 0
		self.simulation_count = len(self.labels_per_agents)*lpa
		self.simulation_count += (len(self.graph_connectivities)+len(self.system_bandwidth_parameters))*(len(self.data_distributions) - lpa)

		self.simulation_counter = 1

		self.ef_hc = EF_HC(
			model_name					=	self.model_name,
			dataset_name				=	self.dataset_name,
			num_epochs					=	self.num_epochs,
			num_agents					=	self.num_agents,
			graph_connectivity			=	0.4,
			system_bandwidth			=	self.avg_bandwidth*self.num_agents,
			system_bandwidth_type		=	"two_slice",
			system_bandwidth_parameter	=	0.8,
			data_distribution			=	"non_iid",
			labels_per_agent			=	None,
			batch_size					=	1,
			learning_rate_type			=	"iter_decay",
			r							=	self.r,
			is_barplots					=	is_barplots
		)


	def save_results(self, results, filepath1, filepath2):
		with pd.ExcelWriter(filepath1) as writer:
			for threshold_type, log in results.items():
				log1, _ = log
				df1 = pd.DataFrame(log1)
				df1.to_excel(writer, sheet_name=threshold_type)

		with pd.ExcelWriter(filepath2) as writer:
			for threshold_type, log in results.items():
				_, log2 = log
				df2 = pd.DataFrame(log2)
				df2.to_excel(writer, sheet_name=threshold_type)


	def run(self):
		system_bandwidth = self.num_agents*self.avg_bandwidth

		for data_distribution in self.data_distributions:
			if data_distribution == "labels_per_agent":
				for labels_per_agent in self.labels_per_agents:
					self.simulate_and_save_results(0.8, 0.4, "labels_per_agent", labels_per_agent, "labels")

			else:
				for graph_connectivity in self.graph_connectivities:
					self.simulate_and_save_results(0.8, graph_connectivity, data_distribution, None, "conns")

				for system_bandwidth_parameter in self.system_bandwidth_parameters:
					self.simulate_and_save_results(system_bandwidth_parameter, 0.4, data_distribution, None, "bws")


	def file_info(self, system_bandwidth_parameter, graph_connectivity, data_distribution, labels_per_agent, name):
		dirname = "../results"
		dirpath = os.path.join(os.getcwd(), dirname)
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)

		filename = f"{self.model_name}_{self.dataset_name}_{self.num_epochs}epochs_{self.num_agents}m"
		filename += f"_{graph_connectivity}conn_{system_bandwidth_parameter}weak_{data_distribution}"
		filename += f"_{labels_per_agent}labels_{self.r}r_{name}"

		filename1 = filename + "_iter.xlsx"
		filename2 = filename + "_iter_sampled.xlsx"
		filepath1 = os.path.join(dirpath, filename1)
		filepath2 = os.path.join(dirpath, filename2)

		return filename, filepath1, filepath2


	def simulate_and_save_results(self, system_bandwidth_parameter, graph_connectivity,
		data_distribution, labels_per_agent, name):
		filename, filepath1, filepath2 = self.file_info(system_bandwidth_parameter, graph_connectivity,
			data_distribution, labels_per_agent, name)

		print(f"Simulation {self.simulation_counter}/{self.simulation_count}: {filename}")
		self.simulation_counter += 1
		if os.path.exists(filepath1) and os.path.exists(filepath2):
			return

		self.ef_hc.reset(
			system_bandwidth_parameter=system_bandwidth_parameter,
			graph_connectivity=graph_connectivity,
			data_distribution=data_distribution,
			labels_per_agent=labels_per_agent
		)
		results = self.ef_hc.run()

		self.save_results(results, filepath1, filepath2)


def main():
	sims = Simulations()
	sims.run()


if __name__ == '__main__':
	main()