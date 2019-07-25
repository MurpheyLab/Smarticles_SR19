from koopman_cluster import *
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import networkx as nx
import pickle

class DSS:
	def __init__(self, Basis, SVM_Basis=None, *args):
		"""
		Initialization uses:

		1 arg: loading model from file
		2 arg: loading model from file, bool if loading raw data
		5 arg: window_size, overlap %, min_cluster_size, x, xn, for no control case
		6 arg: window_size, overlap %, min_cluster_size, x, xn, u, for control
		7 arg: window_size, overlap %, min_cluster_size, x, xn, use_lpf, n, for no control case
		8 arg: window_size, overlap %, min_cluster_size, x, xn, u, use_lpf, n, for control
		"""
		# Clustering related
		self.basis = Basis
		if SVM_Basis is None:
			self.svm_basis = Basis
		else:
			self.svm_basis = SVM_Basis
		self.window_size = 0
		self.overlap = 0
		self.min_cluster_size = 0
		self.cluster = []
		self.gen_prob = False # makes the algorithm run slower, but gives prediction probabilities
		self.classifier = SVC(kernel='linear', probability=self.gen_prob)

		# Graph related
		self.nodes = []
		self.edges = []
		self.weights = []
		self.state_distribution = []
		self.transition_counts = []
		self.transition_mat = np.empty((0,1),int)
		self.graph = nx.MultiDiGraph()

		# Raw data related
		self.SNR = 1.0
		self.filtered_states = []
		self.filtered_controls = []
		self.filtered_state_labels = []
		self.states = []
		self.next_states = []
		self.controls = None
		self.state_labels = []
		if self.gen_prob:
			self.state_probabilities = []

		# if loading from file
		if len(args) == 1:
			self.load(args[0])
		# if loading from file and choosing to load raw data or not
		elif len(args) == 2:
			self.load(args[0],args[1])
		# if directly initializing with no control
		elif len(args) == 5:
			self.window_size = args[0]
			self.overlap = args[1]
			self.min_cluster_size = args[2]
			self.cluster = KoopmanCluster(self.basis, self.window_size, self.overlap, self.min_cluster_size)
			self.train(args[3],args[4])
		# if directly initializing with control
		elif len(args) == 6:
			self.window_size = args[0]
			self.overlap = args[1]
			self.min_cluster_size = args[2]
			self.cluster = KoopmanCluster(self.basis, self.window_size, self.overlap, self.min_cluster_size)
			self.train(args[3],args[4],args[5])
		# if using low pass filter on state labels
		elif len(args) == 7:
			self.window_size = args[0]
			self.overlap = args[1]
			self.min_cluster_size = args[2]
			self.cluster = KoopmanCluster(self.basis, self.window_size, self.overlap, self.min_cluster_size)
			self.train(args[3],args[4],args[5],args[6])
		# if using low pass filter on state labels
		elif len(args) == 8:
			self.window_size = args[0]
			self.overlap = args[1]
			self.min_cluster_size = args[2]
			self.cluster = KoopmanCluster(self.basis, self.window_size, self.overlap, self.min_cluster_size)
			self.train(args[3],args[4],args[5],args[6],args[7])
		else:
			raise ValueError('Wrong number of arguments given')

	def __del__(self):
		class_name = self.__class__.__name__
		print(class_name, "destroyed")

	def train(self,x_data,xn_data,u_data=None, use_lpf=False, window=10):

		"""
		This function applies all other class methods to fully generate
		the DSS from data, as well as the graphical model.
		"""
		self.states = x_data
		self.next_states = xn_data
		self.controls = u_data
		print "Discerning system states..."
		self.generate_nodes(x_data,xn_data,u_data)
		print str(len(self.nodes))+" states identified"
		print "Training state classifier"
		self.generate_indicator(x_data, u_data)
		print "Populating state transition matrix"
		self.generate_edges(use_lpf, window)

	def generate_nodes(self,x_data,xn_data,u_data=None):
		"""
		This function generates the DSS nodes by using the Koopman Cluster
		class to generate the exemplar Koopman operators from data
		"""
		self.cluster.calculate(x_data,xn_data,u_data)
		self.cluster.cluster()
		self.cluster.reconcile()
		self.nodes = self.cluster.koopman_hybrid_modes

	def filter_noise(self,x_data,u_data=None):
		"""
		This function removes data associated with the noise cluster from the
		states used to generate the model. Important note: the state_labels
		are not over the whole dataset, just what the koopman operators
		went over. The rest of the dataset is added afterwards
		"""
		# first loop creates a mask
		n_times = self.window_size-int(np.ceil(np.abs(self.overlap*self.window_size)))
		mask = []
		for label in self.cluster.labels:
			self.state_labels.extend([label]*n_times)
			if label != -1:
				self.filtered_state_labels.extend([label]*n_times)
				mask.extend([False]*n_times)
			else:
				mask.extend([True]*n_times)

		labelLen = len(self.filtered_state_labels)
		if labelLen > 0:
			self.SNR = float(len(mask))/labelLen
		else:
			raise ValueError("No clusters identified")

		# second loop applies mask
		self.filtered_states = np.empty((self.basis.xSz,labelLen),float)
		self.filtered_controls = np.empty((self.basis.uSz,labelLen),float)
		j = 0
		for i in range(len(mask)):
			if not mask[i]:
				self.filtered_states[:,j] = x_data[:,i]
				if self.basis.uSz != 0:
					self.filtered_controls[:,j] = u_data[:,i]
				j+=1

	def generate_indicator(self,x_data,u_data=None):
		"""
		Generates an indicator function given labeled data and choice
		of basis functions
		"""
		self.filter_noise(x_data,u_data)
		dataLen = len(self.filtered_states[0])
		phi_data = np.empty((self.svm_basis.nK,dataLen),float)
		for i in range(dataLen):
			phi_data[:,i] = self.svm_basis.fk(self.filtered_states[:,i],self.filtered_controls[:,i])
		self.classifier.fit(phi_data.T.tolist(), self.filtered_state_labels)

	def indicator(self,phi):
		"""
		This function applies the indicator that was trained in the basis
		function space
		"""
		return self.classifier.predict(phi.reshape(1,self.svm_basis.nK).tolist())

	def f(self,x, u=None):
		"""
		This performs an evolution of a point but returns a phi_{k+1}
		rather than an x_{k+1}, so the state needs to be pulled out
		"""
		phik = self.svm_basis.fk(x,u)
		class_id = self.indicator(phik)
		return self.nodes[class_id[0]].f(x,u)

	def f_stochastic(self, x, u=None):
		if self.gen_prob:
			phik = self.svm_basis.fk(x,u)
			dist = self.classifier.predict_proba(phik.reshape(1,self.svm_basis.nK).tolist())
			node = np.random.choice(self.nodes, p=dist)
		else:
			raise ValueError('Must turn on self.gen_prob flag and generate SVM membership probabilities for f_stochastic to work.')
		return node.f(x,u)
		
	def dfdx(self,x, u=None):
		phik = self.svm_basis.fk(x,u)
		class_id = self.indicator(phik)
		return self.nodes[class_id[0]].dfdx(x,u)

	def dfdu(self,x, u=None):
		phik = self.svm_basis.fk(x,u)
		class_id = self.indicator(phik)
		return self.nodes[class_id[0]].dfdu(x,u)

	def update(self,x,xn,u=None):
		"""
		This performs an update on the DSS given a single point.
		"""
		# Getting new values
		class_id = self.indicator(self.svm_basis.fk(x,u))
		class_id_new = self.indicator(self.svm_basis.fk(xn,u))
		new_edge = (class_id[0], class_id_new[0])

		# Update node
		if u is None:
			self.nodes[class_id[0]].update(x.reshape((self.basis.xSz,1)),\
										   xn.reshape((self.basis.xSz,1)))
		else:
			self.nodes[class_id[0]].update(x.reshape((self.basis.xSz,1)),\
										   xn.reshape((self.basis.xSz,1)),\
										   u.reshape((self.basis.uSz,1)))
		# Updating state distribution
		edgesLen = len(self.edges)
		nodesLen = len(self.nodes)
		total_state_count = float(sum([self.nodes[i].counter for i in range(nodesLen)]))
		self.state_distribution = [self.nodes[j].counter/total_state_count for j in range(nodesLen)]

		# Updating transition counts, and adding new edge if previously unobserved
		if new_edge not in self.edges:
			self.edges.append(new_edge)
			self.transition_counts.append(1.0)
			self.weights.append(0.0)
			self.graph.add_edge(new_edge[0],new_edge[1])
		else:
			for i in range(edgesLen):
				if self.edges[i] == new_edge:
					self.transition_counts[i] += 1.0

		# Updating edge weights and transition matrix
		inds = [ind for ind in range(edgesLen) if self.edges[ind][0] == new_edge[0]]
		nodeTotal = float(sum([self.transition_counts[i] for i in inds]))
		for ind in inds:
			self.weights[ind] = self.transition_counts[ind]/nodeTotal
			self.transition_mat[self.edges[ind][0], self.edges[ind][1]] = self.weights[ind]

		return self.basis.fk(xn,u)

	def reclassify(self,use_lpf=False, window=10):
		"""
		This method takes all the points classified as noise originally and reclassifies
		while updating the DSS nodes. It also reclassifies the data at the edge of
		the final windows that does not get labeled

		NOTE: Batch functionality needs to be added
		"""
		dataLen = len(self.states[0])
		train_dataLen = len(self.state_labels)
		phi_data = np.empty((self.svm_basis.nK,dataLen),float)
		self.state_labels.extend(range(dataLen-train_dataLen))
		for i in range(dataLen):
			x = self.states[:,i].reshape((self.basis.xSz,1))
			xn = self.next_states[:,i].reshape((self.basis.xSz,1))
			u = None
			if i < train_dataLen:
				if self.state_labels[i] == -1:
					if self.controls is not None:
						u = self.controls[:,i].reshape((self.basis.uSz,1))
					phi_data[:,i] = self.svm_basis.fk(x,u)
					class_id = self.indicator(phi_data[:,i])
					self.nodes[class_id[0]].update(x, xn, u)
					self.state_labels[i] = class_id[0]
			else:
				if self.controls is not None:
					u = self.controls[:,i].reshape((self.basis.uSz,1))
				phi_data[:,i] = self.svm_basis.fk(x,u)
				class_id = self.indicator(phi_data[:,i])
				self.nodes[class_id[0]].update(x, xn, u)
				self.state_labels[i] = class_id[0]

		if use_lpf:
			self.state_labels = moving_average(self.state_labels, window)
		if self.gen_prob:
			self.state_probabilities = self.classifier.predict_proba(phi_data.T.tolist())

	def generate_edges(self,use_lpf=False, window=10):
		"""
		This performs a one step analysis to look at the possible
		transitions between states, generates edges in the graph
		and builds the graph. We use maximum likelihood estimation
		for the weights and transition matrix. The observed state
		distribution is strictly an empirical accounting.
		"""

		# Making unique edge list
		self.reclassify(use_lpf, window)
		full_edges = zip(self.state_labels[:-1], self.state_labels[1:])
		self.edges = sorted(list(set(full_edges)))

		# Keeping track of transition probabilities/edge weights
		edgesLen = len(self.edges)
		class_num = len(self.nodes)
		self.weights = range(edgesLen)
		self.transition_mat = np.zeros((class_num, class_num), float)
		self.state_distribution = [self.nodes[j].counter/float(sum([self.nodes[i].counter for i in range(len(self.nodes))])) for j in range(len(self.nodes))]
		for i in range(edgesLen):
			self.transition_counts.append(float(full_edges.count(self.edges[i])))

		for i in range(class_num):
			inds = [ind for ind in range(edgesLen) if self.edges[ind][0] == i]
			nodeTotal = float(sum([self.transition_counts[i] for i in inds]))
			for ind in inds:
				self.weights[ind] = self.transition_counts[ind]/nodeTotal
				self.transition_mat[self.edges[ind][0], self.edges[ind][1]] = self.weights[ind]

		# Putting the whole graph together
		self.graph.add_nodes_from(range(class_num))
		self.graph.add_edges_from([(self.edges[i][0], self.edges[i][1]) for i in range(edgesLen)])

	def draw_graph(self):
		"""
		This plots the FSM generated
		"""
		class_num = len(self.nodes)
		palette = sns.color_palette("Set2",class_num)
		g = plt.figure()
		pos = nx.spring_layout(self.graph)
		nx.draw_networkx(self.graph, pos, font_color='k', node_size=800, edge_color='k', node_color=[palette[node] for node in self.graph.nodes()], alpha=1.0)
		plt.axis('off')

	def state_plot(self,x_var,y_var,*args):
		"""
		This plots the separate modes given two state variables
		"""
		if self.controls is not None:
			full_state_control = np.vstack([self.states, self.controls])
		else:
			full_state_control = self.states

		sp = plt.figure()
		plot_kwds = {'alpha' : 0.3, 's' : 20, 'linewidths':0}
		class_num = len(self.nodes)
		palette = sns.color_palette("Set2", class_num)
		for i in range(class_num):
			mask = (self.state_labels == np.ones(np.shape(self.state_labels))*i)
			plt.subplot(class_num, 1,class_num-i)
			plt.scatter(full_state_control[x_var,mask], full_state_control[y_var,mask], c=palette[i], **plot_kwds)
			if len(args) == 2:
				plt.ylabel(args[1])
				if i == 0:
					plt.xlabel(args[0])
			plt.grid(True)
			plt.xlim([min(full_state_control[x_var]), max(full_state_control[x_var])])
			plt.ylim([min(full_state_control[y_var]), max(full_state_control[y_var])])

	def time_plot(self,t=np.empty((0,1),float)):
		"""
		This is plotting the time histories without the noise terms
		"""
		if self.controls is not None:
			full_state_control = np.vstack([self.states, self.controls])
		else:
			full_state_control = self.states

		tp = plt.figure()
		plot_kwds = {'alpha' : 0.3, 's' : 20, 'linewidths':0}
		class_num = len(self.nodes)
		palette = sns.color_palette("Set2",class_num)
		cmap = ListedColormap(palette)
		norm = BoundaryNorm(range(class_num+1), cmap.N)

		if len(t) == 0:
			t = np.linspace(0,len(full_state_control[0]),len(full_state_control[0]))
		for i in range(self.basis.xSz+self.basis.uSz):
			plt.subplot(self.basis.xSz+self.basis.uSz,1,i+1)
			for j in range(class_num):
				mask = (self.state_labels == np.ones(np.shape(self.state_labels))*j)
				m = cm.ScalarMappable(cmap=cmap, norm=norm)
				plt.scatter(t[mask], full_state_control[i,mask], c=palette[j], **plot_kwds)
			if i == (self.basis.xSz+self.basis.uSz)-1:
				plt.xlabel('Time')
			m.set_array(self.state_labels)
			plt.grid(True)
			plt.xlim([min(t), max(t)])
			plt.ylim([min(full_state_control[i]), max(full_state_control[i])])
			cbar = plt.colorbar(m,ticks=[i+0.5 for i in range(class_num)])
			cbar.ax.set_yticklabels(range(class_num))

	def save(self, filename='dss.pkl', save_raw_data=True):
		if save_raw_data:
			D = {"window_size": self.window_size,
					"overlap": self.overlap,
					"min_cluster_size": self.min_cluster_size,
					"nodes": self.nodes,
					"edges": self.edges,
					"graph": self.graph,
					"weights": self.weights,
					"SNR": self.SNR,
					"transition_counts": self.transition_counts,
					"state_distribution": self.state_distribution,
					"transition_mat": self.transition_mat,
					"filtered_states": self.filtered_states,
					"filtered_controls": self.filtered_controls,
					"filtered_state_labels": self.filtered_state_labels,
					"states": self.states,
					"next_states": self.next_states,
					"controls": self.controls,
					"state_labels": self.state_labels,
					"classifier": self.classifier}
		else:
			D = {"window_size": self.window_size,
					"overlap": self.overlap,
					"min_cluster_size": self.min_cluster_size,
					"nodes": self.nodes,
					"edges": self.edges,
					"graph": self.graph,
					"weights": self.weights,
					"state_distribution": self.state_distribution,
					"transition_mat": self.transition_mat,
					"transition_counts": self.transition_counts,
					"classifier": self.classifier}
		file = open(filename, 'wb')
		pickle.dump(D,file)
		file.close()

	def load(self,filename='dss.pkl',load_raw_data=True):
		file = open(filename,'rb')
		D = pickle.load(file)
		file.close()
		if load_raw_data:
			self.window_size = D['window_size']
			self.overlap = D['overlap']
			self.min_cluster_size = D['min_cluster_size']
			self.nodes = D['nodes']
			self.edges = D['edges']
			self.graph = D['graph']
			self.weights = D['weights']
			self.SNR = D['SNR']
			self.state_distribution = D['state_distribution']
			self.transition_counts = D['transition_counts']
			self.transition_mat = D['transition_mat']
			self.filtered_states = D['filtered_states']
			self.filtered_controls = D['filtered_controls']
			self.filtered_state_labels = D['filtered_state_labels']
			self.states = D['states']
			self.next_states = D['next_states']
			self.controls = D['controls']
			self.state_labels = D['state_labels']
			self.classifier = D['classifier']
		else:
			self.window_size = D['window_size']
			self.overlap = D['overlap']
			self.min_cluster_size = D['min_cluster_size']
			self.nodes = D['nodes']
			self.edges = D['edges']
			self.graph = D['graph']
			self.weights = D['weights']
			self.state_distribution = D['state_distribution']
			self.transition_mat = D['transition_mat']
			self.transition_counts = D['transition_counts']
			self.classifier = D['classifier']

	def show(self):
		plt.show()

def moving_average(a, n=10):
	mat = np.asarray(a)
	matLen = len(mat)
	res = range(matLen)
	indlist = range(n)
	for i in range(matLen):
		if i<matLen-n:
			indlist = range(i,i+n)
		else:
			indlist = [i]*n
		res[i] = int(round(np.sum(mat[indlist])/n))
	return res
