import hdbscan
from koopman import *

class KoopmanCluster:
	"""
	Given a dataset partitioned into x, u and xn arrasys,
	KoopmanCluster further partitions those further according
	to a desired receding time horizon
	"""
	def __init__(self, Basis, window_sz, overlap, min_cluster_size):
		self.basis = Basis
		self.window_sz = window_sz
		self.overlap = overlap
		self.inds = []
		self.koop_list = []
		self.koopman_feature_array = []
		self.labels = []
		self.koop_cluster_list = []
		self.koop_cluster_memb_prob_list = []
		self.koopman_hybrid_modes = []
		self.min_cluster_size = min_cluster_size
		self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,metric='euclidean')

	def calculate(self,x_data,xn_data,u_data=None):
		"""
		This calculates the Koopman Feature array which is an array containing all
		flattened koopman operators to be clustered on.
		"""
		self.inds = window_inds(x_data,self.window_sz, self.overlap)
		xArr_List = [x_data[:, self.inds[i][0]:self.inds[i][1]] for i in range(len(self.inds))]
		xnArr_List = [xn_data[:, self.inds[i][0]:self.inds[i][1]] for i in range(len(self.inds))]
		self.koop_list = [KoopmanOperator(self.basis) for i in range(len(self.inds))]
		self.koopman_feature_array = np.zeros((len(self.koop_list),self.basis.nK**2))

		# This is important for when there's no control
		if u_data is not None:
			uArr_List = [u_data[:, self.inds[i][0]:self.inds[i][1]] for i in range(len(self.inds))]
			for i in range(len(self.koop_list)):
				self.koop_list[i].update(xArr_List[i],xnArr_List[i],uArr_List[i])
				self.koopman_feature_array[i,:] = self.koop_list[i].K.reshape(self.basis.nK**2)
		else:
			for i in range(len(self.koop_list)):
				self.koop_list[i].update(xArr_List[i],xnArr_List[i])
				self.koopman_feature_array[i,:] = self.koop_list[i].K.reshape(self.basis.nK**2)

	def cluster(self):
		"""
		This applies the clustering to the Koopman Feature array and creates a list of koopman operators
		and the cluster they belong to, along with a list of the probabilities of belonging to their
		assigned cluster. Does not include noise as a group in the list.
		"""
		self.clusterer.fit(self.koopman_feature_array)
		self.labels = self.clusterer.labels_
		for j in range(max(self.labels)+1):
			self.koop_cluster_list.append([self.koop_list[i] for i in range(len(self.labels)) if self.labels[i] == j])
			self.koop_cluster_memb_prob_list.append([self.clusterer.probabilities_[i] for i in range(len(self.labels)) if self.labels[i] == j])

	def reconcile(self):
		"""
		This goes ahead and calculates a weighted average of Koopman operators in each cluster
		according to the membership probabilities to generate an examplar Koopman operator
		for each cluster. Does not include noise as a group in the list because the
		probability is 0 for each noise point.
		"""
		class_A = np.empty((self.basis.nK,self.basis.nK),float)
		class_G = np.empty((self.basis.nK,self.basis.nK),float)
		class_koop = np.empty((self.basis.nK,self.basis.nK),float)
		class_num = len(self.koop_cluster_list)
		for i in range(class_num):
			class_mem_num = len(self.koop_cluster_list[i])
			class_counter = 0.0
			for j in range(class_mem_num):
				class_A += (self.koop_cluster_list[i][j]._A*self.koop_cluster_memb_prob_list[i][j])/float(class_mem_num)
				class_G += (self.koop_cluster_list[i][j]._G*self.koop_cluster_memb_prob_list[i][j])/float(class_mem_num)
				class_counter += self.koop_cluster_list[i][j].counter
			class_koop = np.dot(np.linalg.pinv(class_G),class_A)
			self.koopman_hybrid_modes.append(KoopmanOperator(self.basis, class_koop, class_A, class_G, class_counter))

def window_inds(dataset, window_sz, overlap):
	"""
	Helper function that applies a rectangular window to the dataset
	given some overlap percentage, s.t. ov \in [0,1)
	"""
	data_len = len(dataset[0])
	assert window_sz < data_len
	ind1 = 0
	ind2 = window_sz-1
	ind_list = []
	ov_ind_diff = int(np.ceil(np.abs(overlap*window_sz)))
	if ov_ind_diff == window_sz:
		ov_ind_diff += -1
	while ind2 < data_len:
		ind_list.append((ind1,ind2))
		ind1 += window_sz-ov_ind_diff
		ind2 += window_sz-ov_ind_diff
	return ind_list
