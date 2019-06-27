import numpy as np

class KoopmanOperator:
	"""
	Computes Koopman operator either in batch or
	all the way through given a basis. Also keeps
	a copy of what states it was computed over for
	purposes of state-space classification.
	"""
	def __init__(self, Basis, *args):
		self.basis = Basis
		self.counter = 0.0
		self.nX_ = self.basis.xSz
		self.nU_ = self.basis.uSz

		# This distinction is made for whenever you want to
		# instantiate a Koopman from existing K, A and G matrices
		if len(args) == 3:
			self.K = args[0]
			self._A = args[1]
			self._G = args[2]
			self.Khat = self.K.T[:self.basis.xSz,:]
		elif len(args) == 4:
			self.K = args[0]
			self._A = args[1]
			self._G = args[2]
			self.counter = args[3]
			self.Khat = self.K.T[:self.basis.xSz,:]
		else:
			self.K = np.zeros((self.basis.nK, self.basis.nK))
			self._A = np.zeros((self.basis.nK, self.basis.nK))
			self._G = np.zeros((self.basis.nK, self.basis.nK))
			self.Khat = self.K.T[:self.basis.xSz,:]

	# These functions assume euler integration/discrete time systems
	def f(self,x,u=None):
		return x+np.dot(self.Khat,self.basis.fk(x,u))*self.basis.dt*(1.0/120.0)

	def dfdx(self,x,u=None):
		return np.eye(self.basis.xSz)+np.dot(self.Khat, self.basis.fkdx(x,u))*self.basis.dt

	def dfdu(self,x,u=None):
		return np.dot(self.Khat, self.basis.fkdu(x,u))*self.basis.dt

	def update(self,x,xn,u=None):
		"""
		Here we update the Koopman operator, either incrementally,
		or in a batch of data. We assume x,u,and xn have each sample
		as a column, extending over time in the row direction
		"""
		dataLen = len(x[0])
		self.counter += float(dataLen)
		for i in range(dataLen):
			if u is None:
				phix = self.basis.fk(x[:,i],None).reshape((self.basis.nK,1))
				phixpo = self.basis.fk(xn[:,i],None).reshape((self.basis.nK,1))
			else:
				phix = self.basis.fk(x[:,i], u[:,i]).reshape((self.basis.nK,1))
				phixpo = self.basis.fk(xn[:,i], u[:,i]).reshape((self.basis.nK,1))
			self._G += np.dot(phix,phix.T)/self.counter
			self._A += np.dot(phix,phixpo.T)/self.counter

		# the identity and sqrt(dt) terms are only applicable for euler integration
		self.K = (np.dot(np.linalg.pinv(self._G),self._A)-np.eye(self.basis.nK))*self.basis.dt
		# self.K = (np.dot(np.linalg.pinv(self._G),self._A))*self.basis.dt

		self.Khat = self.K.T[:self.basis.xSz,:]

	def save(self,filepath):
		"""
		Saves all the relevant information to a Koopman operator
		into a numpy npz file so that it can be loaded in without
		training from scratch
		"""
		np.savez(filepath, K=self.K, _A=self._A, _G=self._G, counter=self.counter)

	def load(self, filepath):
		"""
		Loads all relevant Koopman operator attributes.
		The numpy file keys are K,_A,_G,states
		"""
		file = np.load(filepath)
		self._A = file['_A']
		self._G = file['_G']
		self.K = file['K']
		self.counter = file['counter']
		self.Khat = self.K.T[:self.basis.xSz,:]
