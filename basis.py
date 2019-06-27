import numpy as np

class BasisFns:
	"""
	Contains the definition of a set of basis functions
	put into a column vector, for computation of a
	Koopman operator
	"""
	def __init__(self, xSz=20, uSz=0, nX=62, nU=0, dt = 1.0/120.0):
		self.xSz = xSz
		self.uSz = uSz
		self.nX = nX
		self.nU = nU
		self.nK = nX+nU
		self.dt = dt

	def fk(self,x,u):
		phix = np.asarray([ x[0], # x1
							x[1], # x2
							x[2], # x3
							x[3], # x4
							x[4], # x5
							x[5], # y1
							x[6], # y2
							x[7], # y3
							x[8], # y4
							x[9], # y5
							x[10], # th1
							x[11], # th2
							x[12], # th3
							x[13], # th4
							x[14], # th5
							x[15], # inactive s1
							x[16], # inactive s2
							x[17], # inactive s3
							x[18], # inactive s4
							x[19], # inactive s5
							x[15]+x[16]+x[17]+x[18]+x[19], # sum of active smarticles
							x[15] and x[16],
							x[15] or x[16],
							x[15] and x[17],
							x[15] or x[17],
							x[15] and x[18],
							x[15] or x[18],
							x[15] and x[19],
							x[15] or x[19],
							x[16] and x[17],
							x[16] or x[17],
							x[16] and x[18],
							x[16] or x[18],
							x[16] and x[19],
							x[16] or x[19],
							x[17] and x[18],
							x[17] or x[18],
							x[17] and x[19],
							x[17] or x[19],
							x[18] and x[19],
							x[18] or x[19],
							x[15] and x[16] and x[17],
							x[15] or x[16] or x[17],
							x[15] and x[16] and x[18],
							x[15] or x[16] or x[18],
							x[15] and x[16] and x[19],
							x[15] or x[16] or x[19],
							x[15] and x[17] and x[18],
							x[15] or x[17] or x[18],
							x[15] and x[17] and x[19],
							x[15] or x[17] or x[19],
							x[15] and x[18] and x[19],
							x[15] or x[18] or x[19],
							x[16] and x[17] and x[18],
							x[16] or x[17] or x[18],
							x[16] and x[17] and x[19],
							x[16] or x[17] or x[19],
							x[16] and x[18] and x[19],
							x[16] or x[18] or x[19],
							x[17] and x[18] and x[19],
							x[17] or x[18] or x[19],
							# x[0]**2.0,
							# x[1]**2.0,
							# x[2]**2.0,
							# x[3]**2.0,
							# x[4]**2.0,
							# x[0]**3.0,
							# x[1]**3.0,
							# x[2]**3.0,
							# x[3]**3.0,
							# x[4]**3.0,
							# x[5]**2.0,
							# x[6]**2.0,
							# x[7]**2.0,
							# x[8]**2.0,
							# x[9]**2.0,
							# x[5]**3.0,
							# x[6]**3.0,
							# x[7]**3.0,
							# x[8]**3.0,
							# x[9]**3.0,
							# x[10]**2.0,
							# x[11]**2.0,
							# x[12]**2.0,
							# x[13]**2.0,
							# x[14]**2.0,
							# x[10]**3.0,
							# x[11]**3.0,
							# x[12]**3.0,
							# x[13]**3.0,
							# x[14]**3.0,
							1
							 ])
		return phix


	def fkdx(self,x,u):
		dphixdx = np.asarray([0,0])
		return dphixdx.T

	def fkdu(self,x,u):
		dphiudu = np.asarray([0,0])
		return dphiudu.T
