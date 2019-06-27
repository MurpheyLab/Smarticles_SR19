from basis import *
from DSS import *
import itertools
import numpy as np
import matplotlib.pyplot as plt

##################
## Import Data ###
##################
filename = 'data/x.csv'
x_world = np.genfromtxt(filename,delimiter=',')
filename = 'data/y.csv'
y_world = np.genfromtxt(filename,delimiter=',')
filename = 'data/theta.csv'
theta_world = np.genfromtxt(filename,delimiter=',')
filename = 'data/ring.csv'
ring_world = np.genfromtxt(filename,delimiter=',')
data = np.hstack([x_world,y_world,theta_world])
sample = data[:-1,:].T

####################################
## Dynamical System Segmentation ###
####################################
basis = BasisFns()
D = DSS(basis,None,'dss_world.pkl')
freq = 120.0

####################################
############# Control ##############
####################################
u_sz = 5
r = 0.096
T = 120.0
time = np.linspace(0,T,int(freq*T))
u_list = np.array(list(itertools.product([0,1], repeat=u_sz)))
dirs = ['Left','Right','Up','Down']
trial_num = 15

# Objectives and Subobjectives
Jcm = lambda xm,ym,xd: (xm-xd[0])**2.0+(ym-xd[1])**2.0
J = lambda x,xd: Jind(x[0],x[5],xd)+Jind(x[1],x[6],xd)+Jind(x[2],x[7],xd)+Jind(x[3],x[8],xd)+Jind(x[4],x[9],xd)
Jind = lambda x,y,xd: (x-xd[0])**2.0+(y-xd[1])**2.0
Jcenter = lambda x,xm,ym,q: q*(Jind(x[0],x[5],[xm,ym])+Jind(x[1],x[6],[xm,ym])+
							 Jind(x[2],x[7],[xm,ym])+Jind(x[3],x[8],[xm,ym])+
							 Jind(x[4],x[9],[xm,ym]))

# Loop over directions: left, right, up, down
for d in range(4):
	print "----------------------------"
	print "Direction: "+dirs[d]
	print "----------------------------"
	plt.figure()
	plt.axhline(y=0, color='red')
	plt.axvline(x=0, color='red')
	plt.gca().set_aspect('equal')
	plt.title(dirs[d])

	# Loop over randomized initial conditions
	for t in range(trial_num):
		print "Iteration "+str(t)
		# Initial conditions and Goals (Left, Right, Up, Down)
		x_init = np.random.normal(sample[:,0],0.005)
		xd_list = [np.array([0,np.mean(x_init[5:10])]),np.array([0.45,np.mean(x_init[5:10])]),
				   np.array([np.mean(x_init[:5]),0.45]),np.array([np.mean(x_init[:5]),0])]
		xd = xd_list[d]
	
		# Initializations
		cost_table = np.zeros(len(u_list))
		x_curr = np.copy(x_init)
		x_traj = np.zeros((len(x_init), int(freq*T)))
		u_traj = np.zeros((u_sz, int(freq*T)))
		J_traj = np.zeros(int(freq*T))
		mode_traj = np.zeros(int(freq*T))
		nearest_smart_traj = np.zeros(int(freq*T))

		# State measurement noise
		noise = True
		stdev = 0
		if noise:
			stdev = 0.0005

		# Loop over time evolution
		for i in range(int(freq*T)):
			# Add noise
			noisy_x = np.random.normal(x_curr,stdev)

			# Loop over control action space
			for u in range(len(u_list[:,0])):
				vec = D.f(np.hstack([noisy_x, u_list[u]]))
				cost_table[u] = J(vec,xd) + Jcenter(vec,np.mean(vec[:5]),np.mean(vec[5:10]),0.6)
			
			# Update variables
			u_curr = u_list[np.argmin(cost_table)]
			J_curr = np.min(cost_table)
			x_curr = D.f(np.hstack([noisy_x, u_curr]))[:15]

			# Update time histories
			x_traj[:,i] = x_curr
			u_traj[:,i] = u_curr
			J_traj[i] = J_curr
			mode_traj[i] = D.indicator(D.svm_basis.fk(np.hstack([noisy_x, u_curr]),None))[0]
			nearest_smart_traj[i] = np.argmin([Jind(x_curr[j],x_curr[j+5],xd) for j in range(5)])
			
		# Centroid trajectory	
		x_traj_mean = np.mean(x_traj[:5],axis=0)
		y_traj_mean = np.mean(x_traj[5:10],axis=0)
		theta_traj_mean = np.mean(x_traj[10:15],axis=0)

		# Plot
		plt.plot(x_traj_mean-x_traj_mean[0], y_traj_mean-y_traj_mean[0])
		plt.xlabel('X (m)')
		plt.ylabel('Y (m)')
		plt.xlim([-0.25,0.25])
		plt.ylim([-0.25,0.25])
		plt.plot(x_traj_mean[-1]-x_traj_mean[0], y_traj_mean[-1]-y_traj_mean[0], 'r.')
		
		# Save
		out = np.vstack([x_traj[:10],u_traj])
		outstr = "data/control_results/greedy_direction_"+str(d)+"_run_"+str(t)+"_data.csv"
		# np.savetxt(outstr,out,delimiter=",")
	plt.plot(r*np.sin(time),r*np.cos(time), c='k')
plt.show()