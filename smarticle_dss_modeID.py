from basis import *
from DSS import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##################
## Import Data ###
##################
filename = 'data/x.csv'
x_world = np.genfromtxt(filename,delimiter=',') # world frame
filename = 'data/y.csv'
y_world = np.genfromtxt(filename,delimiter=',') # world frame
filename = 'data/theta.csv'
theta_world = np.genfromtxt(filename,delimiter=',') # world frame
filename = 'data/ring.csv'
ring_world = np.genfromtxt(filename,delimiter=',') # world frame
filename = 'data/inactive.csv'
inactive =  np.genfromtxt(filename,delimiter=',')

# Change to world frame
x_data = x_world
y_data = y_world
theta_data = theta_world

data = np.hstack([x_data,y_data,theta_data,inactive])
x = data[:-1,:].T
xn = data[1:,:].T

####################################
## Dynamical System Segmentation ###
####################################
freq = 120.0
min_cluster_num = 400
overlap = 0.5
window_sz = 2
basis = BasisFns()

################
# Generate model
# D = DSS(basis,None,window_sz,overlap,min_cluster_num,x,xn)
# D.save()

################
# Load model
D = DSS(basis,None,'dss_world.pkl')
state_labels = D.state_labels

####################################
############# Plot  ################
####################################
plot_kwds = {'alpha' : 0.3, 's' : 20, 'linewidths':0}
class_num = len(D.nodes)
x_mean = np.copy(ring_world[:-1,0])
y_mean = np.copy(ring_world[:-1,1])
palette=[(0.6,0,0.6),(1,0,1),(1,0.6,1),(1,1,.7),(1,0.9,0.1),(1,.5,0),(1,.7,.3),(0.6,0,0),(1,.6,.6),(.9,.1,.3)]
plt.figure()
for i in range(class_num):
    mask = (state_labels == np.ones(np.shape(state_labels))*i)
    plt.scatter(x_mean[mask], y_mean[mask], c=palette[i], **plot_kwds)
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y$',fontsize=16)
plt.title(r'$Supersmarticle\ Behavior\ Identification$')
plt.show()