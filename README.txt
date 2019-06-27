
Contents:
-DSS.py: main DSS code
-koopman.py: code for calculating a Koopman operator used by DSS
-basis.py: basis function class used in calculating Koopman operators
-koopman_cluster.py: non-parametric clustering code used by DSS
-smarticle_dss_modelID.py: example script using DSS to instantiate a model from smarticle observations
-smarticle_control.py: application of greedy controller for supersmarticle control using model
-data/: folder containing representative data used for instantiating a supersmarticle model, as well as data from supersmarticle control runs.

------------------------------------
Dynamical System Segmentation (DSS)
------------------------------------

Description:
-A family of models for system ID and control
-Classifies similar dynamic behaviors or any set of sequential measurements given a set of basis functions/observations of data
-The basis functions must be specified for the particular system in a file basis.py that should be included. A sample file with the expected format is in examples/slip

Usage:
-Needs basis functions for dynamics and classifier (second set can be None such that you only use one basis)
-Needs a window size in number of points, and an overlap percentage \in [0,1)
	-can be given as a frequency and time horizon (e.g. int(freq*time_horizon)) 
-Needs a minimum number of clusters which corresponds to minimum number of times you expect the least frequent behavior you want to pick up on occurs in the dataset
-Needs a measurement x and its sequential measurement xn
	-control vector is optional, can be called without it
-NOTE: koopman.py and basis.py are written with the assumption that you will use euler integration
	-all this implies is that there's a subtraction of an identity matrix in the koopman file as well as some references to a self.basis.dt member from the basis_functions class

Ex: D = DSS(basis,svm_basis,window_sz,overlap,min_cluster_num,x,xn,u)