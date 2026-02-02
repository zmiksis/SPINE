import numpy as np
from dataclasses import dataclass

###########################################################

@dataclass
class Graph:
	nid: int = None
	pid: int = None
	coord: float = None
	neighbors: int = None
	edge_length: float = None
	radius: float = None


###########################################################

def read_swc(neuron, filename):
	# Read in .swc file

	f = open(filename, "r")
	L = f.readlines()
	# Remove initial lines
	L = [x for x in L if not "#" in x]
	# Remove new line character
	L = [x.replace('\n','').split() for x in L]
	# Convert to floats
	for i in range(len(L)):
		L[i] = [float(x) for x in L[i]]

	# Get id of nodes
	nid_old = np.array([int(node[0]) for node in L])
	# Get parent id of nodes
	pid_old = np.array([int(node[6]) for node in L])

	# Map to sequential node numbers
	neuron.graph.nid = np.arange(0,nid_old.shape[0])
	neuron.graph.pid = np.zeros_like(pid_old)
	for i in range(pid_old.shape[0]):
		idx = np.where(nid_old == pid_old[i])[0]
		if len(idx) == 0: idx = -1
		neuron.graph.pid[i] = int(idx)

	# Get coordinates of each node
	neuron.graph.coord = neuron.settings.len_scale*np.array([np.array(node[2:5]) for node in L])

	# Get dendritic radius at each node
	neuron.graph.radius = neuron.settings.len_scale*np.array([node[5] for node in L])
	if neuron.geom_builder['const_rad']: neuron.graph.radius = np.ones_like(neuron.graph.radius)*neuron.settings.rad

	# Get node type (soma: 1, axon: 2, basal dendrite: 3, apical dendrite: 4)
	neuron.graph.ntype = np.array([node[1] for node in L])

###########################################################

def refine_graph(neuron):

	nnodes = len(neuron.graph.nid)

	for i in range(1,nnodes):

		id_new = nnodes - 1 + i

		neuron.graph.nid = np.append(neuron.graph.nid, id_new)
		neuron.graph.pid = np.append(neuron.graph.pid, neuron.graph.pid[i])

		coord_new = (neuron.graph.coord[i] + neuron.graph.coord[neuron.graph.pid[i]])/2.
		neuron.graph.coord = np.append(neuron.graph.coord, [coord_new], axis=0)

		radius_new = (neuron.graph.radius[i] + neuron.graph.radius[neuron.graph.pid[i]])/2.
		neuron.graph.radius = np.append(neuron.graph.radius, radius_new)

		neuron.graph.pid[i] = id_new

###########################################################

def gen_neuron(neuron):
	# Generate neuron from given geometry

	# Initiate lists of neighbors and edge lengths
	neuron.graph.neighbors = []
	neuron.graph.edge_length = []

	for i in range(neuron.graph.nid.shape[0]):

		# Find neighbors of each node
		neuron.graph.neighbors.append([])
		if neuron.graph.pid[neuron.graph.nid[i]] >= 0: neuron.graph.neighbors[i].append(neuron.graph.pid[neuron.graph.nid[i]])
		for j in range(neuron.graph.pid.shape[0]):
			if neuron.graph.pid[j] == neuron.graph.nid[i]: neuron.graph.neighbors[i].append(neuron.graph.nid[j])

	    # Find length of each edge
		neuron.graph.edge_length.append([])
		for j in range(len(neuron.graph.neighbors[i])):
			if neuron.geom_builder['const_edge']:
				neuron.graph.edge_length[i].append(neuron.settings.h)
			else:
				neuron.graph.edge_length[i].append(np.linalg.norm(neuron.graph.coord[i]-neuron.graph.coord[neuron.graph.neighbors[i][j]],ord=2))

###########################################################

def gen_beam(neuron, L=64., n=500):
	# Generate beam geometry

	x = np.linspace(0.,L,n)

	# Get coordinates of each node
	neuron.graph.coord = np.vstack([x,np.zeros_like(x),np.zeros_like(x)]).T
	# Assign id to each node
	neuron.graph.nid = np.arange(0,x.shape[0])
	# Get parent id to each node -- soma has parent id -1
	neuron.graph.pid = neuron.graph.nid-1

	neuron.graph.radius = np.ones_like(x)*neuron.settings.rad

	neuron.graph.ntype = 3*np.ones_like(x)

###########################################################

def gen_branch(neuron, L=64., n=500, factor=1., delta=0.):
	# Generate branch geometry

	x = np.linspace(0.,L,n)

	# Get coordinates of each node
	neuron.graph.coord = np.vstack([x,np.zeros_like(x),np.zeros_like(x)]).T
	# Assign id to each node
	neuron.graph.nid = np.arange(0,x.shape[0])
	# Get parent id to each node -- soma has parent id -1
	neuron.graph.pid = neuron.graph.nid-1

	dx = x[1]-x[0]

	for i in range(n//2):
		# Get coordinates of each node
		neuron.graph.coord = np.append(neuron.graph.coord, [[x[n//2]+(i+1)*np.sqrt((factor*dx)**2 / 2),(i+1)*np.sqrt((factor*dx)**2 / 2), 0.]], axis=0)
		# Assign id to each node
		neuron.graph.nid = np.append(neuron.graph.nid, x.shape[0]+i)
		# Get parent id to each node -- branch originate from center
		if i == 0: neuron.graph.pid = np.append(neuron.graph.pid, n//2)
		else: neuron.graph.pid = np.append(neuron.graph.pid, neuron.graph.nid[x.shape[0]+i]-1)

	neuron.graph.radius = np.ones_like(neuron.graph.nid)*neuron.settings.rad

	for i in range(1,len(neuron.graph.radius)):
		neuron.graph.radius[i] = delta*(L/(n-1))*i + neuron.settings.rad

	neuron.graph.ntype = 3*np.ones_like(neuron.graph.nid)
