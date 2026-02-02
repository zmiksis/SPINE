import numpy as np
import copy
from scipy import linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

def gen_expansion_matrices_PM(neuron):

	dt = neuron.settings.dt

	R = copy.deepcopy(neuron.graph.radius)
	Rer = (15./40.)*R

	dR = gen_central_mat()
	dR = np.matmul(dR,R)
	dRer = (15./40.)*dR

	dx = np.zeros_like(R)
	for i in range(len(dx)):
		dx[i] = np.mean([abs(x) for x in neuron.graph.edge_length[i]])

	# Expansion differential operator matrices
	D1M = gen_central_mat()
	D2M = gen_diffusion_mat()
	D3M = np.matmul(D2M, D1M)

	D1M = (D1M.T * 2.*(R*dR - Rer*dRer)/(R**2 - Rer**2)).T
	D2M = D2M + (D2M.T * ((dx**2)/4.)*(dR**2 - dRer**2)/(R**2 - Rer**2)).T
	D3M = (D3M.T * ((dx**2)/4.)*(R*dR - Rer*dRer)/(R**2 - Rer**2)).T
	
	LHS = 1. + ((dx**2)/12.)*(dR**2 - dRer**2)/(R**2 - Rer**2)

	# SBDF2 splitting
	ADFJ = ((D2M + D3M).T / LHS).T
	BDFJ = ((D1M).T / LHS).T

	LHSDFJO1 = np.eye(len(R)) - dt*neuron.settings.Dc*(ADFJ)
	LHSDFJ = np.eye(len(R)) - dt*(2./3.)*neuron.settings.Dc*(ADFJ)
	LHSDFJLU, LHSDFJP = linalg.lu_factor(LHSDFJ)

	LHSDFJO1b = np.eye(len(R)) - dt*neuron.settings.Db*(ADFJ)
	LHSDFJb = np.eye(len(R)) - dt*(2./3.)*neuron.settings.Db*(ADFJ)
	LHSDFJLUb, LHSDFJPb = linalg.lu_factor(LHSDFJb)

	# Lateral flux expansion
	J1 = gen_central_mat()
	J1 = (J1.T * dR*(2./R)).T
	J2 = gen_diffusion_mat()
	J3 = (1./3.)*np.matmul(J2, gen_central_mat())
	Jmat = np.eye(len(R)) * (2./R)
	Jmat += ((J1 + J2 + J3).T * ((dx**2)/(12.*R))).T
	Jmat = (Jmat.T / LHS).T

	# Reaction expansion
	rf = np.eye(len(R))
	r1 = gen_central_mat()
	r1 = (r1.T *(1./6.)*(dx**2)*dR*(R - Rer)/(R**2 - Rer**2)).T
	rf += r1
	r2 = gen_diffusion_mat()
	r2 = (r2.T * (1./24.)*(dx**2)).T
	rf += r2
	r3 = np.matmul(gen_diffusion_mat(),gen_central_mat())
	r3 = (r3.T * (1./240.)*(dx**4)*dR*(R - Rer)/(R**2 - Rer**2)).T
	rf += r3
	rf = (rf.T / LHS).T

	return LHSDFJO1, LHSDFJLU, LHSDFJP, \
			LHSDFJO1b, LHSDFJLUb, LHSDFJPb, \
			BDFJ, Jmat, rf, LHS

###########################################################

def gen_expansion_matrices_ER(neuron):

	dt = neuron.settings.dt

	R = copy.deepcopy(neuron.graph.radius)
	Rer = (15./40.)*R

	dR = gen_central_mat(neuron)
	dR = np.matmul(dR,Rer)

	dx = np.zeros_like(R)
	for i in range(len(dx)):
		dx[i] = np.mean([abs(x) for x in neuron.graph.edge_length[i]])

	# Expansion differential operator matrices
	D1M = gen_central_mat(neuron)
	D2M = gen_diffusion_mat(neuron)
	D3M = np.matmul(D2M, D1M)

	D1M = (D1M.T * (2./Rer)*dR).T
	D2M = D2M + (D2M.T * ((dx*dR)/(2.*Rer))**2).T
	D3M = (D3M.T * dR*((dx/2.)**2)/Rer).T
	
	LHS = 1. + (1./12.)*((dx*dR/Rer)**2)

	# SBDF2 splitting
	ADFJ = ((D2M + D3M).T / LHS).T
	BDFJ = ((D1M).T / LHS).T

	LHSDFJO1 = np.eye(len(R)) - dt*neuron.settings.De*(ADFJ)
	LHSDFJ = np.eye(len(R)) - dt*(2./3.)*neuron.settings.De*(ADFJ)
	LHSDFJLU, LHSDFJP = linalg.lu_factor(LHSDFJ)

	LHSDFJO1b = np.eye(len(R)) - dt*neuron.settings.Dbe*(ADFJ)
	LHSDFJb = np.eye(len(R)) - dt*(2./3.)*neuron.settings.Dbe*(ADFJ)
	LHSDFJLUb, LHSDFJPb = linalg.lu_factor(LHSDFJb)

	# Lateral flux expansion
	J1 = gen_central_mat(neuron)
	J1 = (J1.T * dR*(2./Rer)).T
	J2 = gen_diffusion_mat(neuron)
	J3 = (1./3.)*np.matmul(J2, gen_central_mat(neuron))
	Jmat = np.eye(len(R)) * (2./Rer)
	Jmat += ((J1 + J2 + J3).T * ((dx**2)/(12.*Rer))).T
	Jmat = (Jmat.T / LHS).T

	# Reaction expansion
	rf = np.eye(len(R))
	rf += (rf.T * (1./12.)*(dx*dR/Rer)**2).T
	r1 = gen_central_mat(neuron)
	r1 = (r1.T *(1./6.)*(dx**2)*(dR/Rer)).T
	rf += r1
	r2 = gen_diffusion_mat(neuron)
	r2 = (r2.T * ((1./24.)*(dx**2) + (1./160.)*((dx**2)*dR/Rer)**2)).T
	rf += r2
	r3 = np.matmul(gen_diffusion_mat(neuron),gen_central_mat(neuron))
	r3 = (r3.T * (1./240.)*(dx**4)*dR/Rer).T
	rf += r3
	rf = (rf.T / LHS).T

	return LHSDFJO1, LHSDFJLU, LHSDFJP, \
			LHSDFJO1b, LHSDFJLUb, LHSDFJPb, \
			BDFJ, Jmat, rf, LHS

###########################################################

def gen_SIAP_matrices(neuron):

	nodes = len(neuron.sol.C)
	dt = neuron.settings.dt

	# SIAP model diffusion matrices
	DIFF = gen_diffusion_mat(neuron)

	LHSdiffO1 = np.eye(nodes) - dt*neuron.settings.Dc*(DIFF)
	LHSdiff = np.eye(nodes) - dt*(2./3.)*neuron.settings.Dc*(DIFF)
	LHSdiffLU = splu(csc_matrix(LHSdiff))

	LHSdiffO1b = np.eye(nodes) - dt*neuron.settings.Db*(DIFF)
	LHSdiffb = np.eye(nodes) - dt*(2./3.)*neuron.settings.Db*(DIFF)
	LHSdiffLUb = splu(csc_matrix(LHSdiffb))

	LHSdiffO1er = np.eye(nodes) - dt*neuron.settings.De*(DIFF)
	LHSdiffer = np.eye(nodes) - dt*(2./3.)*neuron.settings.De*(DIFF)
	LHSdiffLUer = splu(csc_matrix(LHSdiffer))

	LHSdiffO1ber = np.eye(nodes) - dt*neuron.settings.Dbe*(DIFF)
	LHSdiffber = np.eye(nodes) - dt*(2./3.)*neuron.settings.Dbe*(DIFF)
	LHSdiffLUber = splu(csc_matrix(LHSdiffber))

	LHSdiffO1ip3 = np.eye(nodes) - dt*neuron.settings.Dp*(DIFF)
	LHSdiffip3 = np.eye(nodes) - dt*(2./3.)*neuron.settings.Dp*(DIFF)
	LHSdiffLUip3 = splu(csc_matrix(LHSdiffip3))

	return LHSdiffO1, LHSdiffLU, \
			LHSdiffO1b, LHSdiffLUb, \
			LHSdiffO1er, LHSdiffLUer, \
			LHSdiffO1ber, LHSdiffLUber, \
			LHSdiffO1ip3, LHSdiffLUip3

###########################################################

def volt_diffusion_mat(neuron):
	# See J. Rosado PhD dissertation (2022)

	nodes = len(neuron.sol.C)
	dt = neuron.settings.dt
	D = np.zeros((nodes,nodes))
	surface_area = np.ones(nodes)

	for idx in range(nodes):

		h = copy.deepcopy(neuron.graph.edge_length[idx])
		Rn = neuron.graph.radius[neuron.graph.neighbors[idx]]
		Ri = neuron.graph.radius[idx]

		gamma = 0.
		for i in range(len(h)):
			# gammai = Rn[i]
			gammai = Ri
			gammai /= (1. + (Ri/Rn[i])**2)*h[i]*np.mean(h)
			gamma += gammai
			D[idx, neuron.graph.neighbors[idx][i]] = gammai

		D[idx, idx] = (-1.)*gamma

		surface_area[idx] = 2.*neuron.settings.Cm*np.pi*Ri*np.sum(h)

	D /= neuron.settings.Rm*neuron.settings.Cm

	LHSdiffO1 = np.eye(nodes) - dt*(D)
	LHSdiff = np.eye(nodes) - dt*(2./3.)*(D)
	LHSdiffLU = splu(csc_matrix(LHSdiff))

	return LHSdiffO1, LHSdiffLU, surface_area

###########################################################

def gen_diffusion_mat(neuron):
	# Finite difference diffusion matrix for general branched neuron

	nodes = len(neuron.sol.C)
	D = np.zeros((nodes,nodes))

	for idx in range(nodes):

		if neuron.graph.ntype[idx] != 2:

			neighbors = []
			edge_length = []
			for i in range(len(neuron.graph.neighbors[idx])):
				if neuron.graph.ntype[neuron.graph.neighbors[idx][i]] != 2:
					neighbors.append(neuron.graph.neighbors[idx][i])
					edge_length.append(neuron.graph.edge_length[idx][i])

			h = copy.deepcopy(edge_length)
			hinv = [1./node for node in edge_length]

			# Diagonal entry
			D[idx,idx] = -np.sum(hinv)
			# Neighbor entries
			D[idx,neighbors] = hinv

			# Outer coefficient
			D[idx,:] *= 2./np.sum(h)

	return D

###########################################################

def gen_central_mat(neuron):
	# Central finite difference d/dx matrix for general branched neuron

	nodes = len(neuron.sol.C)
	A = np.zeros((nodes,nodes))

	for idx in range(nodes):

		h = copy.deepcopy(neuron.graph.edge_length[idx])
		h_pid = np.where(neuron.graph.neighbors[idx] == neuron.graph.pid[idx])[0]

		if len(h) == 2 and len(h_pid) > 0:
			for i in range(len(h)):
				if i == h_pid[0]:
					A[idx,neuron.graph.neighbors[idx][i]] = -1./np.sum(h)
				else:
					A[idx,neuron.graph.neighbors[idx][i]] = 1./np.sum(h)

		if len(h) == 3 and len(h_pid) > 0:
			for i in range(len(h)):
				if i == h_pid[0]:
					A[idx,neuron.graph.neighbors[idx][i]] = (-1./2.)*(h[1] + h[2] + 2.*h[0])/((h[2] + h[0])*(h[1] + h[0]))
				else:
					A[idx,neuron.graph.neighbors[idx][i]] = (1./2.)/(h[i] + h[0])

		if len(h_pid) == 0:
			# Identify 2-paths
			two_paths = []
			two_paths_dx = []
			for i in range(1,len(neuron.graph.neighbors[idx])):
				node1 = neuron.graph.neighbors[idx][i]
				for j in range(1,len(neuron.graph.neighbors[node1])):
					two_paths.append([])
					two_paths_dx.append([])
					path_num = len(two_paths) - 1

					two_paths[path_num].append(node1)
					two_paths[path_num].append(neuron.graph.neighbors[node1][j])

					two_paths_dx[path_num].append(neuron.graph.edge_length[idx][i])
					two_paths_dx[path_num].append(neuron.graph.edge_length[node1][j])

			# Find derivative along each path
			for i in range(len(two_paths)):

				h1 = two_paths_dx[i][0]
				h2 = two_paths_dx[i][1]

				node1 = two_paths[i][0]
				node2 = two_paths[i][1]

				outer_coef = 1.
				outer_coef /= h1*h2*(h1+h2)

				coef0 = -(2.*h1*h2 + h2*h2)*outer_coef
				coef1 = (h1+h2)*(h1+h2)*outer_coef
				coef2 = -(h1*h1)*outer_coef

				A[idx,idx] += coef0
				A[idx,node1] += coef1
				A[idx,node2] += coef2

		if len(h) == 1 and len(h_pid) == 1:

			node1 = neuron.graph.pid[idx]
			node2 = neuron.graph.pid[node1]

			h1 = abs(neuron.graph.edge_length[idx][0])
			h2 = abs(neuron.graph.edge_length[node1][0])

			outer_coef = 1.
			outer_coef /= h1*h2*(h1+h2)

			coef0 = (2.*h1*h2 + h2*h2)*outer_coef
			coef1 = -(h1+h2)*(h1+h2)*outer_coef
			coef2 = (h1*h1)*outer_coef

			A[idx,idx] += coef0
			A[idx,node1] += coef1
			A[idx,node2] += coef2

	return A

###########################################################

def gen_upwind_mat_manuscript(neuron):
	# Upwind finite difference advection matrix for general branched neuron
	# Leftover from FJ manuscript

	R = copy.deepcopy(neuron.graph.radius)
	dR = gen_central_mat_manuscript()
	dR = np.matmul(dR,R)

	dx = np.zeros_like(R)
	for i in range(len(dx)):
		dx[i] = np.mean([abs(x) for x in neuron.graph.edge_length[i]])

	nodes = len(R)
	A = np.zeros((nodes,nodes))

	for idx in range(nodes):

		# Check if node is end or near end
		end_node = False
		near_end_node = False
		if len(neuron.graph.neighbors[idx]) == 1: end_node = True
		for i in range(len(neuron.graph.neighbors[idx])):
			if any([len(neuron.graph.neighbors[x]) == 1 for x in neuron.graph.neighbors[idx]]):
				near_end_node = True

		# If not end or near end, compute upwind
		if not end_node:

			if dR[idx] > 0:

				# Identify 2-paths
				two_paths = []
				two_paths_dx = []
				for i in range(1,len(neuron.graph.neighbors[idx])):
					node1 = neuron.graph.neighbors[idx][i]
					for j in range(1,len(neuron.graph.neighbors[node1])):
						two_paths.append([])
						two_paths_dx.append([])
						path_num = len(two_paths) - 1

						two_paths[path_num].append(node1)
						two_paths[path_num].append(neuron.graph.neighbors[node1][j])

						two_paths_dx[path_num].append(neuron.graph.edge_length[idx][i])
						two_paths_dx[path_num].append(neuron.graph.edge_length[node1][j])

				# Find derivative along each path
				for i in range(len(two_paths)):

					h1 = two_paths_dx[i][0]
					h2 = two_paths_dx[i][1]

					node1 = two_paths[i][0]
					node2 = two_paths[i][1]

					# dR/dx along particular 2-path
					dR_idx = -(h1*h1)*R[node2] + (h1+h2)*(h1+h2)*R[node1] - (2.*h1*h2 + h2*h2)*R[idx]
					dR_idx /= h1*h2*(h1+h2)

					outer_coef = (2./R[idx])*dR_idx
					outer_coef /= h1*h2*(h1+h2)

					coef0 = -(2.*h1*h2 + h2*h2)*outer_coef
					coef1 = (h1+h2)*(h1+h2)*outer_coef
					coef2 = -(h1*h1)*outer_coef

					A[idx,idx] += coef0
					A[idx,node1] += coef1
					A[idx,node2] += coef2

				if near_end_node:

					A[idx,idx] =  -(2./R[idx])*dR[idx]*3./(2.*dx[idx])
					A[idx,node1] = (2./R[idx])*dR[idx]*4./(2.*dx[idx])

			elif  dR[idx] <= 0 and neuron.graph.pid[idx] != 0:

				node1 = neuron.graph.pid[idx]
				node2 = neuron.graph.pid[node1]

				h1 = abs(neuron.graph.edge_length[idx][0])
				h2 = abs(neuron.graph.edge_length[node1][0])

				dR_idx = dR[idx]

				outer_coef = (2./R[idx])*dR_idx
				outer_coef /= h1*h2*(h1+h2)

				coef0 = (2.*h1*h2 + h2*h2)*outer_coef
				coef1 = -(h1+h2)*(h1+h2)*outer_coef
				coef2 = (h1*h1)*outer_coef

				A[idx,idx] += coef0
				A[idx,node1] += coef1
				A[idx,node2] += coef2

		# If near end, adjust upwind
		if near_end_node:
			if neuron.graph.pid[idx] == 0 and dR[idx] < 0:
				A[idx,neuron.graph.neighbors[idx]] = 0.
				A[idx,0] = -(2./R[idx])*dR[idx]*2./(dx[idx])
				A[idx,idx] = (2./R[idx])*dR[idx]*2./(dx[idx])
			elif neuron.graph.pid[idx] != 0 and dR[idx] > 0:
				A[idx,idx] += -(2./R[idx])*dR[idx]*1./(2.*dx[idx])

		if end_node:
			left = min(idx,neuron.graph.neighbors[idx][0])
			right = max(idx,neuron.graph.neighbors[idx][0])
			if dR[idx] < 0:
				A[idx,left] = (2./R[idx])*dR[idx]*1./(dx[idx])
				A[idx,right] = -(2./R[idx])*dR[idx]*1./(dx[idx])
			else:
				A[idx,left] = -(2./R[idx])*dR[idx]*1./(dx[idx])
				A[idx,right] = (2./R[idx])*dR[idx]*1./(dx[idx])

	return A

###########################################################

def gen_central_mat_manuscript(neuron):
	# Central finite difference d/dx matrix for general branched neuron
	# Leftover from FJ manuscript

	nnodes = len(neuron.sol.C)
	A = np.zeros((nnodes,nnodes))

	for idx in range(nnodes):

		h = copy.deepcopy(neuron.graph.edge_length[idx])
		h_pid = np.where(neuron.graph.neighbors[idx] == neuron.graph.pid[idx])[0]
		if len(h_pid) > 0: h[h_pid[0]] *= -1.
		hinv = [1./node for node in h]

		# Diagonal entry
		A[idx,idx] = (-1.)*np.sum(hinv)
		# Neighbor entries
		A[idx,neuron.graph.neighbors[idx]] = hinv
		# Outer coefficient
		A[idx,:] *= 1./len(h)

	return A

###########################################################

def gen_central_mat_neuron_manuscript(neuron):
	# Central finite difference d/dx matrix for general branched neuron

	nnodes = neuron.sol.C.shape[0]
	A = np.zeros((nnodes,nnodes))

	for idx in range(nnodes):

		h = copy.deepcopy(neuron.graph.edge_length[idx])
		h_pid = np.where(neuron.graph.neighbors[idx] == neuron.graph.pid[idx])[0]

		if len(h) == 2 and len(h_pid) > 0:
			for i in range(len(h)):
				if i == h_pid[0]:
					A[idx,neuron.graph.neighbors[idx][i]] = -1./np.sum(h)
				else:
					A[idx,neuron.graph.neighbors[idx][i]] = 1./np.sum(h)

		if len(h) == 3 and len(h_pid) > 0:
			for i in range(len(h)):
				if i == h_pid[0]:
					A[idx,neuron.graph.neighbors[idx][i]] = (-1./2.)*(h[1] + h[2] + 2.*h[0])/((h[2] + h[0])*(h[1] + h[0]))
				else:
					A[idx,neuron.graph.neighbors[idx][i]] = (1./2.)/(h[i] + h[0])

		if len(h_pid) == 0:
			# Identify 2-paths
			two_paths = []
			two_paths_dx = []
			for i in range(1,len(neuron.graph.neighbors[idx])):
				node1 = neuron.graph.neighbors[idx][i]
				for j in range(1,len(neuron.graph.neighbors[node1])):
					two_paths.append([])
					two_paths_dx.append([])
					path_num = len(two_paths) - 1

					two_paths[path_num].append(node1)
					two_paths[path_num].append(neuron.graph.neighbors[node1][j])

					two_paths_dx[path_num].append(neuron.graph.edge_length[idx][i])
					two_paths_dx[path_num].append(neuron.graph.edge_length[node1][j])

			# Find derivative along each path
			for i in range(len(two_paths)):

				h1 = two_paths_dx[i][0]
				h2 = two_paths_dx[i][1]

				node1 = two_paths[i][0]
				node2 = two_paths[i][1]

				outer_coef = 1.
				outer_coef /= h1*h2*(h1+h2)

				coef0 = -(2.*h1*h2 + h2*h2)*outer_coef
				coef1 = (h1+h2)*(h1+h2)*outer_coef
				coef2 = -(h1*h1)*outer_coef

				A[idx,idx] += coef0
				A[idx,node1] += coef1
				A[idx,node2] += coef2

		if len(h) == 1 and len(h_pid) == 1:

			node1 = neuron.graph.pid[idx]
			node2 = neuron.graph.pid[node1]

			h1 = abs(neuron.graph.edge_length[idx][0])
			h2 = abs(neuron.graph.edge_length[node1][0])

			outer_coef = 1.
			outer_coef /= h1*h2*(h1+h2)

			coef0 = (2.*h1*h2 + h2*h2)*outer_coef
			coef1 = -(h1+h2)*(h1+h2)*outer_coef
			coef2 = (h1*h1)*outer_coef

			A[idx,idx] += coef0
			A[idx,node1] += coef1
			A[idx,node2] += coef2

	return A