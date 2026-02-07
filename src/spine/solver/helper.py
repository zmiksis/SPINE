import numpy as np
import copy
from scipy import linalg

from spine.solver.discretization_matrices import *
from spine.vdcc import initialize_vdcc_states, volt_gate
from spine.writer import vtu_write
from spine import er
from spine import cytosol as cyt
from spine.utils.math import integrate
from spine.flux_components import FluxComponents

def update_solution_arrays(neuron, neuronDict):
	"""Batch unit conversions from computational (neuronDict) to display units (sol).

	This centralizes all unit conversions in one place for better maintainability.
	Conversions: umol/um^3 → μM (×1e15), V → mV (×1e3)
	"""
	# Cytosolic concentrations (umol/um^3 → μM)
	neuron.sol.C[:] = 1e15 * neuronDict['c']
	neuron.sol.B = 1e15 * neuronDict['b']

	# ER concentrations (umol/um^3 → μM)
	neuron.sol.CE = 1e15 * neuronDict['ce']
	neuron.sol.BE = 1e15 * neuronDict['be']

	# IP3 (if enabled)
	if neuron.model['IP3']:
		neuron.sol.IP3 = 1e15 * neuronDict['ip3']

	# Voltage (V → mV)
	neuron.sol.V[:] = 1e3 * neuronDict['V0']

def set_dictionary(neuron, neuronDict, er_scale):

	neuronDict['dt'] = neuron.settings.dt
	neuronDict['nnodes'] = neuron.sol.C.shape[0]

	neuronDict['R'] = neuron.graph.radius.astype('float64').copy()
	neuronDict['Rer'] = er_scale*neuronDict['R']

	neuronDict['siap_JPM_scale'] = 2.*neuronDict['R']/(neuronDict['R']**2 - neuronDict['Rer']**2)
	neuronDict['siap_JER_scale'] = 2.*neuronDict['Rer']/(neuronDict['R']**2 - neuronDict['Rer']**2)

	# Pre-compute node type indices to avoid repeated np.where() calls
	neuronDict['soma_idx'] = np.where(neuron.graph.ntype == 1)[0]
	neuronDict['axon_idx'] = np.where(neuron.graph.ntype == 2)[0]
	# Also store on neuron for use in flux functions
	neuron._axon_idx = neuronDict['axon_idx']
	neuron.sol.soma = np.mean(neuron.sol.C[neuronDict['soma_idx']])

	# Cache results path for recording to avoid repeated string concatenation
	neuron._results_path = neuron.settings.top_folder + 'results/' + neuron.settings.output_folder + '/'
	neuron.sol.avg = np.mean(neuron.sol.C)
	neuron.sol.avg_er = np.mean(neuron.sol.CE)

	if neuron.recorder['soma']:
		np.savetxt(neuron._results_path + 'soma.txt', np.array([0., neuron.sol.soma]).reshape(1,-1))
	if neuron.recorder['avg_cyt']:
		np.savetxt(neuron._results_path + 'avg_cyt.txt', np.array([0., neuron.sol.avg]).reshape(1,-1))
	if neuron.recorder['avg_er']:
		np.savetxt(neuron._results_path + 'avg_er.txt', np.array([0., neuron.sol.avg_er]).reshape(1,-1))

	neuronDict['c'] = 1e-15*neuron.sol.C
	neuronDict['b'] = 1e-15*neuron.sol.B

	neuronDict['ce'] = 1e-15*neuron.sol.CE
	neuronDict['be'] = 1e-15*neuron.sol.BE

	neuronDict['ip3'] = 1e-15*neuron.sol.IP3

	# Initialize current state (will be computed in first step)
	neuronDict['j'] = None
	neuronDict['r'] = None
	neuronDict['je'] = None
	neuronDict['re'] = None
	neuronDict['jip3'] = None
	neuronDict['rip3'] = None

	# Initialize history state (needed for second-order SBDF)
	neuronDict['j0'] = None
	neuronDict['r0'] = None
	neuronDict['c0'] = None
	neuronDict['b0'] = None

	neuronDict['je0'] = None
	neuronDict['re0'] = None
	neuronDict['ce0'] = None
	neuronDict['be0'] = None

	neuronDict['jip30'] = None
	neuronDict['rip30'] = None
	neuronDict['ip30'] = None

###########################################################

def set_states(neuron, neuronDict):

	print('Initializing VDCC states...')
	if neuron.settings.Vdat is None: neuronDict['V0'] = neuron.settings.Vrest*np.ones_like(neuronDict['c'])
	else: neuronDict['V0'] = neuron.settings.volt_scale*np.loadtxt(neuron.settings.Vdat+'vm_0000000.dat', usecols=3)
	neuronDict['V00'] = neuronDict['V0'].copy()
	neuronDict['vdccc_states'] = initialize_vdcc_states(neuron,neuronDict['V0'])

	print('Initializing ryanidine states for neuron ...')
	neuronDict['ryr_state'] = er.init_ryr_state(neuron, neuronDict['c'])

	print('Initializing leakage velocity for neuron ...')
	neuronDict['G0'], _ = volt_gate(neuron, neuronDict['V0'], neuronDict['dt'], neuronDict['vdccc_states'])
	cyt.init_vel(neuron, neuronDict['c'], neuronDict['V0'], neuronDict['G0'])
	er.init_vel_e(neuron, neuronDict['c'], neuronDict['ce'], neuronDict['ip3'], neuronDict['ryr_state'], neuronDict['dt'])

###########################################################

def set_matrices(neuron, neuronDict, solver):
	
	print('Setting SBDF matrices for neuron ...')

	if solver == 'SIAP':
		neuronDict['SAc0'], neuronDict['SAcLU'], neuronDict['SAb0'], neuronDict['SAbLU'], \
			neuronDict['SAce0'], neuronDict['SAceLU'], neuronDict['SAbe0'], neuronDict['SAbeLU'], \
			neuronDict['SAip30'], neuronDict['SAip3LU'] = gen_SIAP_matrices(neuron)
	elif solver == 'FJ':
		neuronDict['Ac0'], neuronDict['AcLU'], neuronDict['AcP'], neuronDict['Ab0'], neuronDict['AbLU'], neuronDict['AbP'], \
			neuronDict['Bcb'], neuronDict['Jmat'], neuronDict['rf'], neuronDict['LHSc'] = gen_expansion_matrices_PM(neuron)
		neuronDict['Ace0'], neuronDict['AceLU'], neuronDict['AceP'], neuronDict['Abe0'], neuronDict['AbeLU'], neuronDict['AbeP'], \
			neuronDict['Bcebe'], neuronDict['Jemat'], neuronDict['rfe'], neuronDict['LHSce'] = gen_expansion_matrices_ER(neuron)
		
	if neuron.model['voltage_coupling']:
		print('Setting voltage diffusion matrices for neuron ...')
		neuronDict['VdiffO1'], neuronDict['VdiffLU'], neuronDict['scaled_surface_area'] = volt_diffusion_mat(neuron)

###########################################################

def SBDF_step(neuron, neuronDict, comm, comm_iter, solver, writeStep, i, t0):

	dt = neuronDict['dt']
	nnodes = neuronDict['nnodes']
	Rer = neuronDict['Rer']
	siap_JPM_scale = neuronDict['siap_JPM_scale']
	siap_JER_scale = neuronDict['siap_JER_scale']

	# Create transient flux container if any flux recording is enabled
	fluxes = None
	if _any_flux_recording_enabled(neuron):
		fluxes = FluxComponents()

	if solver == 'SIAP':
		SAc0 = neuronDict['SAc0']
		SAcLU = neuronDict['SAcLU']
		SAb0 = neuronDict['SAb0']
		SAbLU = neuronDict['SAbLU']

		SAce0 = neuronDict['SAce0']
		SAceLU = neuronDict['SAceLU']
		SAbe0 = neuronDict['SAbe0']
		SAbeLU = neuronDict['SAbeLU']

		if neuron.model['IP3']:
			SAip30 = neuronDict['SAip30']
			SAip3LU = neuronDict['SAip3LU']

	elif solver == 'FJ':
		Ac0 = neuronDict['Ac0']
		AcLU = neuronDict['AcLU']
		AcP = neuronDict['AcP']
		Ab0 = neuronDict['Ab0']
		AbLU = neuronDict['AbLU']
		AbP = neuronDict['AbP']
		Bcb = neuronDict['Bcb']
		Jmat = neuronDict['Jmat']
		rf = neuronDict['rf']
		LHSc = neuronDict['LHSc']

		Ace0 = neuronDict['Ace0']
		AceLU = neuronDict['AceLU']
		AceP = neuronDict['AceP']
		Abe0 = neuronDict['Abe0']
		AbeLU = neuronDict['AbeLU']
		AbeP = neuronDict['AbeP']
		Bcebe = neuronDict['Bcebe']
		Jemat = neuronDict['Jemat']
		rfe = neuronDict['rfe']
		LHSce = neuronDict['LHSce']

	if neuron.model['voltage_coupling']:
		VdiffO1 = neuronDict['VdiffO1']
		VdiffLU = neuronDict['VdiffLU']
		scaled_surface_area = neuronDict['scaled_surface_area']


	# Write visualization data
	if writeStep is not None and i % writeStep == 0:
		vtu_write(neuron,t0,i)

	neuronDict['ryr_state'] = er.update_ryr_state(neuronDict['ryr_state'], neuronDict['c'], dt)

	if not neuron.model['voltage_coupling']: neuronDict['V0'] = neuron.settings.read_voltage(tid=i,length=nnodes)
	neuronDict['G'], neuronDict['vdccc_states'] = volt_gate(neuron, neuronDict['V0'], dt, neuronDict['vdccc_states'])

	if neuron.model['IP3']:
		JPM, JIP3 = cyt.JPM(neuron, neuronDict['c'], neuronDict['ip3'], t0, neuronDict['V0'], neuronDict['G'], comm, comm_iter, fluxes=fluxes)
		if fluxes is not None:
			fluxes.total_ip3 = JIP3.copy() if neuron.recorder.get('flux_total_ip3', False) else None
	else:
		JPM = cyt.JPM(neuron, neuronDict['c'], neuronDict['ip3'], t0, neuronDict['V0'], neuronDict['G'], comm, comm_iter, fluxes=fluxes)
	JER, JSOC = er.JER(neuron, neuronDict['c'], neuronDict['ce'], neuronDict['ip3'], neuronDict['ryr_state'], neuronDict['V0'], dt, fluxes=fluxes)

	if neuron.model['CYT_buffer']: r = cyt.reaction(neuron, neuronDict['c'], neuronDict['b'])
	else: r = np.zeros_like(neuronDict['c'])

	if neuron.model['ER_buffer']: re = er.reaction_er(neuron, neuronDict['ce'], neuronDict['be'])
	else: re = np.zeros_like(neuronDict['ce'])

	if neuron.model['IP3']: rip3 = er.reaction_ip3(neuron, neuronDict['ip3'])
	else: rip3 = np.zeros_like(neuronDict['ip3'])

	if solver == 'SIAP':

		j = siap_JPM_scale*JPM + siap_JER_scale*JER
		if neuron.model['IP3']: jip3 = siap_JPM_scale*JIP3
		je = (-2./Rer)*JER + (2./Rer)*JSOC

		if i == 0:
			ctemp = solve_step(SAc0, dt, neuronDict['c'], j+r)
			btemp = solve_step(SAb0, dt, neuronDict['b'], r)
			cetemp = solve_step(SAce0, dt, neuronDict['ce'], je+re)
			betemp = solve_step(SAbe0, dt, neuronDict['be'], re)

			if neuron.model['IP3']:
				ip3temp = solve_step(SAip30, dt, neuronDict['ip3'], jip3 + rip3)

			if neuron.model['voltage_coupling']:
				neuron.CableSettings.current_profile(comm=comm, comm_iter=comm_iter)
				neuron.CableSettings.update_gating_variables(comm=comm, comm_iter=comm_iter)
				Vprof = neuron.CableSettings.profile

				# Apply voltage clamp if set (overrides computed profile)
				if neuron.settings.voltage_clamp is not None:
					Vprof[neuron.settings.voltage_clamp[0]] = 0.

				vtemp = solve_step(VdiffO1, dt, neuronDict['V0'], Vprof)

				# Sherman-Morrison correction for voltage clamp (if enabled)
				if neuron.settings.voltage_clamp is not None:
					clamp_idx = neuron.settings.voltage_clamp[0]
					e_clamp = np.zeros_like(neuronDict['V0'])
					e_clamp[clamp_idx] = 1.
					z = linalg.solve(VdiffO1, e_clamp)
					vtemp += (1e3 * neuron.settings.voltage_clamp[1] - vtemp[clamp_idx]) * z / z[clamp_idx]


		else:
			ctemp = solve_step(SAcLU, dt, neuronDict['c'], j + r, neuronDict['c0'], neuronDict['j0'] + neuronDict['r0'])
			btemp = solve_step(SAbLU, dt, neuronDict['b'], r, neuronDict['b0'], neuronDict['r0'])
			cetemp = solve_step(SAceLU, dt, neuronDict['ce'], je + re, neuronDict['ce0'], neuronDict['je0'] + neuronDict['re0'])
			betemp = solve_step(SAbeLU, dt, neuronDict['be'], re, neuronDict['be0'], neuronDict['re0'])

			if neuron.model['IP3']:
				ip3temp = solve_step(SAip3LU, dt, neuronDict['ip3'], jip3 + rip3, neuronDict['ip30'], neuronDict['jip30'] + neuronDict['rip30'])

			if neuron.model['voltage_coupling']:
				neuron.CableSettings.current_profile(comm=comm, comm_iter=comm_iter)
				neuron.CableSettings.update_gating_variables(comm=comm, comm_iter=comm_iter)
				Vprof = neuron.CableSettings.profile
				Vprof0 = neuron.CableSettings.profile0

				# Apply voltage clamp if set (overrides computed profile)
				if neuron.settings.voltage_clamp is not None:
					Vprof[neuron.settings.voltage_clamp[0]] = 0.
					Vprof0[neuron.settings.voltage_clamp[0]] = 0.

				vtemp = solve_step(VdiffLU, dt, neuronDict['V0'], Vprof, neuronDict['V00'], Vprof0)

				# Sherman-Morrison correction for voltage clamp (if enabled)
				if neuron.settings.voltage_clamp is not None:
					clamp_idx = neuron.settings.voltage_clamp[0]
					e_clamp = np.zeros_like(neuronDict['V0'])
					e_clamp[clamp_idx] = 1.
					z = VdiffLU.solve(e_clamp)
					vtemp += (1e3 * neuron.settings.voltage_clamp[1] - vtemp[clamp_idx]) * z / z[clamp_idx]

	elif solver == 'FJ':

		j = np.matmul(Jmat, JPM) + np.matmul(((Jemat.T * LHSce) / LHSc).T, JER)
		je = np.matmul(Jemat, JSOC) - np.matmul(Jemat, JER) 

		r = np.matmul(rf, r)
		re = np.matmul(rfe, re)

		if i == 0:
			fc = neuronDict['c'] + dt*(neuron.settings.Dc*np.matmul(Bcb, neuronDict['c']) + j + r)
			fb = neuronDict['b'] + dt*(neuron.settings.Db*np.matmul(Bcb, neuronDict['b']) + r)
			ctemp = np.linalg.solve(Ac0, fc)
			btemp = np.linalg.solve(Ab0, fb)

			fce = neuronDict['ce'] + dt*(neuron.settings.De*np.matmul(Bcebe, neuronDict['ce']) + je + re)
			fbe = neuronDict['be'] + dt*(neuron.settings.Dbe*np.matmul(Bcebe, neuronDict['be']) + re)
			cetemp = np.linalg.solve(Ace0, fce)
			betemp = np.linalg.solve(Abe0, fbe)
		else:
			fc = (4./3.)*neuronDict['c'] + dt*(4./3.)*(neuron.settings.Dc*np.matmul(Bcb, neuronDict['c']) + j + r) \
				- (1./3.)*neuronDict['c0'] - dt*(2./3.)*(neuron.settings.Dc*np.matmul(Bcb, neuronDict['c0']) + neuronDict['j0'] + neuronDict['r0'])
			fb = (4./3.)*neuronDict['b'] + dt*(4./3.)*(neuron.settings.Db*np.matmul(Bcb, neuronDict['b']) + r) \
				- (1./3.)*neuronDict['b0'] - dt*(2./3.)*(neuron.settings.Db*np.matmul(Bcb, neuronDict['b0']) + neuronDict['r0'])
			ctemp = linalg.lu_solve((AcLU, AcP), fc)
			btemp = linalg.lu_solve((AbLU, AbP), fb)

			fce = (4./3.)*neuronDict['ce'] + dt*(4./3.)*(neuron.settings.De*np.matmul(Bcebe, neuronDict['ce']) + je + re) \
				- (1./3.)*neuronDict['ce0'] - dt*(2./3.)*(neuron.settings.De*np.matmul(Bcebe, neuronDict['ce0']) + neuronDict['je0'] + neuronDict['re0'])
			fbe = (4./3.)*neuronDict['be'] + dt*(4./3.)*(neuron.settings.Dbe*np.matmul(Bcebe, neuronDict['be']) + re) \
				- (1./3.)*neuronDict['be0'] - dt*(2./3.)*(neuron.settings.Dbe*np.matmul(Bcebe, neuronDict['be0']) + neuronDict['re0'])
			cetemp = linalg.lu_solve((AceLU, AceP), fce)
			betemp = linalg.lu_solve((AbeLU, AbeP), fbe)

	# Maintain non-negative concentrations
	ctemp[ctemp < 5.e-18] = 5.e-18

	# Use pointer swapping instead of copying to reduce memory bandwidth
	# For first timestep (i=0), initialize history arrays with copies
	# For subsequent timesteps, use pointer swapping
	if i == 0:
		# First timestep: copy to initialize history
		neuronDict['j'] = j
		neuronDict['j0'] = j.copy()
		neuronDict['r'] = r
		neuronDict['r0'] = r.copy()
		neuronDict['c0'] = neuronDict['c'].copy()
		neuronDict['c'] = ctemp
		neuronDict['b0'] = neuronDict['b'].copy()
		neuronDict['b'] = btemp

		neuronDict['je'] = je
		neuronDict['je0'] = je.copy()
		neuronDict['re'] = re
		neuronDict['re0'] = re.copy()
		neuronDict['ce0'] = neuronDict['ce'].copy()
		neuronDict['ce'] = cetemp
		neuronDict['be0'] = neuronDict['be'].copy()
		neuronDict['be'] = betemp

		if neuron.model['IP3']:
			neuronDict['jip3'] = jip3
			neuronDict['jip30'] = jip3.copy()
			neuronDict['rip3'] = rip3
			neuronDict['rip30'] = rip3.copy()
			neuronDict['ip30'] = neuronDict['ip3'].copy()
			neuronDict['ip3'] = ip3temp

		if neuron.model['voltage_coupling']:
			neuronDict['V00'] = neuronDict['V0'].copy()
			neuronDict['V0'] = vtemp
			neuron.CableSettings.profile0 = Vprof.copy()
	else:
		# Subsequent timesteps: pointer swapping (no copying)
		neuronDict['j0'] = neuronDict['j']
		neuronDict['j'] = j
		neuronDict['r0'] = neuronDict['r']
		neuronDict['r'] = r
		neuronDict['c0'] = neuronDict['c']
		neuronDict['c'] = ctemp
		neuronDict['b0'] = neuronDict['b']
		neuronDict['b'] = btemp

		# Swap ER state history
		neuronDict['je0'] = neuronDict['je']
		neuronDict['je'] = je
		neuronDict['re0'] = neuronDict['re']
		neuronDict['re'] = re
		neuronDict['ce0'] = neuronDict['ce']
		neuronDict['ce'] = cetemp
		neuronDict['be0'] = neuronDict['be']
		neuronDict['be'] = betemp

		# Swap IP3 state history (if enabled)
		if neuron.model['IP3']:
			neuronDict['jip30'] = neuronDict['jip3']
			neuronDict['jip3'] = jip3
			neuronDict['rip30'] = neuronDict['rip3']
			neuronDict['rip3'] = rip3
			neuronDict['ip30'] = neuronDict['ip3']
			neuronDict['ip3'] = ip3temp

		# Swap voltage state history (if enabled)
		if neuron.model['voltage_coupling']:
			neuronDict['V00'] = neuronDict['V0']
			neuronDict['V0'] = vtemp
			neuron.CableSettings.profile0 = Vprof

	# Pointer swapping: swap C/C0 and V/V0 before writing
	# After swap, C0 has old data (no write needed), C gets new data
	neuron.swap_history_arrays()

	# Batch unit conversions: neuronDict (computational) → sol (display)
	update_solution_arrays(neuron, neuronDict)

	# Update integrated totals (if recording enabled)
	if neuron.recorder['total_cyt']:
		neuron.sol.total_cyt = integrate(neuron.sol.C, neuron.sol.C0, dt, neuron.sol.total_cyt)

	if neuron.recorder['total_er']:
		neuron.sol.total_er = integrate(neuron.sol.CE, neuron.sol.CE0, dt, neuron.sol.total_er)

	# Return fluxes object for recording (transient, will be garbage collected)
	return fluxes

###########################################################

def solve_step(A, dt, u, F, u0=None, f0=None):
	# First order: Assume A is full
	# Second order: Assume A is LU factorized
	
	if u0 is None:
		b = u + dt*F
		u_new = np.linalg.solve(A, b)
	else:
		b = (4./3.)*u + dt*(4./3.)*F - (1./3.)*u0 - dt*(2./3.)*f0
		u_new = A.solve(b)

	return u_new

###########################################################

def _any_flux_recording_enabled(neuron):
	"""Check if any flux recording is enabled."""
	flux_keys = [
		'flux_pmca', 'flux_ncx', 'flux_pm_leak', 'flux_vdcc', 'flux_synapse',
		'flux_synapse_ip3', 'flux_ryr', 'flux_serca', 'flux_er_leak',
		'flux_ip3r', 'flux_soc', 'flux_total_pm', 'flux_total_er', 'flux_total_ip3'
	]
	return any(neuron.recorder.get(key, False) for key in flux_keys)

###########################################################

def record_steps(neuron, neuronDict, t0, fluxes=None):

	soma_idx = neuronDict['soma_idx']

	if neuron.recorder['full']:
		with open(neuron._results_path + 'cyt.txt', 'ab') as f:
			np.savetxt(f, neuron.sol.C.reshape(1,-1))

	if neuron.recorder['full_er']:
		with open(neuron._results_path + 'er.txt', 'ab') as f:
			np.savetxt(f, neuron.sol.CE.reshape(1,-1))

	if neuron.recorder['full_voltage']:
		with open(neuron._results_path + 'volt.txt', 'ab') as f:
			np.savetxt(f, neuron.sol.V.reshape(1,-1))

	if neuron.recorder['nodes']:
		for n in neuron.recorder['nodes']:
			filename = neuron._results_path + f'node_{n}.txt'
			with open(filename, 'ab') as f:
				np.savetxt(f, np.array([t0, neuron.sol.C[n], neuron.sol.CE[n], neuron.sol.IP3[n], neuron.sol.V[n]]).reshape(1,-1))

	neuron.sol.soma = np.mean(neuron.sol.C[soma_idx])
	neuron.sol.avg = np.mean(neuron.sol.C)
	neuron.sol.avg_er = np.mean(neuron.sol.CE)

	if neuron.recorder['soma']:
		with open(neuron._results_path + 'soma.txt', 'ab') as f:
			np.savetxt(f, np.array([t0, neuron.sol.soma]).reshape(1,-1))
	if neuron.recorder['avg_cyt']:
		with open(neuron._results_path + 'avg_cyt.txt', 'ab') as f:
			np.savetxt(f, np.array([t0, neuron.sol.avg]).reshape(1,-1))
	if neuron.recorder['avg_er']:
		with open(neuron._results_path + 'avg_er.txt', 'ab') as f:
			np.savetxt(f, np.array([t0, neuron.sol.avg_er]).reshape(1,-1))

	# Record individual flux components (transient data, immediately streamed to disk)
	if fluxes is not None:
		# Use cached results path
		results_path = neuron._results_path

		# Cytosolic/PM fluxes
		if neuron.recorder.get('flux_pmca', False) and fluxes.pmca is not None:
			with open(results_path + 'flux_pmca.txt', 'ab') as f:
				np.savetxt(f, fluxes.pmca.reshape(1, -1))

		if neuron.recorder.get('flux_ncx', False) and fluxes.ncx is not None:
			with open(results_path + 'flux_ncx.txt', 'ab') as f:
				np.savetxt(f, fluxes.ncx.reshape(1, -1))

		if neuron.recorder.get('flux_pm_leak', False) and fluxes.pm_leak is not None:
			with open(results_path + 'flux_pm_leak.txt', 'ab') as f:
				np.savetxt(f, fluxes.pm_leak.reshape(1, -1))

		if neuron.recorder.get('flux_vdcc', False) and fluxes.vdcc is not None:
			with open(results_path + 'flux_vdcc.txt', 'ab') as f:
				np.savetxt(f, fluxes.vdcc.reshape(1, -1))

		if neuron.recorder.get('flux_synapse', False) and fluxes.synapse is not None:
			with open(results_path + 'flux_synapse.txt', 'ab') as f:
				np.savetxt(f, fluxes.synapse.reshape(1, -1))

		if neuron.recorder.get('flux_synapse_ip3', False) and fluxes.synapse_ip3 is not None:
			with open(results_path + 'flux_synapse_ip3.txt', 'ab') as f:
				np.savetxt(f, fluxes.synapse_ip3.reshape(1, -1))

		# ER fluxes
		if neuron.recorder.get('flux_ryr', False) and fluxes.ryr is not None:
			with open(results_path + 'flux_ryr.txt', 'ab') as f:
				np.savetxt(f, fluxes.ryr.reshape(1, -1))

		if neuron.recorder.get('flux_serca', False) and fluxes.serca is not None:
			with open(results_path + 'flux_serca.txt', 'ab') as f:
				np.savetxt(f, fluxes.serca.reshape(1, -1))

		if neuron.recorder.get('flux_er_leak', False) and fluxes.er_leak is not None:
			with open(results_path + 'flux_er_leak.txt', 'ab') as f:
				np.savetxt(f, fluxes.er_leak.reshape(1, -1))

		if neuron.recorder.get('flux_ip3r', False) and fluxes.ip3r is not None:
			with open(results_path + 'flux_ip3r.txt', 'ab') as f:
				np.savetxt(f, fluxes.ip3r.reshape(1, -1))

		if neuron.recorder.get('flux_soc', False) and fluxes.soc is not None:
			with open(results_path + 'flux_soc.txt', 'ab') as f:
				np.savetxt(f, fluxes.soc.reshape(1, -1))

		# Aggregates
		if neuron.recorder.get('flux_total_pm', False) and fluxes.total_pm is not None:
			with open(results_path + 'flux_total_pm.txt', 'ab') as f:
				np.savetxt(f, fluxes.total_pm.reshape(1, -1))

		if neuron.recorder.get('flux_total_er', False) and fluxes.total_er is not None:
			with open(results_path + 'flux_total_er.txt', 'ab') as f:
				np.savetxt(f, fluxes.total_er.reshape(1, -1))

		if neuron.recorder.get('flux_total_ip3', False) and fluxes.total_ip3 is not None:
			with open(results_path + 'flux_total_ip3.txt', 'ab') as f:
				np.savetxt(f, fluxes.total_ip3.reshape(1, -1))

		# Node-specific flux recording (more memory efficient than full arrays)
		if neuron.recorder.get('flux_nodes', []):
			for n in neuron.recorder['flux_nodes']:
				filename = results_path + f'flux_node_{n}.txt'

				# Build array of flux values for this node
				flux_values = [t0]  # Start with timestamp

				# Add each enabled flux component for this node
				if neuron.recorder.get('flux_pmca', False) and fluxes.pmca is not None:
					flux_values.append(fluxes.pmca[n])
				if neuron.recorder.get('flux_ncx', False) and fluxes.ncx is not None:
					flux_values.append(fluxes.ncx[n])
				if neuron.recorder.get('flux_pm_leak', False) and fluxes.pm_leak is not None:
					flux_values.append(fluxes.pm_leak[n])
				if neuron.recorder.get('flux_vdcc', False) and fluxes.vdcc is not None:
					flux_values.append(fluxes.vdcc[n])
				if neuron.recorder.get('flux_synapse', False) and fluxes.synapse is not None:
					flux_values.append(fluxes.synapse[n])
				if neuron.recorder.get('flux_synapse_ip3', False) and fluxes.synapse_ip3 is not None:
					flux_values.append(fluxes.synapse_ip3[n])
				if neuron.recorder.get('flux_ryr', False) and fluxes.ryr is not None:
					flux_values.append(fluxes.ryr[n])
				if neuron.recorder.get('flux_serca', False) and fluxes.serca is not None:
					flux_values.append(fluxes.serca[n])
				if neuron.recorder.get('flux_er_leak', False) and fluxes.er_leak is not None:
					flux_values.append(fluxes.er_leak[n])
				if neuron.recorder.get('flux_ip3r', False) and fluxes.ip3r is not None:
					flux_values.append(fluxes.ip3r[n])
				if neuron.recorder.get('flux_soc', False) and fluxes.soc is not None:
					flux_values.append(fluxes.soc[n])
				if neuron.recorder.get('flux_total_pm', False) and fluxes.total_pm is not None:
					flux_values.append(fluxes.total_pm[n])
				if neuron.recorder.get('flux_total_er', False) and fluxes.total_er is not None:
					flux_values.append(fluxes.total_er[n])
				if neuron.recorder.get('flux_total_ip3', False) and fluxes.total_ip3 is not None:
					flux_values.append(fluxes.total_ip3[n])

				# Write to file
				with open(filename, 'ab') as f:
					np.savetxt(f, np.array(flux_values).reshape(1, -1))

###########################################################