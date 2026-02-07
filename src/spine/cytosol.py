# Cytosol module
import numpy as np
from typing import Optional

from spine.vdcc import GHK
from spine.flux_components import FluxComponents

###########################################################

# Dict for storing exchange mechanism parameters
params = {
	# PMCA pumps
	'Ip' : 1.7e-17, # umol / sec
	'Kp' : 60.e-18, # umol / um^3
	'rhop' : 500., # um^{-2}

	# NCX exchangers
	'In' : 2.5e-15, # umol / sec
	'Kn' : 1.8e-15, # umol / um^3
	'rhon' : 15., # um^{-2}

	# VDCC
	'rhovdcc' : 1., # um^{-2}
}

###########################################################

# Global variables
constants = {
	'co': 1.0e-12, # mu mol / mu m^3
	'btot': 4.*40e-15, # mu mol / mu m^3
	'kbpos': 27e15, # mu m^3 / (mu mol x sec)
	'kbneg': 19., # sec^{-1}
	'leak_const': None,
	'vlp': None
}

###########################################################

def reaction(neuron,c,b):
	# Calcium-calbindin reaction term

	f = neuron.cytConstants['kbneg']*(neuron.cytConstants['btot'] - b) - neuron.cytConstants['kbpos']*b*c

	return f

###########################################################

def init_calb(neuron, cceq):
	# Initialize calbindin concentration

	return neuron.cytConstants['kbneg']*neuron.cytConstants['btot']/(neuron.cytConstants['kbneg'] + neuron.cytConstants['kbpos']*cceq)

###########################################################

def init_vel(neuron,c,V,G):
	# Initialize leakage velocity

	if neuron.cytConstants['leak_const'] is None:
		numerator = np.zeros_like(c)
		if neuron.model['PMCA']: numerator += jP(neuron,c)
		if neuron.model['NCX']: numerator += jN(neuron,c)
		if neuron.model['VDCC']: numerator -= jVDCC(neuron,c,V,G)

		neuron.cytConstants['vlp'] = numerator/(neuron.cytConstants['co'] - neuron.settings.cceq)
	else:
		neuron.cytConstants['vlp'] = neuron.cytConstants['leak_const']


###########################################################

def jP(neuron,c):
	# PMCA pumps

	v = neuron.cytExchangeParams['rhop']*neuron.cytExchangeParams['Ip']*(c**2)/(neuron.cytExchangeParams['Kp']*neuron.cytExchangeParams['Kp'] + (c**2))

	return v

###########################################################

def jN(neuron,c):
	# NCX exchangers

	v = neuron.cytExchangeParams['rhon']*neuron.cytExchangeParams['In']*c/(neuron.cytExchangeParams['Kn'] + c)

	return v

###########################################################

def jLP(neuron,c, V):
	# Leakage flux

	v = neuron.cytConstants['vlp']*(neuron.cytConstants['co']-c)

	return v

###########################################################

def jVDCC(neuron,c,V,G):

	F = GHK(neuron,V,c)
	v = neuron.cytExchangeParams['rhovdcc']*G*F

	return v

###########################################################

def jSYN(neuron,c,p,t,V, comm, comm_iter):
	"""Compute synaptic flux from all synapse instances.

	Supports both legacy synapse class and new Synapse architecture.

	Args:
		neuron: NeuronModel instance
		c: Cytosolic calcium (umol/um^3)
		p: IP3 concentration (umol/um^3)
		t: Current time (s)
		V: Voltage (V)
		comm: SynapseCommunicator instance
		comm_iter: Neuron index in communicator

	Returns:
		v: Calcium flux (umol/(um^2*s))
		vip3: IP3 flux (umol/(um^2*s)) if IP3 model enabled
	"""
	v = np.zeros_like(c)
	if neuron.model['IP3']: vip3 = np.zeros_like(c)

	for syn in neuron.synapse_instances:
		# Check if this is new Synapse class (has compute_current method)
		if hasattr(syn, 'compute_current'):
			# Voltage-domain synapses (AMPA, NMDA, GABA) are handled
			# entirely in current_model.py via the cable equation.
			if getattr(syn, 'domain', None) == 'voltage':
				continue

			# New architecture
			flux = syn.compute_current(t, neuron, comm, comm_iter)

			# Route to correct output based on domain
			if syn.domain == 'cytosol':
				v += flux
			elif syn.domain == 'ip3' and neuron.model['IP3']:
				vip3 += flux
			elif syn.domain == 'receptor':
				v += flux

			# Update synapse state if needed
			if hasattr(syn, 'update_state'):
				syn.update_state(neuron.settings.dt, neuron, comm, comm_iter)

		else:
			# Legacy synapse class (backward compatibility)
			if syn.domain == 'cytosol':
				for node in syn.node:
					reset = syn.release_pattern(t=t, node=node)
					v[node] += syn.release_profile(node=node, c=c[node], V=V[node])
				if not reset: syn.t += neuron.settings.dt
			elif syn.domain == 'ip3' and neuron.model['IP3']:
				for node in syn.node:
					reset = syn.release_pattern(t=t, node=node)
					vip3[node] += syn.release_profile(node=node, p=p[node], V=V[node])
				if not reset: syn.t += neuron.settings.dt
			elif syn.domain == 'receptor':
				for node in syn.node:
					reset = syn.release_pattern(t=t, node=node)
					v[node] += syn.release_profile(node=node, c=c[node], V=V[node], neuron=neuron, comm=comm, comm_iter=comm_iter)

	if neuron.model['IP3']: return v, vip3
	else: return v


###########################################################

def JPM(neuron, c, p, t, V, G, comm, comm_iter, fluxes: Optional[FluxComponents] = None):
	"""Compute total plasma membrane calcium flux.

	Args:
		neuron: NeuronModel instance
		c: Cytosolic calcium concentration (umol/um^3)
		p: IP3 concentration (umol/um^3)
		t: Current time (s)
		V: Voltage (V)
		G: VDCC gating variable
		comm: Synapse communicator
		comm_iter: Communicator iteration index
		fluxes: Optional FluxComponents object for recording individual fluxes

	Returns:
		b: Total PM flux (umol/(um^2 * s))
		jip3: IP3 flux if IP3 model enabled (umol/(um^2 * s))
	"""
	b = np.zeros_like(c)

	# PMCA pumps (calcium extrusion)
	if neuron.model['PMCA']:
		jP_val = jP(neuron, c)
		if fluxes is not None and neuron.recorder['flux_pmca']:
			fluxes.pmca = jP_val.copy()
		b -= jP_val

	# NCX exchangers (calcium extrusion)
	if neuron.model['NCX']:
		jN_val = jN(neuron, c)
		if fluxes is not None and neuron.recorder['flux_ncx']:
			fluxes.ncx = jN_val.copy()
		b -= jN_val

	# PM leak (calcium influx)
	if neuron.model['PM_leak']:
		jLP_val = jLP(neuron, c, V)
		if fluxes is not None and neuron.recorder['flux_pm_leak']:
			fluxes.pm_leak = jLP_val.copy()
		b += jLP_val

	# VDCC (calcium influx)
	if neuron.model['VDCC']:
		jVDCC_val = jVDCC(neuron, c, V, G)
		if fluxes is not None and neuron.recorder['flux_vdcc']:
			fluxes.vdcc = jVDCC_val.copy()
		b += jVDCC_val

	# Synaptic flux (calcium and/or IP3 influx)
	if neuron.model['synapse']:
		if neuron.model['IP3']:
			jsyn, jip3 = jSYN(neuron, c, p, t, V, comm, comm_iter)
			if fluxes is not None and neuron.recorder['flux_synapse_ip3']:
				fluxes.synapse_ip3 = jip3.copy()
		else:
			jsyn = jSYN(neuron, c, p, t, V, comm, comm_iter)

		if fluxes is not None and neuron.recorder['flux_synapse']:
			fluxes.synapse = jsyn.copy()
		b += jsyn

	if not neuron.model['synapse'] and neuron.model['IP3']:
		jip3 = np.zeros_like(c)

	# No flux terms in axons (this causes blowup in axons...why?)
	# Use pre-computed axon indices for performance
	if hasattr(neuron, '_axon_idx'):
		b[neuron._axon_idx] = 0.
	else:
		b[np.where(neuron.graph.ntype == 2)[0]] = 0.

	# Store total PM flux if recording
	if fluxes is not None:
		fluxes.total_pm = b.copy()

	if neuron.model['IP3']:
		return b, jip3
	else:
		return b



