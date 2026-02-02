# Endoplasmic reticulum module
import numpy as np
import copy
from typing import Optional

from spine.flux_components import FluxComponents

###########################################################

# Dict for storing exchange mechanism parameters
params = {
	# RyR
	'rhoryr' : 3., # um^{-2}
	'Irefryr' : 3.5e-12, # umol / s

	# SERCA
	'Is' : 6.5e-30, # umol^2 / um^3 sec
	'Ks' : 180.e-18, # umol / um^3
	'rhos' : 2390., # um^{-2}

	# IP3
	'kp' : 1.e3, # s^{-1}
	'd1' : 0.13e-15, # umol / um^3
	'd2' : 1.05e-15, # umol / um^3
	'd3' : 0.94e-15, # umol / um^3
	'd5' : 82.3e-18, # umol / um^3
	'rhoI' : 17.3, # um^{-2}
	'IrefI' : 1.1e-13 # umol / s
}

###########################################################

constants = {
	'betot' : 4.*3.6e-12, # umol/um^3
	'kbepos' : 1.e14, # um^3 / (umol x sec)
	'kbeneg' : 200., # s^{-1}
	'leak_const' : None,
	'vle' : None
}

###########################################################

def reaction_er(neuron, ce, be):
	# Calcium-calbindin reaction term

	f = neuron.erConstants['kbeneg']*(neuron.erConstants['betot'] - be) - neuron.erConstants['kbepos']*be*ce

	return f

###########################################################

def reaction_ip3(neuron, p):

	f = (-1.)*neuron.erExchangeParams['kp']*(p - neuron.settings.pr)

	return f

###########################################################

def init_calbe(neuron, ceeq):
	# Initialize calreticulin concentration

	return neuron.erConstants['kbeneg']*neuron.erConstants['betot']/(neuron.erConstants['kbeneg'] + neuron.erConstants['kbepos']*ceeq)

###########################################################

def init_ryr_state(neuron, c):

	cceq = neuron.settings.cceq # umol/um^3
	kaneg = 28.8 # sec^{-1}
	kapos = 1500.e60 # um^12/(umol^4 x s) 
	kbneg = 385.9 # sec^{-1}
	kbpos = 1500.e45 # um^9/(umol^3 x s)
	kcneg = 0.1 # sec^{-1}
	kcpos = 1.75 # sec^{-1}

	A = np.array([[1., 1., 1., 1.], \
		[kaneg, 0., -kapos*(cceq**4), 0.], \
		[kbpos*(cceq**3), -kbneg, 0., 0.], \
		[kcpos, 0., 0., -kcneg]])
	B = np.array([1., 0., 0., 0.])
	RyRProb = np.linalg.solve(A,B)

	o1 = np.ones_like(c)*RyRProb[0] 
	o2 = np.ones_like(c)*RyRProb[1]
	c1 = np.ones_like(c)*RyRProb[2]
	c2 = 1. - o1 - o2 - c1

	return np.array([o1, o2, c1, c2]) 

###########################################################

def init_vel_e(neuron, c, ce, p, ryr_state, dt):

	if neuron.erConstants['leak_const'] is None:
		numerator = np.zeros_like(c)
		if neuron.model['RyR']: numerator -= jRYR(neuron, c, ce, ryr_state, dt)
		if neuron.model['SERCA']: numerator += jS(neuron, c, ce)
		if neuron.model['IP3']: numerator -= jIP3(neuron, p, c, ce)

		neuron.erConstants['vle'] = numerator/(neuron.settings.ceeq - neuron.settings.cceq)
	else:
		neuron.erConstants['vle'] = neuron.erConstants['leak_const']

###########################################################

def update_ryr_state(ryr_state, ctemp, dt):

	c = ctemp
	# Use shallow copy for 1D arrays (10-100x faster than deepcopy)
	o1 = ryr_state[0,:].copy()
	o2 = ryr_state[1,:].copy()
	c1 = ryr_state[2,:].copy()
	c2 = ryr_state[3,:].copy()

	kaneg = 28.8 # sec^{-1}
	kapos = 1500.e60 # m^12/(mol^4 x s) 
	kbneg = 385.9 # sec^{-1}
	kbpos = 1500.e45 # m^9/(mol^3 x s)
	kcneg = 0.1 # sec^{-1}
	kcpos = 1.75 # sec^{-1}

	a11 = 1. + kbneg*dt + kbpos*(c**3)*dt
	a12 = kbpos*(c**3)*dt
	a13 = kbpos*(c**3)*dt
	a21 = kaneg*dt
	a22 = 1. + kaneg*dt + kapos*(c**4)*dt
	a23 = kaneg*dt
	a31 = kcpos*dt
	a32 = kcpos*dt
	a33 = 1. + kcpos*dt + kcneg*dt

	M11 = a22*a33 - a23*a32
	M12 = np.ones_like(c, dtype='float64')*(a21*a33 - a23*a31)
	M13 = a21*a32 - a22*a31  
	M21 = a12*a33 - a13*a32  
	M22 = a11*a33 - a13*a31  
	M23 = a11*a32 - a12*a31
	M31 = a12*a23 - a13*a22 
	M32 = a11*a23 - a13*a21  
	M33 = a11*a22 - a12*a21

	C = np.array([-kbpos*dt*(c**3), \
		 np.ones_like(c, dtype='float64')*(-kaneg*dt), \
		 np.ones_like(c, dtype='float64')*(-kcpos*dt)], dtype='float64')
	detA = M11*a11 - M12*a12 + M13*a13
	b = np.array([o2, c1, c2], dtype='float64')
	bc = b - C
	bc1 = bc[0,:]
	bc2 = bc[1,:]
	bc3 = bc[2,:]

	o2 = (M11/detA)*(bc1) - (M21/detA)*(bc2) + (M31/detA)*(bc3)
	c1 = -(M12/detA)*(bc1) + (M22/detA)*(bc2) - (M32/detA)*(bc3)
	c2 = (M13/detA)*(bc1) - (M23/detA)*(bc2) + (M33/detA)*(bc3)

	o1 = 1. - o2 - c1 - c2

	return np.array([o1, o2, c1, c2]) 


###########################################################

def jRYR(neuron, c, ce, ryr_state, dt):

	Iryr = neuron.erExchangeParams['Irefryr']*(ce - c)/neuron.settings.ceeq

	o1 = ryr_state[0,:]
	o2 = ryr_state[1,:]
	rhoOryr = o1 + o2

	v = neuron.erExchangeParams['rhoryr'] * rhoOryr * Iryr

	return v

###########################################################

def jIP3(neuron, p, c, ce):

	II = neuron.erExchangeParams['IrefI']*(ce - c)/neuron.settings.ceeq

	# Combine divisions to reduce temporary array creation
	numerator = neuron.erExchangeParams['d2']*c*p
	denominator = (c*p + neuron.erExchangeParams['d2']*p + neuron.erExchangeParams['d3']*c +
	               neuron.erExchangeParams['d1']*neuron.erExchangeParams['d2']) * (c + neuron.erExchangeParams['d5'])
	pO = (numerator / denominator) ** 3

	v = neuron.erExchangeParams['rhoI'] * pO * II

	return v

###########################################################

def jS(neuron, c, ce):

	v = neuron.erExchangeParams['rhos']*neuron.erExchangeParams['Is']*c/((neuron.erExchangeParams['Ks'] + c)*ce)

	return v

###########################################################

def jSOC(neuron, ce):

	ccutoff = 247.602e-15 # umol / um^3
	rhoOsoc = np.ones_like(ce, dtype = 'float64')
	# Vectorized conditional update instead of loop
	mask = ce > ccutoff
	rhoOsoc[mask] = (np.exp(1e15*(neuron.settings.ceeq - ce[mask])) - 1.)/10.

	Irefsoc = 2.1e-15 # Ampere
	Far = 96485.e-6 # Faraday's constant (C/mu mol)
	z = 2. # valence

	Ao = 0.25e-6 # Area of ORAI channel -- for testing SOC response.

	Isoc = np.log(10.*neuron.cytConstants['co']/ce)*Irefsoc/(Far*z*Ao)
	rhosoc = 0.4

	v = rhosoc * rhoOsoc * Isoc

	R = neuron.graph.radius.astype('float64')
	v *= R/0.4

	return v

###########################################################

def jLE(neuron, c, ce, V):

	v = neuron.erConstants['vle'] * (ce - c)

	return v

###########################################################

def JER(neuron, c, ce, p, ryr_state, V, dt, fluxes: Optional[FluxComponents] = None):
	"""Compute total ER calcium flux.

	Args:
		neuron: NeuronModel instance
		c: Cytosolic calcium concentration (umol/um^3)
		ce: ER calcium concentration (umol/um^3)
		p: IP3 concentration (umol/um^3)
		ryr_state: RyR state variables
		V: Voltage (V)
		dt: Time step (s)
		fluxes: Optional FluxComponents object for recording individual fluxes

	Returns:
		b: Total ER flux (umol/(um^2 * s))
		jsoc: SOC flux (umol/(um^2 * s))
	"""
	b = np.zeros_like(c)

	# RyR (ER calcium release)
	if neuron.model['RyR']:
		jryr = jRYR(neuron, c, ce, ryr_state, dt)
		if fluxes is not None and neuron.recorder['flux_ryr']:
			fluxes.ryr = jryr.copy()
		b += jryr

	# SERCA pumps (ER calcium uptake)
	if neuron.model['SERCA']:
		jS_val = jS(neuron, c, ce)
		if fluxes is not None and neuron.recorder['flux_serca']:
			fluxes.serca = jS_val.copy()
		b -= jS_val

	# ER leak
	if neuron.model['ER_leak']:
		jLE_val = jLE(neuron, c, ce, V)
		if fluxes is not None and neuron.recorder['flux_er_leak']:
			fluxes.er_leak = jLE_val.copy()
		b += jLE_val

	# IP3 receptors (ER calcium release)
	if neuron.model['IP3']:
		jIP3_val = jIP3(neuron, p, c, ce)
		if fluxes is not None and neuron.recorder['flux_ip3r']:
			fluxes.ip3r = jIP3_val.copy()
		b += jIP3_val

	# SOC (store-operated channels)
	if neuron.model['SOC']:
		jsoc = jSOC(neuron, ce)
		if fluxes is not None and neuron.recorder['flux_soc']:
			fluxes.soc = jsoc.copy()
	else:
		jsoc = np.zeros_like(ce)

	# No flux terms in axons
	# Use pre-computed axon indices for performance
	if hasattr(neuron, '_axon_idx'):
		b[neuron._axon_idx] = 0.
		jsoc[neuron._axon_idx] = 0.
	else:
		b[np.where(neuron.graph.ntype == 2)[0]] = 0.
		jsoc[np.where(neuron.graph.ntype == 2)[0]] = 0.

	# Store total ER flux if recording
	if fluxes is not None:
		fluxes.total_er = b.copy()

	return b, jsoc



