import numpy as np

###########################################################

# Global constants
Rgas = 8.314e-6
Far = 9.6485e-2
T = 310.

###########################################################

def GHK(neuron,V,cc):
	# Goldman-Hodgkin-Katz equation

    if neuron.model['VDCC_type'] == 'N': pca = 3.8e1
    if neuron.model['VDCC_type'] == 'T': pca = 1.9e1
    if neuron.model['VDCC_type'] == 'L': pca = 5.7e1
    
    z = 2.     # valence
    
    c1 = z*Far*V/(Rgas*T)
    F = pca*c1*(cc - neuron.cytConstants['co']*np.exp(-c1))/(1. - np.exp(-c1))
    
    ind = (abs(V)<=1e-8)
    F[ind] = pca*((neuron.cytConstants['co'] - cc[ind]) - (Far/(Rgas*T))*(neuron.cytConstants['co'] + cc[ind])*V[ind])

    return F

###########################################################

def volt_gate(neuron,V,dt,vdcc_states):
    # Gating function

    sK = vdcc_states[0]
    sL = vdcc_states[1]

    num_steps = 25
    k = dt/num_steps

    aK, bK = ratesK(neuron,V)
    aL, bL = ratesL(neuron,V)

    if neuron.model['VDCC_type'] == 'N': tK0, tL0  = 1.7e-3, 70.0e-3 # V
    if neuron.model['VDCC_type'] == 'T': tK0, tL0  = 1.5e-3, 10.0e-3 # V
    if neuron.model['VDCC_type'] == 'L': tK0, tL0  = 1.5e-3, 0. # V

    for _ in range(num_steps):
        vK = f(sK,aK,bK,tK0)
        vL = f(sL,aL,bL,tL0)
        sK += k*vK
        sL += k*vL

    G = sK*(sL**2)
        
    return G, [sK, sL]

###########################################################

def f(s,a,b,tau0):
	# RHS of gating function ODEs

    with np.errstate(divide='ignore', invalid='ignore'):
        xinf = a/(a + b)
        tau_s = 1./(a + b) + tau0
        v = (xinf - s)/tau_s

    if np.isnan(v).any():
        v = np.zeros_like(s)

    return v

###########################################################

def ratesK(neuron,V):
	# Rate functions K

    if neuron.model['VDCC_type'] == 'N':
        K = 1.7e-3 # V
        gm = 0.
        Vs = -21.0e-3 # V
        z = 2.
    if neuron.model['VDCC_type'] == 'T':
        K = 1.5e-3 # V
        gm = 0.
        Vs = -36.0e-3 # V
        z = 2.
    if neuron.model['VDCC_type'] == 'L':
        K = 1.5e-3 # V
        gm = 0.
        Vs = -1.0e-3 # V
        z = 2.
    
    a = K*np.exp(z*gm*(V - Vs)*Far/(Rgas*T))
    b = K*np.exp(-z*(1. - gm)*(V - Vs)*Far/(Rgas*T))

    return a, b

###########################################################

def ratesL(neuron,V):
	# Rate functions L
	
    if neuron.model['VDCC_type'] == 'N':
        K = 70.0e-3 # V
        gm = 0.
        Vs = -40.0e-3 # V
        z = 1.
    if neuron.model['VDCC_type'] == 'T':
        K = 10.0e-3 # V
        gm = 0.
        Vs = -68.0e-3 # V
        z = 1.
    if neuron.model['VDCC_type'] == 'L':
        K = 0. # V
        gm = 0.
        Vs = 0. # V
        z = 0.

    a = K*np.exp(z*gm*(V - Vs)*Far/(Rgas*T))
    b = K*np.exp(-z*(1. - gm)*(V - Vs)*Far/(Rgas*T))

    return a, b

###########################################################

def initialize_vdcc_states(neuron,v):
	# Initialization of VDCC states

    if neuron.model['VDCC_type'] == 'N':
        Va = -21.e-3 # V
        Vb = -40.e-3 # V
        za = 2. 
        zb = 1.
    if neuron.model['VDCC_type'] == 'T':
        Va = -36.e-3 # V
        Vb = -68.e-3 # V
        za = 2. 
        zb = 1.
    if neuron.model['VDCC_type'] == 'L':
        Va = -1.e-3 # V
        Vb = 0. # V
        za = 2. 
        zb = 0.
    
    s1 = 1./(1. + np.exp(-za*(v - Va)*Far/(Rgas*T)))
    s2 = 1./(1. + np.exp(-zb*(v - Vb)*Far/(Rgas*T)))

    return [s1, s2]



