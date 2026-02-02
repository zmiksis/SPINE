# Vrest set in neuron.settings
import numpy as np
import copy

currents = {
    'I_Na': False,
    'I_K': False,
    'I_Ca': False,
    'I_leak': False,
    'I_syn': False
}

# This class is under development and may be subject to 
# changes in future releases.
class IonChannel():
    def __init__(self, neuron, name, g, E, 
                    Vt=None, 
                    state_vars=None, 
                    state_exp=None,
                    flux=None):

        self.neuron = neuron
        self.Cm = neuron.settings.Cm
        self.Rm = neuron.settings.Rm

        self.name = name
        self.g = g  # Maximum conductance (S / um^2)
        self.E = E  # Reversal potential (V)
        self.Vt = Vt  # Threshold potential (V), if applicable 

        if state_vars is not None:
            self.state_vars = state_vars  # List of state variable names
        else:
            self.state_vars = []
        if state_exp is not None:
            self.state_exp = state_exp  # List of state variable exponents
        else:
            self.state_exp = []

        self._flux = flux or self.default_flux

    def flux(self):
        return self._flux(self)
    
    def default_flux(self, V):  
        V = 1e-3*self.neuron.sol.V # convert to V
        flux = (self.g / self.Cm) * (V - self.E)
        for var, exp in zip(self.state_vars, self.state_exp):
            flux *= var**exp
        return flux

class CableSettings():
    def __init__(self, neuron):

        self.neuron = neuron
        self.Cm = neuron.settings.Cm
        self.Rm = neuron.settings.Rm
        self.Ext = 0.0  # External potential (V)

        # NEW: Extensible channel list (factory pattern)
        self.channels = []  # List of ChannelModel instances

        # LEGACY: Old flag-based system (for backward compatibility)
        # Current parameters
        self.Na = IonChannel(self.neuron, 'Na', 500.0e-12, 50.0e-3, -60.0e-3)
        self.m = None
        self.h = None
        self.m0 = None
        self.h0 = None

        self.K = IonChannel(self.neuron, 'K', 50.0e-12, -90.0e-3, -60.0e-3)
        self.n = None
        self.n0 = None

        self.g_Ca = 5.16*6e-12      # S / um^2
        self.E_Ca = None        # V
        self.K_Ca = 0.01e3        # uM
        self.q = None
        self.q0 = None

        self.leak = IonChannel(self.neuron, 'leak', 0.05e-12, neuron.settings.Vrest)

        self.g_syn = 2.7e-10  # S / um^2
        self.E_syn = 0.0   # V
        self.s = None
        self.s0 = None

        self.profile = None
        self.profile0 = None

    def add_channel(self, channel):
        """Add an ion channel to the neuron (NEW factory pattern API).

        Args:
            channel: ChannelModel instance from ChannelFactory

        Example:
            from spine.cable_eq import ChannelFactory
            neuron.CableSettings.add_channel(ChannelFactory.create_Na())
            neuron.CableSettings.add_channel(ChannelFactory.create_K())
        """
        self.channels.append(channel)

    def initialize_channels(self):
        """Initialize gating variables for all channels in channel list.

        Called automatically by SBDF solver if channels list is non-empty.
        """
        if not self.channels:
            return

        V_rest = self.neuron.settings.Vrest
        nnodes = self.neuron.sol.C.shape[0]

        for channel in self.channels:
            channel.initialize_state(V_rest, nnodes)

        print('Channel gating variables initialized!')

    def set_leak_reversal(self):
        """Compute balancing current to maintain resting potential.

        Supports both NEW channel list and LEGACY flag-based systems.
        """
        I_total = 0.0

        # NEW: Use channel list if available
        if self.channels:
            V = 1e-3 * self.neuron.sol.V  # Convert mV → V
            C = 1e-15 * self.neuron.sol.C  # Convert μM → umol/um³

            for channel in self.channels:
                if channel.contributes_to_leak_balance():
                    I_total += channel.compute_current(V, C, self.Cm)

        # LEGACY: Fall back to flag-based system
        else:
            if self.neuron.currents['I_leak']:
                I_total += self.I_leak()

            if self.neuron.currents['I_K']:
                I_total += self.I_K()

            if self.neuron.currents['I_Na']:
                I_total += self.I_Na()

            if self.neuron.currents['I_Ca']:
                I_total += self.I_Ca()

        self.Ext = I_total
        
    def I_leak(self):

        V = 1e-3*self.neuron.sol.V # convert to V
        return (self.leak.g / self.Cm) * (V - self.leak.E) 
    
    def I_K(self):

        V = 1e-3*self.neuron.sol.V # convert to V
        return (self.K.g / self.Cm) * self.n**4 * (V - self.K.E)

    def I_Na(self):

        V = 1e-3*self.neuron.sol.V # convert to V
        return (self.Na.g / self.Cm) * self.m**3 * self.h * (V - self.Na.E)
    
    def I_Ca(self):

        V = 1e-3*self.neuron.sol.V # convert to V

        co = self.neuron.cytConstants['co']
        E = 12.5e-3 * np.log(co / 1e-15*self.neuron.sol.C)   # V
        rho = self.K_Ca / (self.K_Ca + self.neuron.sol.C)  # dimensionless

        return (self.g_Ca / self.Cm) * self.q * rho * (V - E)

    def I_syn(self, comm, comm_iter):

        _, idx = comm.get_pre_V(comm_iter)
        V = np.zeros_like(self.neuron.sol.V)
        V[idx] = 1e-3*self.neuron.sol.V[idx] # convert to V

        weight = np.zeros_like(V)

        for neuron in comm.weight_map[f'{comm_iter}']:
            for idx, pair in enumerate(comm.pre_post_node_map[f'{comm_iter}'][neuron]):
                weight[pair[1]] = comm.weight_map[f'{comm_iter}'][neuron][idx]

        return weight * (self.g_syn / self.Cm) * self.s * (V - self.E_syn)

    def current_profile(self, comm, comm_iter):
        """Compute total current profile from all channels.

        Supports both NEW channel list and LEGACY flag-based systems.
        """
        I_total = 0.0

        # NEW: Use channel list if available
        if self.channels:
            V = 1e-3 * self.neuron.sol.V  # Convert mV → V
            C = 1e-15 * self.neuron.sol.C  # Convert μM → umol/um³

            for channel in self.channels:
                I_total -= channel.compute_current(V, C, self.Cm)

            # Add synaptic current if needed (still uses legacy system)
            if self.neuron.currents.get('I_syn', False):
                I_total -= self.I_syn(comm, comm_iter)

        # LEGACY: Fall back to flag-based system
        else:
            if self.neuron.currents['I_leak']:
                I_total -= self.I_leak()

            if self.neuron.currents['I_K']:
                I_total -= self.I_K()

            if self.neuron.currents['I_Na']:
                I_total -= self.I_Na()

            if self.neuron.currents['I_Ca']:
                I_total -= self.I_Ca()

            if self.neuron.currents['I_syn']:
                I_total -= self.I_syn(comm, comm_iter)

        self.profile = I_total + self.Ext

    def n_state(self, n, V):
        Vt = 1e3*self.K.Vt  # mV
        an = 1.0e3 * -0.032 * (V - Vt - 15.) / (np.exp(-1.0 * (V - Vt - 15.)/5.) - 1.)
        bn = 1.0e3 * 0.5 * np.exp(-1.0 * (V - Vt - 10.)/40.)
        return an * (1. - n) - bn * n
    
    def m_state(self, m, V):
        Vt = 1e3*self.Na.Vt  # mV
        am = -1.0e3 * 0.32 * (V - Vt - 13.) / (np.exp(-1.0 * (V - Vt - 13.)/4.) - 1.)
        bm = 1.0e3 * 0.28 * (V - Vt - 40.) / (np.exp((V - Vt - 40.)/5.) - 1.)
        return am * (1. - m) - bm * m
    
    def h_state(self, h, V):
        Vt = 1e3*self.Na.Vt  # mV
        ah = 1.0e3 *0.128 * np.exp(-1.0 * (V - Vt - 17.)/18.)
        bh = 1.0e3 * 4.0 / (1. + np.exp(-1.0 * (V - Vt - 40.)/5.))
        return ah * (1. - h) - bh * h
    
    def q_state(self, q, V):
        tau = 7.8 / (np.exp((V + 6.)/16.) + np.exp(-1.0 * (V + 6.)/16.))
        sigma = 1.0 / (1.0 + np.exp(-(V - 3)/8.))
        return (sigma - q) / tau

    def s_state(self, s, V):
        alpha = 0.55
        beta = 4.0
        tau = 5.26e-3  # seconds
        return alpha * (1. + np.tanh(V / beta)) * (1. - s) - s / tau
    
    def set_gating_variables(self):

        Vrest = 1e3*self.neuron.settings.Vrest

        if self.neuron.currents['I_K']:
            n0 = 1.
            n = 0.5
            while abs(n - n0) > 1e-6:
                tmp = copy.deepcopy(n)
                n = n - self.n_state(n, Vrest) * (n - n0) / (self.n_state(n, Vrest) - self.n_state(n0, Vrest))
                n = np.clip(n, 0.0, 1.0)
                n0 = copy.deepcopy(tmp)
            self.n = n*np.ones_like(self.neuron.sol.C)      
            self.n0 = n*np.ones_like(self.neuron.sol.C) 

        if self.neuron.currents['I_Na']:
            m0 = 1.
            m = 0.5
            while abs(m - m0) > 1e-6:
                tmp = copy.deepcopy(m)
                m = m - self.m_state(m, Vrest) * (m - m0) / (self.m_state(m, Vrest) - self.m_state(m0, Vrest))
                m = np.clip(m, 0.0, 1.0)
                m0 = copy.deepcopy(tmp)
            self.m = m*np.ones_like(self.neuron.sol.C)      
            self.m0 = m*np.ones_like(self.neuron.sol.C) 

            h0 = 1.
            h = 0.5
            while abs(h - h0) > 1e-6:
                tmp = copy.deepcopy(h)
                h = h - self.h_state(h, Vrest) * (h - h0) / (self.h_state(h, Vrest) - self.h_state(h0, Vrest))
                h = np.clip(h, 0.0, 1.0)
                h0 = copy.deepcopy(tmp)
            self.h = h*np.ones_like(self.neuron.sol.C)      
            self.h0 = h*np.ones_like(self.neuron.sol.C) 

        if self.neuron.currents['I_Ca']:
            q0 = 1.
            q = 0.5
            while abs(q - q0) > 1e-6:
                tmp = copy.deepcopy(q)
                q = q - self.q_state(q, Vrest) * (q - q0) / (self.q_state(q, Vrest) - self.q_state(q0, Vrest))
                q = np.clip(q, 0.0, 1.0)
                q0 = copy.deepcopy(tmp)
            self.q = q*np.ones_like(self.neuron.sol.C)      
            self.q0 = q*np.ones_like(self.neuron.sol.C) 

        if self.neuron.currents['I_syn']:
            s0 = 1.
            s = 0.5
            while abs(s - s0) > 1e-6:
                tmp = copy.deepcopy(s)
                s = s - self.s_state(s, Vrest) * (s - s0) / (self.s_state(s, Vrest) - self.s_state(s0, Vrest))
                s = np.clip(s, 0.0, 1.0)
                s0 = copy.deepcopy(tmp)
            self.s = s*np.ones_like(self.neuron.sol.C)      
            self.s0 = s*np.ones_like(self.neuron.sol.C) 
        
        print('Gating variables converged!')

    def update_gating_variables(self, comm, comm_iter):
        """Update gating variables for all channels.

        Supports both NEW channel list and LEGACY flag-based systems.
        Note: gating variable updates use mV and s units.
        """
        V = self.neuron.sol.V    # mV
        V0 = self.neuron.sol.V0  # mV
        dt = self.neuron.settings.dt

        # NEW: Use channel list if available
        if self.channels:
            for channel in self.channels:
                channel.update_state(V, V0, dt)

            # Update synaptic gating if needed (still uses legacy system)
            if self.neuron.currents.get('I_syn', False):
                V_syn = np.zeros_like(V)
                v, pre_nodes = comm.get_pre_V(comm_iter)
                V_syn[pre_nodes] = np.array(v)

                V0_syn = np.zeros_like(V0)
                v, pre_nodes = comm.get_pre_V0(comm_iter)
                V0_syn[pre_nodes] = np.array(v)

                state = self.s_state(self.s, V_syn)
                state0 = self.s_state(self.s0, V0_syn)
                tmp = self.s.copy()
                self.s = (4./3.)*self.s + (4./3.)*state*dt - (1./3.)*self.s0 - (2./3.)*state0*dt
                self.s0 = tmp

        # LEGACY: Fall back to flag-based system
        else:
            if self.neuron.currents['I_K']:
                # Update n
                state = self.n_state(self.n, V)
                state0 = self.n_state(self.n0, V0)
                tmp = self.n.copy()
                self.n = (4./3.)*self.n + (4./3.)*state*dt - (1./3.)*self.n0 - (2./3.)*state0*dt
                self.n0 = tmp

            if self.neuron.currents['I_Na']:
                # Update m
                state = self.m_state(self.m, V)
                state0 = self.m_state(self.m0, V0)
                tmp = self.m.copy()
                self.m = (4./3.)*self.m + (4./3.)*state*dt - (1./3.)*self.m0 - (2./3.)*state0*dt
                self.m0 = tmp

                # Update h
                state = self.h_state(self.h, V)
                state0 = self.h_state(self.h0, V0)
                tmp = self.h.copy()
                self.h = (4./3.)*self.h + (4./3.)*state*dt - (1./3.)*self.h0 - (2./3.)*state0*dt
                self.h0 = tmp

            if self.neuron.currents['I_Ca']:
                # Update q
                state = self.q_state(self.q, V)
                state0 = self.q_state(self.q0, V0)
                tmp = self.q.copy()
                self.q = (4./3.)*self.q + (4./3.)*state*dt - (1./3.)*self.q0 - (2./3.)*state0*dt
                self.q0 = tmp

            if self.neuron.currents['I_syn']:
                V_syn = np.zeros_like(V)
                v, pre_nodes = comm.get_pre_V(comm_iter)
                V_syn[pre_nodes] = np.array(v)

                V0_syn = np.zeros_like(V0)
                v, pre_nodes = comm.get_pre_V0(comm_iter)
                V0_syn[pre_nodes] = np.array(v)

                state = self.s_state(self.s, V_syn)
                state0 = self.s_state(self.s0, V0_syn)
                tmp = self.s.copy()
                self.s = (4./3.)*self.s + (4./3.)*state*dt - (1./3.)*self.s0 - (2./3.)*state0*dt
                self.s0 = tmp