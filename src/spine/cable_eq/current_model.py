import numpy as np
import copy

currents = {
    'I_Na': False,
    'I_K': False,
    'I_Ca': False,
    'I_leak': False,
    'I_syn': False
}

class ChannelParams():
    """Default parameters for an ion channel type.

    Users can modify these before running the solver to customize
    channel behavior when using the flag-based API:
        neuron.CableSettings.Na.g = 800e-12
        neuron.CableSettings.K.E = -85e-3
    """
    def __init__(self, g, E, Vt=None):
        self.g = g    # Maximum conductance (S/um^2)
        self.E = E    # Reversal potential (V)
        self.Vt = Vt  # Threshold potential (V)

class CableSettings():
    def __init__(self, neuron):

        self.neuron = neuron
        self.Cm = neuron.settings.Cm
        self.Rm = neuron.settings.Rm
        self.Ext = 0.0  # Balancing current at rest (V/s)

        # Channel object list (populated by _auto_populate_channels or add_channel)
        self.channels = []

        # Default parameters for flag-based channel creation
        self.Na = ChannelParams(500.0e-12, 50.0e-3, -60.0e-3)
        self.K = ChannelParams(50.0e-12, -90.0e-3, -60.0e-3)
        self.g_Ca = 5.16*6e-12      # S/um^2
        self.K_Ca = 0.01e3          # uM
        self.leak = ChannelParams(0.05e-12, neuron.settings.Vrest)

        # Synaptic current parameters (used by I_syn flag)
        self.g_syn = 2.7e-10  # S/um^2
        self.E_syn = 0.0      # V
        self.s = None
        self.s0 = None

        self.profile = None
        self.profile0 = None

    def _auto_populate_channels(self):
        """Create channel objects from currents flags if none were added explicitly.

        This allows users to simply set flags like neuron.currents['I_Na'] = True
        and have the corresponding channel created automatically with default
        parameters. Users can customize parameters beforehand by modifying
        attributes (e.g. neuron.CableSettings.Na.g = 800e-12).

        If channels were already added via add_channel() / ChannelFactory,
        this method does nothing.
        """
        if self.channels:
            return

        from .channels import SodiumChannel, PotassiumChannel, CalciumChannel, LeakChannel

        if self.neuron.currents.get('I_Na', False):
            self.channels.append(SodiumChannel(
                g_max=self.Na.g, E_rev=self.Na.E, Vt=self.Na.Vt
            ))

        if self.neuron.currents.get('I_K', False):
            self.channels.append(PotassiumChannel(
                g_max=self.K.g, E_rev=self.K.E, Vt=self.K.Vt
            ))

        if self.neuron.currents.get('I_Ca', False):
            self.channels.append(CalciumChannel(
                g_max=self.g_Ca,
                K_Ca=self.K_Ca * 1e-15,
                co=self.neuron.cytConstants['co']
            ))

        if self.neuron.currents.get('I_leak', False):
            self.channels.append(LeakChannel(
                g_max=self.leak.g, E_rev=self.leak.E
            ))

    def add_channel(self, channel):
        """Add an ion channel to the neuron.

        Args:
            channel: ChannelModel instance from ChannelFactory

        Example:
            from spine.cable_eq import ChannelFactory
            neuron.CableSettings.add_channel(ChannelFactory.create_Na())
        """
        self.channels.append(channel)

    def initialize_channels(self):
        """Initialize gating variables for all channels at resting potential."""
        if not self.channels:
            return

        V_rest = self.neuron.settings.Vrest
        nnodes = self.neuron.sol.C.shape[0]

        for channel in self.channels:
            channel.initialize_state(V_rest, nnodes)

    def set_leak_reversal(self):
        """Compute balancing current to maintain resting potential."""
        I_total = 0.0

        V = 1e-3 * self.neuron.sol.V
        C = 1e-15 * self.neuron.sol.C

        for channel in self.channels:
            if channel.contributes_to_leak_balance():
                I_total += channel.compute_current(V, C, self.Cm)

        self.Ext = I_total

    def I_syn(self, comm, comm_iter):

        _, idx = comm.get_pre_V(comm_iter)
        V = np.zeros_like(self.neuron.sol.V)
        V[idx] = 1e-3*self.neuron.sol.V[idx]

        weight = np.zeros_like(V)

        for neuron in comm.weight_map[f'{comm_iter}']:
            for idx, pair in enumerate(comm.pre_post_node_map[f'{comm_iter}'][neuron]):
                weight[pair[1]] = comm.weight_map[f'{comm_iter}'][neuron][idx]

        return weight * (self.g_syn / self.Cm) * self.s * (V - self.E_syn)

    def current_profile(self, comm, comm_iter):
        """Compute total current profile from all sources."""
        I_total = 0.0

        V = 1e-3 * self.neuron.sol.V
        C = 1e-15 * self.neuron.sol.C

        for channel in self.channels:
            I_total -= channel.compute_current(V, C, self.Cm)

        if self.neuron.currents.get('I_syn', False):
            I_total -= self.I_syn(comm, comm_iter)

        # Voltage-domain synapses (AMPA, NMDA, GABA from synapse_instances)
        for syn in self.neuron.synapse_instances:
            if getattr(syn, 'domain', None) == 'voltage':
                I_total -= syn.compute_current(0.0, self.neuron, comm, comm_iter)

        self.profile = I_total + self.Ext

    def s_state(self, s, V):
        alpha = 0.55
        beta = 4.0
        tau = 5.26e-3  # seconds
        return alpha * (1. + np.tanh(V / beta)) * (1. - s) - s / tau

    def set_gating_variables(self):

        # Auto-create channel objects from flags if user didn't add them explicitly
        self._auto_populate_channels()

        # Initialize gating variables for all channel objects
        self.initialize_channels()

        # Synaptic gating is handled separately (not a standard ion channel)
        if self.neuron.currents.get('I_syn', False):
            Vrest = 1e3*self.neuron.settings.Vrest
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
        """Update gating variables for all channels and synapses."""
        V = self.neuron.sol.V    # mV
        V0 = self.neuron.sol.V0  # mV
        dt = self.neuron.settings.dt

        for channel in self.channels:
            channel.update_state(V, V0, dt)

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

        # Voltage-domain synapses (AMPA, NMDA, GABA from synapse_instances)
        for syn in self.neuron.synapse_instances:
            if getattr(syn, 'domain', None) == 'voltage':
                syn.update_state(dt, self.neuron, comm, comm_iter)
