# Extensible Ion Channel Architecture

## Design Philosophy
Mirror the synapse system's factory pattern while preserving the leak balancing mechanism.

## 1. Base Classes (`cable_eq/base.py`)

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional

class ChannelModel(ABC):
    """Abstract base for ion channel models."""

    def __init__(self, name: str, g_max: float, E_rev: float = None):
        """
        Args:
            name: Channel identifier (e.g., 'Na', 'K', 'leak')
            g_max: Maximum conductance (S/um²)
            E_rev: Reversal potential (V), None = computed dynamically
        """
        self.name = name
        self.g_max = g_max
        self.E_rev = E_rev
        self.state = {}  # Dictionary of gating variables

    @abstractmethod
    def compute_current(self, V: np.ndarray, C: np.ndarray, Cm: float) -> np.ndarray:
        """Compute channel current density.

        Args:
            V: Membrane voltage (V)
            C: Cytosolic calcium (umol/um³)
            Cm: Membrane capacitance (F/um²)

        Returns:
            Current density (V/s) for cable equation
        """
        pass

    @abstractmethod
    def update_state(self, V: np.ndarray, V0: np.ndarray, dt: float):
        """Update gating variables using SBDF2."""
        pass

    @abstractmethod
    def initialize_state(self, V_rest: float, nnodes: int):
        """Initialize gating variables at steady state."""
        pass

    def get_reversal_potential(self, V: np.ndarray, C: np.ndarray = None) -> float:
        """Get reversal potential (constant or dynamic).

        Override for channels with dynamic reversal (e.g., Ca via Nernst).
        """
        return self.E_rev

    def contributes_to_leak_balance(self) -> bool:
        """Whether this channel is included in leak balancing calculation.

        Returns True for most channels. Leak channel should return False
        to avoid circular dependency.
        """
        return self.name != 'leak'


class GatingVariable:
    """Helper class for managing individual gating variables."""

    def __init__(self, name: str, exponent: int = 1):
        """
        Args:
            name: Variable name (e.g., 'n', 'm', 'h')
            exponent: Power in conductance formula (e.g., n^4 has exponent=4)
        """
        self.name = name
        self.exponent = exponent
        self.current = None  # Current value
        self.history = None  # Previous timestep value

    def initialize(self, value: float, nnodes: int):
        """Initialize at steady state."""
        self.current = value * np.ones(nnodes)
        self.history = value * np.ones(nnodes)

    def update_sbdf2(self, kinetics_fn, V: np.ndarray, V0: np.ndarray, dt: float):
        """Update using second-order backward differentiation.

        Args:
            kinetics_fn: Function(state, V) -> dstate/dt
            V, V0: Current and previous voltage (mV)
            dt: Time step (s)
        """
        dstate = kinetics_fn(self.current, V)
        dstate0 = kinetics_fn(self.history, V0)

        tmp = self.current.copy()
        self.current = (4./3.)*self.current + (4./3.)*dstate*dt \
                     - (1./3.)*self.history - (2./3.)*dstate0*dt
        self.current = np.clip(self.current, 0, 1)
        self.history = tmp
```

## 2. Specific Channels (`cable_eq/channels.py`)

```python
import numpy as np
from .base import ChannelModel, GatingVariable

class SodiumChannel(ChannelModel):
    """Fast sodium channel (Hodgkin-Huxley m³h model)."""

    def __init__(self, g_max=500e-12, E_rev=50e-3, Vt=-60e-3):
        super().__init__('Na', g_max, E_rev)
        self.Vt = Vt  # Threshold for kinetics
        self.m = GatingVariable('m', exponent=3)
        self.h = GatingVariable('h', exponent=1)

    def compute_current(self, V, C, Cm):
        """I_Na = g_Na * m³ * h * (V - E_Na)"""
        return (self.g_max / Cm) * self.m.current**3 * self.h.current * (V - self.E_rev)

    def update_state(self, V, V0, dt):
        """Update m and h gates."""
        self.m.update_sbdf2(self._m_kinetics, V, V0, dt)
        self.h.update_sbdf2(self._h_kinetics, V, V0, dt)

    def _m_kinetics(self, m, V):
        """dm/dt = alpha_m * (1-m) - beta_m * m"""
        Vt = 1e3*self.Vt
        alpha = -1e3 * 0.32 * (V - Vt - 13.) / (np.exp(-(V - Vt - 13.)/4.) - 1.)
        beta = 1e3 * 0.28 * (V - Vt - 40.) / (np.exp((V - Vt - 40.)/5.) - 1.)
        return alpha * (1. - m) - beta * m

    def _h_kinetics(self, h, V):
        """dh/dt = alpha_h * (1-h) - beta_h * h"""
        Vt = 1e3*self.Vt
        alpha = 1e3 * 0.128 * np.exp(-(V - Vt - 17.)/18.)
        beta = 1e3 * 4.0 / (1. + np.exp(-(V - Vt - 40.)/5.))
        return alpha * (1. - h) - beta * h

    def initialize_state(self, V_rest, nnodes):
        """Find steady-state m∞ and h∞."""
        # Newton-Raphson to find roots of dm/dt = 0, dh/dt = 0
        V_rest_mV = 1e3 * V_rest

        # m infinity
        m_val = 0.5
        for _ in range(50):
            f = self._m_kinetics(m_val, V_rest_mV)
            if abs(f) < 1e-10:
                break
            df = (self._m_kinetics(m_val + 1e-6, V_rest_mV) - f) / 1e-6
            m_val = np.clip(m_val - f/df, 0, 1)

        # h infinity
        h_val = 0.5
        for _ in range(50):
            f = self._h_kinetics(h_val, V_rest_mV)
            if abs(f) < 1e-10:
                break
            df = (self._h_kinetics(h_val + 1e-6, V_rest_mV) - f) / 1e-6
            h_val = np.clip(h_val - f/df, 0, 1)

        self.m.initialize(m_val, nnodes)
        self.h.initialize(h_val, nnodes)


class PotassiumChannel(ChannelModel):
    """Delayed rectifier K channel (HH n⁴ model)."""

    def __init__(self, g_max=50e-12, E_rev=-90e-3, Vt=-60e-3):
        super().__init__('K', g_max, E_rev)
        self.Vt = Vt
        self.n = GatingVariable('n', exponent=4)

    def compute_current(self, V, C, Cm):
        """I_K = g_K * n⁴ * (V - E_K)"""
        return (self.g_max / Cm) * self.n.current**4 * (V - self.E_rev)

    def update_state(self, V, V0, dt):
        self.n.update_sbdf2(self._n_kinetics, V, V0, dt)

    def _n_kinetics(self, n, V):
        """dn/dt = alpha_n * (1-n) - beta_n * n"""
        Vt = 1e3*self.Vt
        alpha = 1e3 * -0.032 * (V - Vt - 15.) / (np.exp(-(V - Vt - 15.)/5.) - 1.)
        beta = 1e3 * 0.5 * np.exp(-(V - Vt - 10.)/40.)
        return alpha * (1. - n) - beta * n

    def initialize_state(self, V_rest, nnodes):
        V_rest_mV = 1e3 * V_rest
        n_val = 0.5
        for _ in range(50):
            f = self._n_kinetics(n_val, V_rest_mV)
            if abs(f) < 1e-10:
                break
            df = (self._n_kinetics(n_val + 1e-6, V_rest_mV) - f) / 1e-6
            n_val = np.clip(n_val - f/df, 0, 1)
        self.n.initialize(n_val, nnodes)


class CalciumChannel(ChannelModel):
    """Calcium channel with Ca-dependent inactivation."""

    def __init__(self, g_max=5.16*6e-12, K_Ca=140e-18):
        super().__init__('Ca', g_max, E_rev=None)  # Dynamic E_rev
        self.K_Ca = K_Ca
        self.q = GatingVariable('q', exponent=1)

    def compute_current(self, V, C, Cm):
        """I_Ca = g_Ca * q * ρ(C) * (V - E_Ca(C))"""
        rho = self.K_Ca / (self.K_Ca + C)
        E_Ca = self.get_reversal_potential(V, C)
        return (self.g_max / Cm) * self.q.current * rho * (V - E_Ca)

    def get_reversal_potential(self, V, C):
        """Nernst potential for calcium."""
        co = 2000e-18  # External Ca (umol/um³)
        return 12.5e-3 * np.log(co / C)

    def update_state(self, V, V0, dt):
        self.q.update_sbdf2(self._q_kinetics, V, V0, dt)

    def _q_kinetics(self, q, V):
        """dq/dt = (σ(V) - q) / τ(V)"""
        tau = 7.8 / (np.exp((V + 6.)/16.) + np.exp(-(V + 6.)/16.))
        sigma = 1.0 / (1.0 + np.exp(-(V - 3)/8.))
        return (sigma - q) / tau

    def initialize_state(self, V_rest, nnodes):
        V_rest_mV = 1e3 * V_rest
        q_val = 0.5
        for _ in range(50):
            f = self._q_kinetics(q_val, V_rest_mV)
            if abs(f) < 1e-10:
                break
            df = (self._q_kinetics(q_val + 1e-6, V_rest_mV) - f) / 1e-6
            q_val = np.clip(q_val - f/df, 0, 1)
        self.q.initialize(q_val, nnodes)


class LeakChannel(ChannelModel):
    """Passive leak channel (no gating)."""

    def __init__(self, g_max=0.05e-12, E_rev=-70e-3):
        super().__init__('leak', g_max, E_rev)

    def compute_current(self, V, C, Cm):
        """I_leak = g_leak * (V - E_leak)"""
        return (self.g_max / Cm) * (V - self.E_rev)

    def update_state(self, V, V0, dt):
        pass  # No gating

    def initialize_state(self, V_rest, nnodes):
        pass  # No state

    def contributes_to_leak_balance(self) -> bool:
        return False  # Don't include leak in its own balancing
```

## 3. Channel Factory (`cable_eq/factory.py`)

```python
from typing import Type
from .channels import SodiumChannel, PotassiumChannel, CalciumChannel, LeakChannel
from .base import ChannelModel

class ChannelFactory:
    """Factory for creating ion channels."""

    @staticmethod
    def create_Na(g_max=500e-12, E_rev=50e-3, Vt=-60e-3):
        """Create fast sodium channel (HH m³h).

        Args:
            g_max: Maximum conductance (S/um²)
            E_rev: Reversal potential (V)
            Vt: Threshold voltage (V)
        """
        return SodiumChannel(g_max, E_rev, Vt)

    @staticmethod
    def create_K(g_max=50e-12, E_rev=-90e-3, Vt=-60e-3):
        """Create delayed rectifier K channel (HH n⁴)."""
        return PotassiumChannel(g_max, E_rev, Vt)

    @staticmethod
    def create_Ca(g_max=5.16*6e-12, K_Ca=140e-18):
        """Create calcium channel with Ca-dependent inactivation."""
        return CalciumChannel(g_max, K_Ca)

    @staticmethod
    def create_leak(g_max=0.05e-12, V_rest=-70e-3):
        """Create passive leak channel.

        Reversal potential set to V_rest. Will be balanced by
        set_leak_reversal() to ensure resting equilibrium.
        """
        return LeakChannel(g_max, E_rev=V_rest)

    @staticmethod
    def create_custom(channel_class: Type[ChannelModel], **kwargs):
        """Create custom channel from user class.

        Example:
            class MyChannel(ChannelModel):
                def __init__(self, g_max, tau):
                    super().__init__('custom', g_max, -70e-3)
                    self.tau = tau
                # ... implement abstract methods

            ch = ChannelFactory.create_custom(MyChannel, g_max=10e-12, tau=5e-3)
        """
        return channel_class(**kwargs)
```

## 4. Refactored CableSettings (`cable_eq/current_model.py`)

```python
import numpy as np
from typing import List
from .base import ChannelModel

class CableSettings:
    """Manages voltage dynamics and ion channels."""

    def __init__(self, neuron):
        self.neuron = neuron
        self.Cm = neuron.settings.Cm
        self.Rm = neuron.settings.Rm
        self.Ext = 0.0  # Leak balancing current

        self.channels: List[ChannelModel] = []
        self.profile = None
        self.profile0 = None

    def add_channel(self, channel: ChannelModel):
        """Add an ion channel."""
        self.channels.append(channel)

    def initialize_all_channels(self, V_rest: float, nnodes: int):
        """Initialize gating variables at resting potential."""
        for channel in self.channels:
            channel.initialize_state(V_rest, nnodes)

    def set_leak_reversal(self):
        """Compute balancing current to maintain resting potential.

        After all channels are initialized at V_rest with steady-state
        gating variables, compute the net current. This is stored as Ext
        and added to the current profile to ensure equilibrium at rest.
        """
        V = 1e-3 * self.neuron.sol.V  # Convert mV → V
        C = 1e-15 * self.neuron.sol.C  # Convert μM → umol/um³

        I_total = 0.0
        for channel in self.channels:
            if channel.contributes_to_leak_balance():
                I_total += channel.compute_current(V, C, self.Cm)

        # Store balancing current (will be added in current_profile)
        self.Ext = I_total

    def current_profile(self, comm=None, comm_iter=None):
        """Compute total current from all channels."""
        V = 1e-3 * self.neuron.sol.V
        C = 1e-15 * self.neuron.sol.C

        I_total = 0.0
        for channel in self.channels:
            I_total -= channel.compute_current(V, C, self.Cm)

        # Add synaptic current if needed
        if self.neuron.currents.get('I_syn', False):
            I_total -= self.I_syn(comm, comm_iter)

        # Add balancing current for resting equilibrium
        self.profile = I_total + self.Ext

    def update_gating_variables(self, comm=None, comm_iter=None):
        """Update gating variables for all channels."""
        V = self.neuron.sol.V  # mV
        V0 = self.neuron.sol.V0  # mV
        dt = self.neuron.settings.dt

        for channel in self.channels:
            channel.update_state(V, V0, dt)

    def I_syn(self, comm, comm_iter):
        """Synaptic current (unchanged from original)."""
        _, idx = comm.get_pre_V(comm_iter)
        V = np.zeros_like(self.neuron.sol.V)
        V[idx] = 1e-3*self.neuron.sol.V[idx]

        weight = np.zeros_like(V)
        for neuron in comm.weight_map[f'{comm_iter}']:
            for idx, pair in enumerate(comm.pre_post_node_map[f'{comm_iter}'][neuron]):
                weight[pair[1]] = comm.weight_map[f'{comm_iter}'][neuron][idx]

        g_syn = 2.7e-10  # S/um²
        E_syn = 0.0  # V
        s = getattr(self, 's', np.zeros_like(V))  # Synaptic variable

        return weight * (g_syn / self.Cm) * s * (V - E_syn)
```

## 5. Usage Examples

### Basic Setup
```python
from spine.utils import NeuronModel
from spine.cable_eq import ChannelFactory
from spine.graph import gen_beam

# Create neuron
neuron = NeuronModel()
gen_beam(neuron, L=100, n=500)

# Add standard Hodgkin-Huxley channels
neuron.CableSettings.add_channel(ChannelFactory.create_Na())
neuron.CableSettings.add_channel(ChannelFactory.create_K())
neuron.CableSettings.add_channel(ChannelFactory.create_Ca())
neuron.CableSettings.add_channel(ChannelFactory.create_leak())

# Enable voltage coupling
neuron.model['voltage_coupling'] = True

# Channels are automatically initialized and balanced
```

### Custom Channels
```python
# Add L-type calcium channel
neuron.CableSettings.add_channel(
    ChannelFactory.create_Ca(g_max=8e-12, K_Ca=200e-18)
)

# Add SK-type calcium-activated potassium channel
from spine.cable_eq.base import ChannelModel

class SKChannel(ChannelModel):
    """Small-conductance Ca-activated K channel."""

    def __init__(self, g_max, K_half, n_hill=4):
        super().__init__('SK', g_max, -90e-3)
        self.K_half = K_half
        self.n_hill = n_hill

    def compute_current(self, V, C, Cm):
        # Hill equation for Ca activation
        activation = C**self.n_hill / (C**self.n_hill + self.K_half**self.n_hill)
        return (self.g_max / Cm) * activation * (V - self.E_rev)

    def update_state(self, V, V0, dt):
        pass  # Instantaneous activation

    def initialize_state(self, V_rest, nnodes):
        pass  # No state

neuron.CableSettings.add_channel(SKChannel(g_max=20e-12, K_half=0.5e-15))
```

### Per-Channel Control
```python
# Enable/disable channels dynamically
na_channel = ChannelFactory.create_Na(g_max=500e-12)
k_channel = ChannelFactory.create_K(g_max=50e-12)

neuron.CableSettings.add_channel(na_channel)
neuron.CableSettings.add_channel(k_channel)

# Later: modify conductance
na_channel.g_max = 800e-12  # Increase Na conductance

# Or remove a channel
neuron.CableSettings.channels.remove(k_channel)
```
