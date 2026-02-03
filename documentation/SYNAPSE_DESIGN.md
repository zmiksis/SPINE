# Extensible Synapse Architecture

## Design Philosophy

SPINE's synapse system uses a factory pattern to separate concerns: temporal dynamics, receptor kinetics, and spatial targeting. This allows users to mix-and-match components or create entirely custom synapses without modifying core code.

## Key Concepts

1. **Temporal Patterns** - Define when/how stimulation occurs (pulse, train, exponential decay)
2. **Receptor Models** - Define biophysical response (AMPA, NMDA, GABA kinetics)
3. **Chemical Synapses** - Combine temporal + receptor for voltage-based transmission
4. **Direct Calcium Synapses** - Bypass receptors, inject Ca²⁺ directly
5. **Calcium-Coupled Synapses** - Inter-neuron communication via Ca²⁺ levels

## 1. Base Classes (`synapse/base.py`)

### TemporalPattern (Abstract Base)

```python
from abc import ABC, abstractmethod
from typing import Optional

class TemporalPattern(ABC):
    """Defines when and how stimulation occurs."""

    def __init__(self, duration: Optional[float] = None, start_time: float = 0.0):
        """
        Args:
            duration: Total duration (s), None = always active
            start_time: When to begin (s)
        """
        self.duration = duration
        self.start_time = start_time

    @abstractmethod
    def compute_amplitude(self, t: float, base_amplitude: float) -> float:
        """Compute time-varying amplitude.

        Args:
            t: Current time (s)
            base_amplitude: Peak amplitude

        Returns:
            Effective amplitude at time t
        """
        pass

    def is_active(self, t: float) -> bool:
        """Check if pattern is currently active."""
        if t < self.start_time:
            return False
        if self.duration is None:
            return True
        return t < (self.start_time + self.duration)
```

### ReceptorModel (Abstract Base)

```python
class ReceptorModel(ABC):
    """Defines biophysical response of synaptic receptor."""

    @abstractmethod
    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Compute synaptic flux based on pre/post state.

        Args:
            pre_data: {'V': voltage (V), 'C': calcium (μM)}
            post_data: {'V': voltage (V), 'C': calcium (μM)}

        Returns:
            Flux value (units depend on receptor type)
        """
        pass

    @abstractmethod
    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update internal state variables (e.g., gating, desensitization)."""
        pass

    def initialize_state(self, nnodes: int):
        """Initialize state variables."""
        pass
```

### Synapse (Abstract Base)

```python
class Synapse(ABC):
    """Base class for all synapse types."""

    def __init__(self, nodes: list):
        """
        Args:
            nodes: List of node indices where synapse is present
        """
        self.nodes = nodes

    @abstractmethod
    def compute_flux(self, neuron, t: float, comm=None, comm_iter: int = 0) -> np.ndarray:
        """Compute flux at all nodes.

        Args:
            neuron: NeuronModel instance
            t: Current time (s)
            comm: SynapseCommunicator (for inter-neuron synapses)
            comm_iter: Index in communicator

        Returns:
            Flux array (shape: nnodes)
        """
        pass
```

## 2. Temporal Patterns (`synapse/temporal.py`)

### ConstantPattern

```python
class ConstantPattern(TemporalPattern):
    """Constant amplitude during active window."""

    def compute_amplitude(self, t: float, base_amplitude: float) -> float:
        if not self.is_active(t):
            return 0.0
        return base_amplitude
```

### LinearDecayPattern

```python
class LinearDecayPattern(TemporalPattern):
    """Linear ramp down from peak to zero."""

    def compute_amplitude(self, t: float, base_amplitude: float) -> float:
        if not self.is_active(t):
            return 0.0

        elapsed = t - self.start_time
        remaining_fraction = 1.0 - (elapsed / self.duration)
        return base_amplitude * max(0.0, remaining_fraction)
```

### ExponentialDecayPattern

```python
class ExponentialDecayPattern(TemporalPattern):
    """Exponential decay with time constant tau."""

    def __init__(self, tau: float, duration: Optional[float] = None, start_time: float = 0.0):
        """
        Args:
            tau: Decay time constant (s)
        """
        super().__init__(duration, start_time)
        self.tau = tau

    def compute_amplitude(self, t: float, base_amplitude: float) -> float:
        if not self.is_active(t):
            return 0.0

        elapsed = t - self.start_time
        return base_amplitude * np.exp(-elapsed / self.tau)
```

### TrainPattern

```python
class TrainPattern(TemporalPattern):
    """Pulse train with specified frequency."""

    def __init__(self, pulse_duration: float, frequency: float, n_pulses: int, start_time: float = 0.0):
        """
        Args:
            pulse_duration: Width of each pulse (s)
            frequency: Pulse frequency (Hz)
            n_pulses: Total number of pulses
        """
        self.pulse_duration = pulse_duration
        self.frequency = frequency
        self.n_pulses = n_pulses
        self.period = 1.0 / frequency
        total_duration = n_pulses * self.period
        super().__init__(total_duration, start_time)

    def compute_amplitude(self, t: float, base_amplitude: float) -> float:
        if not self.is_active(t):
            return 0.0

        elapsed = t - self.start_time
        pulse_number = int(elapsed / self.period)

        if pulse_number >= self.n_pulses:
            return 0.0

        time_in_pulse = elapsed - pulse_number * self.period
        if time_in_pulse < self.pulse_duration:
            return base_amplitude
        return 0.0
```

## 3. Receptor Models (`synapse/receptors.py`)

### AMPAReceptor

```python
class AMPAReceptor(ReceptorModel):
    """Fast excitatory AMPA glutamate receptor."""

    def __init__(self, g_max: float = 2.7e-10, E_rev: float = 0.0,
                 tau_rise: float = 0.5e-3, tau_decay: float = 2.0e-3):
        """
        Args:
            g_max: Maximum conductance (S/um²)
            E_rev: Reversal potential (V)
            tau_rise: Rise time constant (s)
            tau_decay: Decay time constant (s)
        """
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.s = 0.0  # Gating variable

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Flux = g * s * (V_post - E_rev)."""
        V_post = post_data.get('V', -0.072)
        return self.g_max * self.s * (V_post - self.E_rev)

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update gating: ds/dt = alpha*(1-s) - s/tau_decay."""
        V_pre = pre_data.get('V', -0.072)

        # Sigmoidal activation by presynaptic voltage
        alpha = 0.55 / (1.0 + np.exp(-V_pre / 0.004))

        ds_dt = alpha * (1.0 - self.s) - self.s / self.tau_decay
        self.s += ds_dt * dt
        self.s = np.clip(self.s, 0.0, 1.0)
```

### NMDAReceptor

```python
class NMDAReceptor(ReceptorModel):
    """Slow excitatory NMDA receptor with Mg²⁺ block."""

    def __init__(self, g_max: float = 1.0e-10, E_rev: float = 0.0,
                 tau_rise: float = 2.0e-3, tau_decay: float = 50.0e-3,
                 Mg_conc: float = 1.0):
        """
        Args:
            Mg_conc: External Mg²⁺ concentration (mM)
        """
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.Mg_conc = Mg_conc
        self.s = 0.0

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Flux = g * s * B(V) * (V_post - E_rev).

        B(V) = Mg²⁺ block factor (voltage-dependent)
        """
        V_post = post_data.get('V', -0.072)

        # Mg²⁺ block (Jahr & Stevens 1990)
        B = 1.0 / (1.0 + (self.Mg_conc / 3.57) * np.exp(-0.062 * V_post * 1000))

        return self.g_max * self.s * B * (V_post - self.E_rev)

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update with slower kinetics than AMPA."""
        V_pre = pre_data.get('V', -0.072)
        alpha = 0.3 / (1.0 + np.exp(-V_pre / 0.004))

        ds_dt = alpha * (1.0 - self.s) - self.s / self.tau_decay
        self.s += ds_dt * dt
        self.s = np.clip(self.s, 0.0, 1.0)
```

### GABAReceptor

```python
class GABAReceptor(ReceptorModel):
    """Fast inhibitory GABA_A receptor."""

    def __init__(self, g_max: float = 5.0e-10, E_rev: float = -0.080,
                 tau_rise: float = 0.5e-3, tau_decay: float = 5.0e-3):
        """
        Args:
            E_rev: Reversal potential (typically -80 mV for Cl⁻)
        """
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.s = 0.0

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        V_post = post_data.get('V', -0.072)
        return self.g_max * self.s * (V_post - self.E_rev)

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        V_pre = pre_data.get('V', -0.072)
        alpha = 0.5 / (1.0 + np.exp(-V_pre / 0.004))

        ds_dt = alpha * (1.0 - self.s) - self.s / self.tau_decay
        self.s += ds_dt * dt
        self.s = np.clip(self.s, 0.0, 1.0)
```

### CalciumModulatedAMPAReceptor

```python
class CalciumModulatedAMPAReceptor(AMPAReceptor):
    """AMPA receptor with Ca²⁺-dependent facilitation."""

    def __init__(self, g_max: float = 2.7e-10, E_rev: float = 0.0,
                 ca_sensitivity: float = 5.0, baseline_release: float = 0.3,
                 ca_baseline: float = 0.05, **kwargs):
        """
        Args:
            ca_sensitivity: Multiplicative factor per μM Ca²⁺
            baseline_release: Release probability at resting Ca²⁺
            ca_baseline: Resting Ca²⁺ level (μM)
        """
        super().__init__(g_max, E_rev, **kwargs)
        self.ca_sensitivity = ca_sensitivity
        self.baseline_release = baseline_release
        self.ca_baseline = ca_baseline

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Scale conductance by presynaptic Ca²⁺ level."""
        V_post = post_data.get('V', -0.072)
        C_pre = pre_data.get('C', self.ca_baseline)

        # Ca²⁺-dependent facilitation
        Ca_factor = self.baseline_release + self.ca_sensitivity * (C_pre - self.ca_baseline)
        Ca_factor = max(0.0, Ca_factor)  # Prevent negative

        return self.g_max * self.s * Ca_factor * (V_post - self.E_rev)
```

## 4. Concrete Synapse Classes

### DirectCalciumRelease (`synapse/temporal.py`)

```python
class DirectCalciumRelease(Synapse):
    """Direct calcium injection (no receptor dynamics)."""

    def __init__(self, nodes: list, temporal_pattern: TemporalPattern, amplitude: float):
        """
        Args:
            temporal_pattern: When/how to inject
            amplitude: Peak Ca²⁺ flux (umol/(um²·s))
        """
        super().__init__(nodes)
        self.temporal_pattern = temporal_pattern
        self.amplitude = amplitude

    def compute_flux(self, neuron, t: float, comm=None, comm_iter: int = 0) -> np.ndarray:
        """Return flux array with non-zero values at target nodes."""
        flux = np.zeros(neuron.sol.C.shape[0])

        current_amplitude = self.temporal_pattern.compute_amplitude(t, self.amplitude)
        flux[self.nodes] = current_amplitude

        return flux
```

### ChemicalSynapse (`synapse/receptors.py`)

```python
class ChemicalSynapse(Synapse):
    """Voltage-based synaptic transmission via receptor."""

    def __init__(self, post_nodes: list, receptor_model: ReceptorModel, weight: float = 1.0):
        """
        Args:
            post_nodes: Postsynaptic node indices
            receptor_model: ReceptorModel instance
            weight: Synaptic strength multiplier
        """
        super().__init__(post_nodes)
        self.receptor_model = receptor_model
        self.weight = weight

    def compute_flux(self, neuron, t: float, comm=None, comm_iter: int = 0) -> np.ndarray:
        """Compute receptor-mediated flux at postsynaptic nodes."""
        flux = np.zeros(neuron.sol.C.shape[0])

        if comm is None:
            # Single neuron - no presynaptic data
            return flux

        # Get presynaptic voltage and calcium
        V_pre, pre_nodes = comm.get_pre_V(comm_iter)
        C_pre, _ = comm.get_pre_C(comm_iter)

        # Get postsynaptic state
        V_post = neuron.sol.V[self.nodes] * 1e-3  # mV → V
        C_post = neuron.sol.C[self.nodes] * 1e-15  # μM → umol/um³

        # Compute flux for each synapse
        for i, post_node in enumerate(self.nodes):
            pre_data = {'V': V_pre[i] if i < len(V_pre) else -0.072,
                       'C': C_pre[i] if i < len(C_pre) else 0.05}
            post_data = {'V': V_post[i], 'C': C_post[i]}

            flux[post_node] = self.weight * self.receptor_model.compute_flux(pre_data, post_data)

        return flux
```

### CalciumCoupledSynapse (`synapse/calcium_coupled.py`)

```python
class CalciumCoupledSynapse(Synapse):
    """Inter-neuron coupling based on presynaptic Ca²⁺ level."""

    def __init__(self, post_nodes: list, sensitivity: float = 1e3):
        """
        Args:
            sensitivity: Flux per μM presynaptic Ca²⁺
        """
        super().__init__(post_nodes)
        self.sensitivity = sensitivity

    def compute_flux(self, neuron, t: float, comm=None, comm_iter: int = 0) -> np.ndarray:
        """Flux proportional to presynaptic calcium."""
        flux = np.zeros(neuron.sol.C.shape[0])

        if comm is None:
            return flux

        C_pre, _ = comm.get_pre_C(comm_iter)

        for i, post_node in enumerate(self.post_nodes):
            if i < len(C_pre):
                flux[post_node] = self.sensitivity * C_pre[i]  # C_pre in μM

        return flux
```

## 5. Factory Pattern (`synapse/factory.py`)

```python
class SynapseFactory:
    """Convenient factory for creating common synapse types."""

    # ===== Direct Calcium Injection =====

    @staticmethod
    def create_calcium_pulse(nodes: list, amplitude: float, duration: float, start_time: float = 0.0):
        """Constant Ca²⁺ injection for specified duration."""
        pattern = ConstantPattern(duration=duration, start_time=start_time)
        return DirectCalciumRelease(nodes, pattern, amplitude)

    @staticmethod
    def create_calcium_linear(nodes: list, amplitude: float, duration: float, start_time: float = 0.0):
        """Ca²⁺ injection with linear decay."""
        pattern = LinearDecayPattern(duration=duration, start_time=start_time)
        return DirectCalciumRelease(nodes, pattern, amplitude)

    @staticmethod
    def create_calcium_exponential(nodes: list, amplitude: float, tau: float,
                                   duration: float = None, start_time: float = 0.0):
        """Ca²⁺ injection with exponential decay."""
        pattern = ExponentialDecayPattern(tau=tau, duration=duration, start_time=start_time)
        return DirectCalciumRelease(nodes, pattern, amplitude)

    @staticmethod
    def create_calcium_train(nodes: list, amplitude: float, pulse_duration: float,
                            frequency: float, n_pulses: int, start_time: float = 0.0):
        """Ca²⁺ injection as pulse train."""
        pattern = TrainPattern(pulse_duration=pulse_duration, frequency=frequency,
                              n_pulses=n_pulses, start_time=start_time)
        return DirectCalciumRelease(nodes, pattern, amplitude)

    @staticmethod
    def create_ip3_pulse(nodes: list, amplitude: float, duration: float, start_time: float = 0.0):
        """IP3 injection (stored in neuron.synapse_instances)."""
        pattern = ConstantPattern(duration=duration, start_time=start_time)
        syn = DirectCalciumRelease(nodes, pattern, amplitude)
        syn.domain = 'ip3'  # Flag for IP3 targeting
        return syn

    # ===== Chemical Synapses (Voltage-Based) =====

    @staticmethod
    def create_AMPA(post_nodes: list, g_max: float = 2.7e-10, E_rev: float = 0.0, weight: float = 1.0):
        """AMPA glutamate receptor (fast excitatory)."""
        receptor = AMPAReceptor(g_max=g_max, E_rev=E_rev)
        return ChemicalSynapse(post_nodes, receptor, weight)

    @staticmethod
    def create_NMDA(post_nodes: list, g_max: float = 1.0e-10, E_rev: float = 0.0,
                    Mg_conc: float = 1.0, weight: float = 1.0):
        """NMDA glutamate receptor (slow excitatory with Mg²⁺ block)."""
        receptor = NMDAReceptor(g_max=g_max, E_rev=E_rev, Mg_conc=Mg_conc)
        return ChemicalSynapse(post_nodes, receptor, weight)

    @staticmethod
    def create_GABA(post_nodes: list, g_max: float = 5.0e-10, E_rev: float = -0.080, weight: float = 1.0):
        """GABA_A receptor (fast inhibitory)."""
        receptor = GABAReceptor(g_max=g_max, E_rev=E_rev)
        return ChemicalSynapse(post_nodes, receptor, weight)

    @staticmethod
    def create_calcium_modulated_AMPA(post_nodes: list, g_max: float = 2.7e-10,
                                      ca_sensitivity: float = 5.0, baseline_release: float = 0.3,
                                      ca_baseline: float = 0.05, weight: float = 1.0):
        """AMPA with Ca²⁺-dependent facilitation (short-term plasticity)."""
        receptor = CalciumModulatedAMPAReceptor(g_max=g_max, ca_sensitivity=ca_sensitivity,
                                               baseline_release=baseline_release,
                                               ca_baseline=ca_baseline)
        return ChemicalSynapse(post_nodes, receptor, weight)

    # ===== Calcium-Coupled Synapses =====

    @staticmethod
    def create_calcium_coupled(post_nodes: list, sensitivity: float = 1e3):
        """Inter-neuron coupling via presynaptic Ca²⁺."""
        return CalciumCoupledSynapse(post_nodes, sensitivity)

    # ===== Custom Synapses =====

    @staticmethod
    def create_custom_receptor(post_nodes: list, receptor_class, weight: float = 1.0, **receptor_kwargs):
        """Create synapse with user-defined receptor."""
        receptor = receptor_class(**receptor_kwargs)
        return ChemicalSynapse(post_nodes, receptor, weight)

    @staticmethod
    def create_custom_temporal(nodes: list, temporal_class, amplitude: float, **temporal_kwargs):
        """Create direct Ca²⁺ synapse with user-defined temporal pattern."""
        pattern = temporal_class(**temporal_kwargs)
        return DirectCalciumRelease(nodes, pattern, amplitude)
```

## 6. Integration with Solver

The solver (`solver/sbdf.py`) processes synapses in this order:

1. **Initialization** (before time loop):
   ```python
   for syn in neuron.synapse_instances:
       if hasattr(syn, 'receptor_model') and hasattr(syn.receptor_model, 'initialize_state'):
           syn.receptor_model.initialize_state(nnodes)
   ```

2. **During time loop**:
   ```python
   # Update receptor states
   for syn in neuron.synapse_instances:
       if hasattr(syn, 'receptor_model'):
           pre_data = {...}  # From communicator
           post_data = {...}  # From neuron
           syn.receptor_model.update_state(dt, pre_data, post_data)

   # Compute fluxes
   total_flux = np.zeros(nnodes)
   for syn in neuron.synapse_instances:
       flux = syn.compute_flux(neuron, t, comm, comm_iter)

       # Route to appropriate compartment
       if hasattr(syn, 'domain') and syn.domain == 'ip3':
           neuron.sol.IP3 += flux * dt
       else:
           total_flux += flux

   # Add to calcium dynamics
   neuron.sol.C += total_flux * dt
   ```

## 7. Usage Examples

### Example 1: Single Neuron Calcium Wave

```python
from spine.utils import NeuronModel
from spine.synapse import SynapseFactory
from spine.solver import SBDF

neuron = NeuronModel()
neuron.model['CYT_buffer'] = True
neuron.model['PMCA'] = True

# Add calcium pulse
syn = SynapseFactory.create_calcium_pulse(
    nodes=[500],
    amplitude=3e-12,
    duration=1e-3
)
neuron.synapse_instances = [syn]

SBDF(neuron)
```

### Example 2: Inter-Neuron AMPA Synapse

```python
from spine.synapse import SynapseFactory, SynapseCommunicator

# Create two neurons
neuron1 = NeuronModel()  # Presynaptic
neuron2 = NeuronModel()  # Postsynaptic

# Add AMPA to postsynaptic neuron
ampa = SynapseFactory.create_AMPA(
    post_nodes=[100],
    g_max=3e-10,
    weight=1.5
)
neuron2.synapse_instances = [ampa]

# Setup communication
comm = SynapseCommunicator([neuron1, neuron2])
comm.add_synapse_connection((0, 0), (1, 100))  # pre node 0 → post node 100

SBDF([neuron1, neuron2], comm=comm)
```

### Example 3: Custom Receptor

```python
from spine.synapse.base import ReceptorModel
from spine.synapse import SynapseFactory

class MyReceptor(ReceptorModel):
    def __init__(self, g_max, tau):
        self.g_max = g_max
        self.tau = tau
        self.s = 0.0

    def compute_flux(self, pre_data, post_data):
        V_post = post_data['V']
        return self.g_max * self.s * V_post

    def update_state(self, dt, pre_data, post_data):
        V_pre = pre_data['V']
        target = 1.0 if V_pre > -0.050 else 0.0
        self.s += (target - self.s) / self.tau * dt

# Use custom receptor
syn = SynapseFactory.create_custom_receptor(
    post_nodes=[50],
    receptor_class=MyReceptor,
    weight=2.0,
    g_max=1e-10,
    tau=5e-3
)
neuron.synapse_instances = [syn]
```
