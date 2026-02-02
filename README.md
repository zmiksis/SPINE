# SPINE

**Simulator for Plasticity and Integrated Neuronal Events**

*An integrated voltage-calcium-synaptic simulator for neuronal networks*

## Overview

Calcium ions (Ca²⁺) are universal intracellular messengers that orchestrate critical neuronal processes including synaptic plasticity, gene expression, and cell survival. Understanding calcium signaling is essential for investigating learning and memory mechanisms, as well as neurodegenerative diseases like Alzheimer's and Parkinson's. However, the complex interplay between voltage dynamics, calcium release from intracellular stores, and diffusion along dendritic arbors makes calcium signaling inherently difficult to study experimentally.

SPINE addresses this challenge by providing a **spatially-explicit, mechanistically-detailed** simulation framework that integrates:
- **Biophysical realism**: Cytosolic and ER calcium dynamics coupled to voltage via cable equation with Hodgkin-Huxley channels
- **Multiple spatial scales**: From single-compartment to complex dendritic morphologies (10,000+ nodes)
- **Network-level interactions**: Inter-neuron synaptic coupling with calcium-dependent plasticity
- **Computational efficiency**: Optimized for performance with parallel simulation support

### What Makes SPINE Novel?

**Integrated Multi-Compartment Modeling**: SPINE tightly couples the cable equation with spatially-distributed calcium dynamics, ER stores, and IP3 signaling in a unified framework.

**Extensible Architecture**: Modern factory patterns for ion channels and synapses make it trivial to add custom mechanisms (new channels, receptors, or plasticity rules) without modifying core.

**Performance-Optimized**: Implements state-of-the-art numerical techniques that enable simulations of large morphologies at microsecond temporal resolution.

**Python-First Design**: Leverages NumPy/SciPy for ease of use. No compilation required.

### Use Cases

- **Synaptic plasticity research**: Model calcium-dependent learning rules (LTP/LTD) with spatial resolution
- **Calcium wave propagation**: Investigate regenerative calcium signals and their interactions with voltage
- **Drug discovery**: Test effects of channel blockers or modulators on calcium homeostasis
- **Neurodegenerative disease modeling**: Simulate pathological calcium dysregulation
- **Educational tool**: Visualize complex calcium dynamics with ParaView integration

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [User Guide](#user-guide)
  - [Basic Workflow](#basic-workflow)
  - [Calcium Dynamics Models](#calcium-dynamics-models)
  - [Synaptic Inputs](#synaptic-inputs)
  - [Recording and Visualization](#recording-and-visualization)
  - [Performance Optimization](#performance-optimization)
- [Examples](#examples)
- [Developer Guide](#developer-guide)
  - [Code Architecture](#code-architecture)
  - [Adding New Features](#adding-new-features)
  - [Memory Model](#memory-model)
  - [Testing](#testing)
  - [Contributing](#contributing)
- [Units and Conventions](#units-and-conventions)
- [Citation](#citation)
- [License](#license)

---

## Features

### Core Capabilities
- **1D Calcium Dynamics**: Cytosolic and ER calcium on dendritic morphologies
- **Multiple Exchange Mechanisms**: PMCA, NCX, VDCC, RyR, SERCA, IP3R, SOC channels
- **Voltage Coupling**: Cable equation with Hodgkin-Huxley-type channels (Na, K, Ca, leak)
- **Synaptic Transmission**: AMPA, NMDA, GABA receptors with voltage and calcium dependence
- **Inter-Neuron Coupling**: Calcium and voltage-based synaptic communication
- **Multiprocessing**: Efficient parallel simulation of neuron populations

### Advanced Features
- **Extensible Synapse Architecture**: Easy-to-use factory patterns for custom synapses
- **Calcium-Dependent Plasticity**: Short-term facilitation, depression modeling
- **Flux Recording**: Node-specific or full-array flux data
- **Customizable VTU Output**: ParaView visualization with selective field output

---

## Installation

### From Source (Editable Install for Development)

```bash
git clone https://github.com/zmiksis/SPINE.git
cd SPINE
pip install -e .
```

### From Built Package

```bash
python -m build
pip install dist/spine-*.whl
```

### To Uninstall

```bash
pip uninstall spine -y
```

### Dependencies

- Python ≥ 3.8
- NumPy
- SciPy

---

## Quick Start

### Simple Calcium Wave

```python
from spine.utils import NeuronModel
from spine.solver import SBDF
from spine.graph import gen_beam
from spine.synapse import SynapseFactory

# Create neuron model instance
neuron = NeuronModel()

# Top data directory, where problem setup and results are stored
neuron.settings.top_folder = './'

# Subdirectory with data for this specific problem
neuron.settings.data_folder = 'calcium_wave/'
neuron.settings.output_folder = 'calcium_wave-output/'

# Settings for coupled calcium model
neuron.model['CYT_buffer'] = True
neuron.model['PMCA'] = True
neuron.model['NCX'] = True
neuron.model['SERCA'] = True
neuron.model['PM_leak'] = True
neuron.model['ER_leak'] = True
neuron.model['RyR'] = True
neuron.model['IP3'] = True
neuron.model['synapse'] = True

# Record all cytosolic calcium data
neuron.recorder['full'] = True
neuron.recorder['full_er'] = True

# Create beam neuron
neuron.geom_builder['const_rad'] = True
neuron.settings.rad = 0.2  # um
gen_beam(neuron, 64, 1000)  # um length, 1000 nodes

# Final time (s)
neuron.settings.T = 70.e-3  # seconds
# Fixed time step
neuron.settings.set_time_step(5.e-6)  # seconds

# Create calcium and IP3 synapses using NEW SynapseFactory
# Linear decay calcium influx at node 500
syn_ca = SynapseFactory.create_calcium_linear(
    nodes=[500],
    amplitude=2.5e-12,
    duration=1.e-3
)

# Linear decay IP3 influx at node 500
syn_ip3 = SynapseFactory.create_ip3_linear(
    nodes=[500],
    amplitude=5.0e-12,
    duration=200.e-3
)

neuron.synapse_instances = [syn_ca, syn_ip3]

# Run solver
SBDF(neuron)

# Results saved to results/<output_folder>/cyt.txt
```

---

## User Guide

### Basic Workflow

1. **Create Neuron Model**
   ```python
   from spine.utils import NeuronModel
   neuron = NeuronModel()
   ```

2. **Configure Calcium Dynamics**
    ```python
    neuron.model['CYT_buffer'] = True
    neuron.model['PMCA'] = True
    neuron.model['NCX'] = True
    neuron.model['SERCA'] = True
    neuron.model['PM_leak'] = True
    neuron.model['ER_leak'] = True
    neuron.model['RyR'] = True
    neuron.model['IP3'] = True
    neuron.model['synapse'] = True
    ```

3. **Generate Geometry**
    ```python
    from spine.graph import gen_beam, read_swc

    # Option 1: Simple beam
    neuron.geom_builder['const_rad'] = True
    neuron.settings.rad = 0.2  # um
    gen_beam(neuron, 64, 1000)  # um length, 1000 nodes

   # Option 2: Load from SWC file
   read_swc(neuron, 'morphology.swc')
   ```

4. **Set Recording Options**
   ```python
   neuron.recorder['full'] = True          # All nodes, all timesteps
   neuron.recorder['nodes'] = [0, 100, 250]  # Specific nodes only
   neuron.recorder['flux_pmca'] = True    # Record PMCA flux
   neuron.recorder['flux_nodes'] = [100]  # Flux at specific nodes
   ```

5. **Run Solver**
   ```python
   from spine.solver import SBDF
   SBDF(neuron)

   # Parallel simulation
   SBDF([neuron1, neuron2], n_workers=2)
   ```

### Voltage Dynamics and Ion Channels

#### NEW Architecture (Recommended)

SPINE provides an extensible ion channel system via `ChannelFactory`, similar to the synapse architecture:

```python
from spine.cable_eq import ChannelFactory
from spine.utils import NeuronModel

neuron = NeuronModel()

# Add Hodgkin-Huxley channels using factory
neuron.CableSettings.add_channel(ChannelFactory.create_Na(g_max=500e-12))
neuron.CableSettings.add_channel(ChannelFactory.create_K(g_max=50e-12))
neuron.CableSettings.add_channel(ChannelFactory.create_Ca(g_max=30e-12))
neuron.CableSettings.add_channel(ChannelFactory.create_leak(g_max=0.05e-12))

# Custom parameters
na_channel = ChannelFactory.create_Na(
    g_max=800e-12,      # High density sodium
    E_rev=55e-3,        # Shifted reversal potential
    Vt=-55e-3           # Lower threshold
)
neuron.CableSettings.add_channel(na_channel)
```

**Available Channel Types:**
- `create_Na()` - Fast sodium channel (m³h gating)
- `create_K()` - Delayed rectifier potassium (n⁴ gating)
- `create_Ca()` - Calcium channel with Ca²⁺-dependent inactivation
- `create_leak()` - Passive leak channel
- `create_custom()` - User-defined channel classes

**Adding Custom Channels:**

```python
from spine.cable_eq.base import ChannelModel, GatingVariable
import numpy as np

class HChannel(ChannelModel):
    """H-current (HCN channel)."""

    def __init__(self, g_max=5e-12, E_rev=-30e-3):
        super().__init__('H', g_max, E_rev)
        self.r = GatingVariable('r', exponent=1)

    def compute_current(self, V, C, Cm):
        return (self.g_max / Cm) * self.r.current * (V - self.E_rev)

    def update_state(self, V, V0, dt):
        self.r.update_sbdf2(self._r_kinetics, V, V0, dt)

    def _r_kinetics(self, r, V):
        tau = 1000.0 / (np.exp((V + 70)/20) + np.exp(-(V + 70)/20))
        r_inf = 1.0 / (1.0 + np.exp((V + 70)/6))
        return (r_inf - r) / tau

    def initialize_state(self, V_rest, nnodes):
        # Find steady state...
        self.r.initialize(r_inf, nnodes)

# Use custom channel
h_channel = ChannelFactory.create_custom(HChannel, g_max=5e-12)
neuron.CableSettings.add_channel(h_channel)
```

#### LEGACY Architecture (Backward Compatible)

```python
# Old flag-based system (still supported)
neuron.currents['I_Na'] = True
neuron.currents['I_K'] = True
neuron.currents['I_Ca'] = True
neuron.currents['I_leak'] = True

# Access legacy parameters
neuron.CableSettings.Na.g = 500e-12
neuron.CableSettings.K.g = 50e-12
neuron.CableSettings.g_Ca = 30e-12
neuron.CableSettings.leak.g = 0.05e-12
```

### Calcium Dynamics Models

#### Exchange Mechanisms

| Mechanism | Description | Key Parameters |
|-----------|-------------|----------------|
| **PMCA** | Plasma membrane Ca²⁺ ATPase | `Ip`, `Kp`, `rhop` |
| **NCX** | Na⁺/Ca²⁺ exchanger | `In`, `Kn`, `rhon` |
| **VDCC** | Voltage-dependent Ca²⁺ channels | `rhovdcc` |
| **RyR** | Ryanodine receptor | `rhoryr`, `Irefryr` |
| **SERCA** | Sarco/ER Ca²⁺ ATPase | `Is`, `Ks`, `rhos` |
| **IP3R** | IP3 receptor | `kp`, `d1`, `d2`, `d3`, `d5`, `rhoI`, `IrefI` |
<!-- | **SOC** | Store-operated channels | Hard-coded in `jSOC()` | -->

#### Accessing Parameters

```python
# Cytosolic exchange parameters (plasma membrane)
neuron.cytExchangeParams['Ip'] = 1.7e-17      # PMCA max current (umol/s)
neuron.cytExchangeParams['Kp'] = 60.e-18      # PMCA Michaelis constant (umol/um³)
neuron.cytExchangeParams['rhop'] = 500.       # PMCA density (um⁻²)
neuron.cytExchangeParams['In'] = 2.5e-15      # NCX max current (umol/s)
neuron.cytExchangeParams['Kn'] = 1.8e-15      # NCX Michaelis constant (umol/um³)
neuron.cytExchangeParams['rhon'] = 15.        # NCX density (um⁻²)
neuron.cytExchangeParams['rhovdcc'] = 1.      # VDCC density (um⁻²)

# ER exchange parameters
neuron.erExchangeParams['Is'] = 6.5e-30       # SERCA rate (umol²/(um³·s))
neuron.erExchangeParams['Ks'] = 180.e-18      # SERCA Michaelis constant (umol/um³)
neuron.erExchangeParams['rhos'] = 2390.       # SERCA density (um⁻²)
neuron.erExchangeParams['rhoryr'] = 3.        # RyR density (um⁻²)
neuron.erExchangeParams['Irefryr'] = 3.5e-12  # RyR reference current (umol/s)
neuron.erExchangeParams['rhoI'] = 17.3        # IP3R density (um⁻²)
neuron.erExchangeParams['IrefI'] = 1.1e-13    # IP3R reference current (umol/s)
neuron.erExchangeParams['d1'] = 0.13e-15      # IP3R Ca²⁺ dissociation constant (umol/um³)
neuron.erExchangeParams['d5'] = 82.3e-18      # IP3R IP3 dissociation constant (umol/um³)
```

### Synaptic Inputs

#### NEW Architecture (Recommended)

SPINE provides a modern, extensible synapse system via `SynapseFactory`:

##### Single-Neuron Stimulation

```python
from spine.synapse import SynapseFactory

# Constant calcium pulse
syn_pulse = SynapseFactory.create_calcium_pulse(
    nodes=[500],
    amplitude=2.5e-12,
    duration=1e-3,
    start_time=10e-3
)

# Linear decay
syn_linear = SynapseFactory.create_calcium_linear(
    nodes=[500],
    amplitude=2.5e-12,
    duration=10e-3
)

# Exponential decay
syn_exp = SynapseFactory.create_calcium_exponential(
    nodes=[500],
    amplitude=3e-12,
    tau=10e-3
)

# Pulse train (e.g., 10 Hz stimulation)
syn_train = SynapseFactory.create_calcium_train(
    nodes=[500],
    amplitude=2.5e-12,
    pulse_duration=1e-3,
    frequency=10.0,
    n_pulses=5
)

# IP3 injection
syn_ip3 = SynapseFactory.create_ip3_pulse(
    nodes=[500],
    amplitude=5e-12,
    duration=100e-3
)

neuron.synapse_instances = [syn_pulse, syn_ip3]
```

##### Inter-Neuron Synaptic Transmission

```python
from spine.synapse import SynapseFactory, SynapseCommunicator

# Create two neurons
neuron1 = NeuronModel()  # Presynaptic
neuron2 = NeuronModel()  # Postsynaptic

# ... configure neurons ...

# Add AMPA synapse to postsynaptic neuron
ampa = SynapseFactory.create_AMPA(
    post_nodes=[100],
    g_max=3e-10,
    weight=1.5
)
neuron2.synapse_instances = [ampa]

# Setup communicator
comm = SynapseCommunicator([neuron1, neuron2])

# Connect presynaptic node 0 to postsynaptic node 100
comm.add_synapse_connection((0, 0), (1, 100))

# Run simulation
SBDF([neuron1, neuron2], comm=comm)
```

##### Available Receptor Types

**Chemical Synapses (Voltage-Based)**
- `create_AMPA()` - Fast excitatory (AMPA glutamate)
- `create_NMDA()` - Slow excitatory (NMDA with Mg²⁺ block)
- `create_GABA()` - Fast inhibitory (GABA_A)
- `create_calcium_modulated_AMPA()` - AMPA with Ca²⁺-dependent facilitation

**Calcium-Coupled Synapses**
- `create_calcium_coupled()` - Ca²⁺-dependent inter-neuron coupling

##### Synaptic Plasticity Example

```python
# Calcium-modulated AMPA with facilitation
ampa_plastic = SynapseFactory.create_calcium_modulated_AMPA(
    post_nodes=[100],
    g_max=2.7e-10,
    ca_sensitivity=5.0,      # 5× per μM Ca²⁺ increase
    baseline_release=0.3,    # Low baseline → high dynamic range
    ca_baseline=0.05         # Resting Ca²⁺ (μM)
)

# This synapse will show paired-pulse facilitation:
# - First spike → Ca²⁺ elevation
# - Second spike → stronger response due to elevated Ca²⁺
```

#### Legacy Architecture (Backward Compatible)

```python
from spine.synapse import synapse

# Old-style synapse (still supported)
syn = synapse()
syn.type = 'linear'
syn.duration = 1.e-3
syn.j = 2.5e-12
syn.node = [500]
syn.domain = 'cytosol'

neuron.synapse_instances = [syn]
```

### Recording and Visualization

#### Data Recording Options

```python
# Concentration data
neuron.recorder['full'] = True              # All cytosolic Ca²⁺ (all nodes × all timesteps)
neuron.recorder['full_er'] = True           # All ER Ca²⁺
neuron.recorder['full_voltage'] = True      # All voltage
neuron.recorder['nodes'] = [0, 100, 500]    # Specific nodes (time, C, CE, IP3, V)

# Aggregate data
neuron.recorder['soma'] = True              # Soma-averaged Ca²⁺
neuron.recorder['avg_cyt'] = True           # Spatially-averaged cytosolic Ca²⁺
neuron.recorder['avg_er'] = True            # Spatially-averaged ER Ca²⁺

# Integrated calcium
neuron.recorder['total_cyt'] = True         # Cumulative cytosolic Ca²⁺
neuron.recorder['total_er'] = True          # Cumulative ER Ca²⁺
```

#### Flux Recording

```python
# Option 1: Full arrays (all nodes)
neuron.recorder['flux_pmca'] = True         # PMCA flux at all nodes
neuron.recorder['flux_ryr'] = True          # RyR flux at all nodes
neuron.recorder['flux_vdcc'] = True         # VDCC flux at all nodes

# Option 2: Node-specific (MUCH more efficient!)
neuron.recorder['flux_nodes'] = [100, 250, 500]  # Only these nodes
neuron.recorder['flux_pmca'] = True         # PMCA at nodes 100, 250, 500
neuron.recorder['flux_ryr'] = True          # RyR at nodes 100, 250, 500

# Available flux types:
# - flux_pmca, flux_ncx, flux_vdcc, flux_pm_leak, flux_synapse
# - flux_ryr, flux_serca, flux_er_leak, flux_ip3r, flux_soc
# - flux_total_pm, flux_total_er, flux_total_ip3
```

#### VTU Visualization (ParaView)

```python
# Customize VTU output fields
neuron.recorder['vtu_cyt'] = True      # Cytosolic Ca²⁺
neuron.recorder['vtu_er'] = True       # ER Ca²⁺
neuron.recorder['vtu_volt'] = True     # Voltage
neuron.recorder['vtu_radius'] = False  # Skip radius (constant)
neuron.recorder['vtu_calb'] = False    # Skip buffer

# Run with VTU output every 10 timesteps
SBDF(neuron, writeStep=10)

# VTU files written to: results/<output_folder>/vtu/sol_t0000000.vtu
# Load in ParaView for 3D visualization
```

### Performance Optimization

#### Parallel Simulations

```python
# Option 1: Multi-process (recommended for large neurons)
SBDF([neuron1, neuron2, neuron3], n_workers=3)

# Option 2: Thread limiting (for single neuron)
from spine.utils.threading import limit_to_one_thread
limit_to_one_thread()  # Prevent oversubscription
SBDF(neuron)
```

#### Memory Optimization Tips

1. **Use node-specific recording** instead of full arrays
   ```python
   neuron.recorder['flux_nodes'] = [100]  # Not: flux_pmca = True for all nodes
   ```

2. **Disable unnecessary VTU fields**
   ```python
   neuron.recorder['vtu_radius'] = False  # Constant anyway
   neuron.recorder['vtu_calb'] = False    # Not needed
   ```

3. **Reduce VTU output frequency**
   ```python
   SBDF(neuron, writeStep=100)  # Every 100 steps instead of every step
   ```

---

## Examples

| Example | Description | Architecture |
|---------|-------------|--------------|
| `calcium_wave.py` | Basic calcium wave propagation | Legacy |
| `calcium_wave_new.py` | Calcium wave with new synapses | **NEW** |
| `calcium_wave_synapse.py` | Inter-neuron calcium coupling | Legacy |
| `calcium_wave_synapse_new.py` | Inter-neuron coupling | **NEW** |
| `plasticity_demo.py` | Calcium-dependent facilitation | **NEW** |
| `NEURON_wave.py` | Voltage-calcium coupling | Mixed |
| `synapseComm.py` | Communicator demonstration | Utility |
| `parallel_scaling_test.py` | Performance benchmarking | Utility |

**Run an example:**
```bash
cd examples
python calcium_wave_new.py
```

---

## Developer Guide

### Code Architecture

```
SPINE/
├── src/spine/
│   ├── solver/          # SBDF2 time stepping, discretization
│   │   ├── sbdf.py      # Main solver entry point
│   │   ├── helper.py    # Solver utilities, state management
│   │   └── discretization_matrices.py
│   ├── cytosol.py       # Cytosolic calcium dynamics (JPM)
│   ├── er.py            # ER calcium dynamics (JER)
│   ├── cable_eq/        # Voltage dynamics, ion channels
│   │   ├── base.py      # Abstract ChannelModel class
│   │   ├── channels.py  # Specific implementations (Na, K, Ca, leak)
│   │   ├── factory.py   # ChannelFactory (user entry point)
│   │   └── current_model.py  # Cable equation, gating kinetics
│   ├── synapse/         # Synaptic transmission
│   │   ├── base.py      # Abstract classes
│   │   ├── temporal.py  # Temporal patterns
│   │   ├── receptors.py # Receptor models (AMPA, NMDA, GABA, ...)
│   │   ├── calcium_coupled.py  # Concrete synapse classes
│   │   ├── factory.py   # SynapseFactory (user entry point)
│   │   ├── communicator.py  # Inter-neuron data exchange
│   │   └── model.py     # Legacy synapse (backward compatibility)
│   ├── graph/           # Geometry generation
│   ├── utils/           # Settings, neuron model, threading
│   ├── writer.py        # VTU output
│   └── flux_components.py  # Transient flux storage
├── examples/            # Example scripts
└── tests/               # Unit tests
```

### Adding New Features

#### Adding a New Receptor Model

1. **Create receptor class** in `src/spine/synapse/receptors.py`:
   ```python
   class MyReceptor(ReceptorModel):
       def __init__(self, param1, param2):
           self.param1 = param1
           self.param2 = param2
           self.state = 0.0

       def compute_flux(self, pre_data: dict, post_data: dict) -> float:
           V_pre = pre_data.get('V', -0.072)
           C_pre = pre_data.get('C', 0.05)
           # ... compute flux based on pre/post data ...
           return flux_value

       def update_state(self, dt: float, pre_data: dict, post_data: dict):
           # Update internal state variables
           self.state += (target - self.state) / tau * dt
   ```

2. **Add to exports** in `src/spine/synapse/__init__.py`:
   ```python
   from .receptors import MyReceptor
   __all__ = [..., 'MyReceptor']
   ```

3. **Add factory method** (optional) in `src/spine/synapse/factory.py`:
   ```python
   @staticmethod
   def create_my_receptor(post_nodes, param1=1.0, param2=2.0):
       receptor = MyReceptor(param1, param2)
       return ChemicalSynapse(post_nodes, receptor)
   ```

#### Adding a New Ion Channel

SPINE's extensible channel architecture (similar to synapses) makes it easy to add custom voltage-gated channels:

1. **Create channel class** in `src/spine/cable_eq/channels.py` (or your own module):
   ```python
   from spine.cable_eq.base import ChannelModel, GatingVariable
   import numpy as np

   class MyChannel(ChannelModel):
       """My custom ion channel."""

       def __init__(self, g_max=10e-12, E_rev=-50e-3, tau=10.0):
           super().__init__('MyChannel', g_max, E_rev)
           self.tau = tau  # Time constant (ms)
           self.x = GatingVariable('x', exponent=2)  # x² gating

       def compute_current(self, V, C, Cm):
           """Compute I = g * x² * (V - E)."""
           return (self.g_max / Cm) * self.x.current**2 * (V - self.E_rev)

       def update_state(self, V, V0, dt):
           """Update gating variable using SBDF2."""
           self.x.update_sbdf2(self._x_kinetics, V, V0, dt)

       def _x_kinetics(self, x, V):
           """dx/dt = (x_inf - x) / tau."""
           x_inf = 1.0 / (1.0 + np.exp(-(V + 30)/10))
           return (x_inf - x) / self.tau

       def initialize_state(self, V_rest, nnodes):
           """Find steady state at resting potential."""
           V_rest_mV = 1e3 * V_rest
           x_inf = 1.0 / (1.0 + np.exp(-(V_rest_mV + 30)/10))
           self.x.initialize(x_inf, nnodes)
   ```

2. **Add to exports** in `src/spine/cable_eq/__init__.py`:
   ```python
   from .channels import MyChannel
   __all__ = [..., 'MyChannel']
   ```

3. **Add factory method** (optional) in `src/spine/cable_eq/factory.py`:
   ```python
   @staticmethod
   def create_my_channel(g_max=10e-12, E_rev=-50e-3, tau=10.0):
       """Create my custom channel.

       Args:
           g_max: Maximum conductance (S/um²)
           E_rev: Reversal potential (V)
           tau: Time constant (ms)

       Returns:
           MyChannel instance
       """
       return MyChannel(g_max, E_rev, tau)
   ```

4. **Use in simulations**:
   ```python
   from spine.cable_eq import ChannelFactory

   # Via factory
   my_channel = ChannelFactory.create_my_channel(g_max=20e-12, tau=5.0)
   neuron.CableSettings.add_channel(my_channel)

   # Or directly
   from spine.cable_eq.channels import MyChannel
   my_channel = MyChannel(g_max=20e-12, tau=5.0)
   neuron.CableSettings.add_channel(my_channel)
   ```

**Key Design Principles:**
- All channels extend `ChannelModel` abstract class
- Use `GatingVariable` helper for SBDF2 time integration
- Voltage passed to kinetics in **mV**, but stored/used in **V**
- `compute_current()` returns current density in **V/s** for cable equation
- Leak channel should return `False` from `contributes_to_leak_balance()` to avoid circular dependency

#### Adding a New Exchange Mechanism

1. **Add flux computation** in `cytosol.py` or `er.py`:
   ```python
   def jMYCHANNEL(neuron, c, V):
       """Compute my channel flux."""
       # Access parameters
       g_max = neuron.cytExchangeParams['mychannel_gmax']
       Kd = neuron.cytExchangeParams['mychannel_Kd']

       # Compute flux
       flux = g_max * (c / (c + Kd)) * some_function(V)
       return flux
   ```

2. **Add to JPM or JER** aggregation:
   ```python
   if neuron.model['MYCHANNEL']:
       jmychannel = jMYCHANNEL(neuron, c, V)
       # Add to flux recording if needed
       if fluxes is not None and neuron.recorder.get('flux_mychannel', False):
           fluxes.mychannel = jmychannel.copy()
   ```

3. **Add parameters** in `cytosol.py` or `er.py`:
   ```python
   params = {
       'mychannel_gmax': 1e-12,
       'mychannel_Kd': 0.5e-15,
   }
   ```

4. **Add model flag** in `utils/settings.py`:
   ```python
   model = {
       'MYCHANNEL': False,
   }
   ```

### Memory Model

#### Shared Memory Architecture

SPINE uses Python's `multiprocessing.shared_memory` for zero-copy inter-process communication:

```python
# Creation (in main process)
self.shmC = shared_memory.SharedMemory(create=True, size=...)
self.sol.C = np.ndarray(shape, dtype=np.float64, buffer=self.shmC.buf)

# Access (in worker processes)
self.shmC = shared_memory.SharedMemory(name=neuron.C_name)
self.sol.C = np.ndarray(shape, dtype=np.float64, buffer=self.shmC.buf)
```

**Shared Arrays:**
- `C`, `C0` - Cytosolic calcium (current and history)
- `V`, `V0` - Membrane voltage (current and history)

**Non-Shared Arrays** (local to each process):
- `B`, `BE` - Buffers
- `CE` - ER calcium
- `IP3` - IP3 concentration

### Testing

#### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_synapse.py

# With coverage
pytest --cov=spine tests/
```

#### Writing Tests

```python
# tests/test_my_feature.py
import pytest
from spine.utils import NeuronModel
from spine.synapse import SynapseFactory

def test_my_receptor():
    neuron = NeuronModel()
    # ... setup ...
    syn = SynapseFactory.create_my_receptor(post_nodes=[0])
    neuron.synapse_instances = [syn]
    # ... assertions ...
    assert syn.receptor_model.param1 == expected_value
```

### Contributing

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/my-new-feature`
3. **Make changes** following the code style
4. **Add tests** for new functionality
5. **Run tests**: `pytest tests/`
6. **Commit changes**: `git commit -m "Add my new feature"`
7. **Push to branch**: `git push origin feature/my-new-feature`
8. **Submit a Pull Request**

---

## Units and Conventions

### Internal Computation Units

| Quantity | Unit | Symbol |
|----------|------|--------|
| Concentration | umol/um³ | μmol/μm³ |
| Length | um | μm |
| Time | s | s |
| Voltage | V | V |
| Flux | umol/(um²·s) | μmol/(μm²·s) |

### Output Units

| Quantity | Unit | Notes |
|----------|------|-------|
| Concentration | μM (micromolar) | Multiply internal by 1×10¹⁵ |
| Voltage | mV (millivolts) | Multiply internal by 1×10³ |

### Input File Units

| File Type | Expected Unit | Scaling Factor |
|-----------|---------------|----------------|
| SWC morphology | meters | `len_scale = 1.0e6` (m → μm) |
| Voltage traces | millivolts | `volt_scale = 1.0e-3` (mV → V) |

**To change scaling:**
```python
neuron.settings.len_scale = 1.0e6  # m to μm
neuron.settings.volt_scale = 1.0e-3  # mV to V
```

---

## Citation

If you use SPINE in your research, please cite:

```bibtex
@software{spine2025,
  author = {Miksis, Zachary M.},
  title = {SPINE: Simulator for Plasticity and Integrated Neuronal Events},
  year = {2025},
  url = {https://github.com/zmiksis/SPINE}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

Stable calcium wave parameters adapted from:
> P. Borole, "CalciumSim: Simulator for calcium dynamics on neuron graphs using dimensionally reduced model," MS thesis, Temple University, 2022.

---

## Support

- **Issues**: https://github.com/zmiksis/SPINE/issues
- **Discussions**: https://github.com/zmiksis/SPINE/discussions
- **Email**: miksis@temple.edu

---

**Last Updated**: January 2026
**Version**: 1.0.0
