"""Synapse module for pyCalSim.

Provides classes for defining synaptic inputs and inter-neuron coupling.

New Architecture (recommended):
    - SynapseFactory: Easy creation of common synapse types
    - Base classes: Synapse, TemporalPattern, ReceptorModel
    - Concrete synapses: CalciumInfluxSynapse, IP3InfluxSynapse, etc.
    - Receptors: AMPA, NMDA, GABA, etc.

Legacy (backward compatible):
    - synapse: Original synapse class (still supported)
    - SynapseCommunicator: Inter-neuron communication (unchanged)

Example (new):
    from spine.synapse import SynapseFactory

    syn = SynapseFactory.create_calcium_pulse(nodes=[500], amplitude=2.5e-12, duration=1e-3)
    ampa = SynapseFactory.create_AMPA(post_nodes=[1000])

Example (legacy):
    from spine.synapse import synapse

    syn = synapse()
    syn.node = [500]
    syn.j = 2.5e-12
"""

# Communicator (unchanged)
from .communicator import SynapseCommunicator

# Legacy synapse class (backward compatibility)
from .model import synapse

# Factory (recommended entry point)
from .factory import SynapseFactory

# Base classes (for advanced users)
from .base import Synapse, TemporalPattern, ReceptorModel

# Temporal patterns
from .temporal import (
    ConstantPattern,
    LinearDecayPattern,
    ExponentialDecayPattern,
    PulseTrainPattern,
    CustomPattern
)

# Receptors
from .receptors import (
    DirectCalciumRelease,
    DirectIP3Release,
    CalciumDependentRelease,
    AMPAReceptor,
    NMDAReceptor,
    GABAReceptor,
    CalciumModulatedAMPA,
    CustomReceptor,
    AMPAVModulatedReceptor
)

# Concrete synapse types
from .calcium_coupled import (
    CalciumInfluxSynapse,
    IP3InfluxSynapse,
    CalciumCoupledSynapse,
    ChemicalSynapse
)

__all__ = [
    # Communicator
    'SynapseCommunicator',

    # Legacy
    'synapse',

    # Factory (main entry point)
    'SynapseFactory',

    # Base classes
    'Synapse',
    'TemporalPattern',
    'ReceptorModel',

    # Temporal patterns
    'ConstantPattern',
    'LinearDecayPattern',
    'ExponentialDecayPattern',
    'PulseTrainPattern',
    'CustomPattern',

    # Receptors
    'DirectCalciumRelease',
    'DirectIP3Release',
    'CalciumDependentRelease',
    'AMPAReceptor',
    'NMDAReceptor',
    'GABAReceptor',
    'CalciumModulatedAMPA',
    'CustomReceptor',
    'AMPAVModulatedReceptor',
    
    # Concrete synapses
    'CalciumInfluxSynapse',
    'IP3InfluxSynapse',
    'CalciumCoupledSynapse',
    'ChemicalSynapse',
]