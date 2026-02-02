"""Abstract base classes for synapse architecture.

This module provides the foundation for the synapse system with three
key abstractions:

1. TemporalPattern: Defines WHEN/HOW signal is delivered (time-dependent amplitude)
2. ReceptorModel: Defines WHAT happens (receptor kinetics, flux computation)
3. Synapse: Combines temporal pattern + receptor model for complete synapse

The separation allows:
- Temporal patterns to be reused across different receptor types
- Receptor models to work with different temporal patterns
- Easy extension without modifying existing code
- Clean testing of individual components
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Union


class TemporalPattern(ABC):
    """Abstract base for temporal release patterns.

    Defines the time-dependent amplitude of synaptic activity.
    Temporal patterns are independent of the biological mechanism.

    Examples: constant pulse, exponential decay, burst patterns, custom functions
    """

    @abstractmethod
    def get_amplitude(self, t: float) -> float:
        """Return amplitude at time t.

        Args:
            t: Current time (seconds)

        Returns:
            Amplitude (dimensionless, typically 0-1 range, but can be > 1)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset temporal state (for repeated simulations)."""
        pass


class ReceptorModel(ABC):
    """Abstract base for synaptic receptor models.

    Defines the biological mechanism of synaptic transmission:
    - How presynaptic activity affects postsynaptic response
    - Receptor kinetics and gating variables
    - Flux computation based on pre/post-synaptic state

    Examples: AMPA, NMDA, GABA receptors, calcium-coupled release
    """

    @abstractmethod
    def compute_flux(self, pre_data: dict, post_data: dict) -> Union[float, np.ndarray]:
        """Compute synaptic flux based on pre/post-synaptic state.

        Args:
            pre_data: Presynaptic data dictionary
                For single neuron: may be empty or contain local state
                For inter-neuron: {'C': calcium, 'V': voltage, etc.}
            post_data: Postsynaptic data dictionary
                Contains neuron object, concentrations, parameters

        Returns:
            flux: Flux values (umol/(um^2*s) for calcium, V/s for voltage)
                  Can be scalar or array depending on receptor type
        """
        pass

    @abstractmethod
    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update receptor state variables (gating, etc.).

        Called once per timestep to update internal state (e.g., gating variables
        for voltage-dependent receptors).

        Args:
            dt: Time step (seconds)
            pre_data: Presynaptic data dictionary
            post_data: Postsynaptic data dictionary
        """
        pass


class Synapse(ABC):
    """Abstract base for all synapse types.

    Combines TemporalPattern + ReceptorModel to create a complete synapse.
    Handles the interface between the solver and the biological model.

    Subclasses define specific synapse types:
    - CalciumInfluxSynapse: Direct calcium injection (single neuron)
    - IP3InfluxSynapse: IP3 injection (single neuron)
    - CalciumCoupledSynapse: Calcium-dependent inter-neuron coupling
    - ChemicalSynapse: Neurotransmitter receptors (AMPA, NMDA, GABA)
    """

    def __init__(self, nodes: Union[int, List[int]],
                 temporal_pattern: TemporalPattern,
                 receptor_model: ReceptorModel):
        """Initialize synapse.

        Args:
            nodes: Node index or list of node indices where synapse acts
            temporal_pattern: TemporalPattern instance defining time course
            receptor_model: ReceptorModel instance defining mechanism
        """
        self.nodes = nodes if isinstance(nodes, list) else [nodes]
        self.temporal_pattern = temporal_pattern
        self.receptor_model = receptor_model

        # Domain marker for backward compatibility with jSYN()
        # Subclasses should set this appropriately
        self.domain = None

    @abstractmethod
    def compute_current(self, t: float, neuron, comm=None, comm_iter=None) -> np.ndarray:
        """Compute synaptic current/flux at time t.

        This is the main interface called by the solver (cytosol.py:jSYN()).

        Args:
            t: Current time (seconds)
            neuron: NeuronModel instance (postsynaptic neuron)
            comm: SynapseCommunicator instance (None for single neuron)
            comm_iter: Postsynaptic neuron index in communicator

        Returns:
            flux: Array of flux values at all nodes (same shape as neuron.sol.C or neuron.sol.V)
        """
        pass

    def update_state(self, dt: float, neuron, comm=None, comm_iter=None):
        """Update synapse state (optional, default does nothing).

        Override if synapse has state that needs updating beyond receptor state.

        Args:
            dt: Time step (seconds)
            neuron: NeuronModel instance
            comm: SynapseCommunicator instance
            comm_iter: Postsynaptic neuron index
        """
        pass
