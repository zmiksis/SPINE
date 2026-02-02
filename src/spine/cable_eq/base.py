"""Abstract base classes for extensible ion channel system.

This module provides the foundation for the channel factory pattern,
similar to the synapse system architecture.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional


class ChannelModel(ABC):
    """Abstract base for ion channel models.

    All ion channels (Na, K, Ca, leak, custom) should extend this class
    and implement the required methods. This allows easy addition of new
    channel types without modifying existing code.
    """

    def __init__(self, name: str, g_max: float, E_rev: float = None):
        """Initialize channel.

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
        """Update gating variables using SBDF2.

        Args:
            V: Current voltage (mV)
            V0: Previous voltage (mV)
            dt: Time step (s)
        """
        pass

    @abstractmethod
    def initialize_state(self, V_rest: float, nnodes: int):
        """Initialize gating variables at resting potential.

        Args:
            V_rest: Resting potential (mV)
            nnodes: Number of nodes
        """
        pass

    def get_reversal_potential(self, V: np.ndarray, C: np.ndarray = None) -> float:
        """Get reversal potential (constant or dynamic).

        Override for channels with dynamic reversal (e.g., Ca via Nernst).

        Args:
            V: Voltage (for context, not always used)
            C: Calcium concentration (for Nernst potential)

        Returns:
            Reversal potential (V)
        """
        return self.E_rev

    def contributes_to_leak_balance(self) -> bool:
        """Whether this channel is included in leak balancing calculation.

        Returns True for most channels. Leak channel should return False
        to avoid circular dependency.

        Returns:
            True if channel contributes to leak balance, False otherwise
        """
        return self.name != 'leak'


class GatingVariable:
    """Helper class for managing individual gating variables.

    Encapsulates the state and history needed for SBDF2 time integration.
    """

    def __init__(self, name: str, exponent: int = 1):
        """Initialize gating variable.

        Args:
            name: Variable name (e.g., 'n', 'm', 'h')
            exponent: Power in conductance formula (e.g., n^4 has exponent=4)
        """
        self.name = name
        self.exponent = exponent
        self.current = None  # Current value
        self.history = None  # Previous timestep value

    def initialize(self, value: float, nnodes: int):
        """Initialize at steady state.

        Args:
            value: Steady-state value (0-1)
            nnodes: Number of nodes
        """
        self.current = value * np.ones(nnodes)
        self.history = value * np.ones(nnodes)

    def update_sbdf2(self, kinetics_fn, V: np.ndarray, V0: np.ndarray, dt: float):
        """Update using second-order backward differentiation formula.

        Args:
            kinetics_fn: Function(state, V) -> dstate/dt
            V: Current voltage (mV)
            V0: Previous voltage (mV)
            dt: Time step (s)
        """
        dstate = kinetics_fn(self.current, V)
        dstate0 = kinetics_fn(self.history, V0)

        tmp = self.current.copy()
        self.current = (4./3.)*self.current + (4./3.)*dstate*dt \
                     - (1./3.)*self.history - (2./3.)*dstate0*dt
        self.current = np.clip(self.current, 0, 1)
        self.history = tmp
