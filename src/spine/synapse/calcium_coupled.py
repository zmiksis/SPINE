"""Concrete synapse classes for calcium and IP3 dynamics.

This module provides ready-to-use synapse classes for:
1. Single-neuron synthetic stimulation (calcium, IP3)
2. Inter-neuron calcium-coupled communication
3. Inter-neuron chemical synapses (AMPA, NMDA, GABA)

These classes combine temporal patterns and receptor models to create
complete synapses that work with the solver (cytosol.py:jSYN()).
"""

import numpy as np
from typing import List, Union
from spine.synapse.base import Synapse, TemporalPattern, ReceptorModel
from spine.synapse.temporal import ConstantPattern


class CalciumInfluxSynapse(Synapse):
    """Synapse that directly injects calcium into cytosol (single neuron).

    Used for synthetic stimulation to trigger calcium waves without
    inter-neuron coupling. The temporal pattern controls when/how
    calcium is injected.

    Example:
        from spine.synapse.temporal import ConstantPattern
        from spine.synapse.receptors import DirectCalciumRelease

        pattern = ConstantPattern(amplitude=1.0, duration=1e-3)
        receptor = DirectCalciumRelease(max_flux=2.5e-12)
        syn = CalciumInfluxSynapse(nodes=[500], temporal_pattern=pattern,
                                    receptor_model=receptor)
    """

    def __init__(self, nodes: Union[int, List[int]],
                 temporal_pattern: TemporalPattern,
                 receptor_model: ReceptorModel = None,
                 max_flux: float = None):
        """Initialize calcium influx synapse.

        Args:
            nodes: Node index or list where calcium is injected
            temporal_pattern: TemporalPattern defining time course
            receptor_model: ReceptorModel for flux (default: DirectCalciumRelease)
            max_flux: Maximum flux if receptor_model not provided (umol/(um^2*s))
        """
        # If no receptor provided, create default
        if receptor_model is None:
            from spine.synapse.receptors import DirectCalciumRelease
            flux = max_flux if max_flux is not None else 1e-11
            receptor_model = DirectCalciumRelease(max_flux=flux)

        super().__init__(nodes, temporal_pattern, receptor_model)
        self.domain = 'cytosol'  # For backward compatibility with jSYN()

    def compute_current(self, t: float, neuron, comm=None, comm_iter=None) -> np.ndarray:
        """Compute calcium flux at time t.

        Args:
            t: Current time (s)
            neuron: NeuronModel instance
            comm: Not used (single neuron)
            comm_iter: Not used

        Returns:
            flux: Array of calcium flux values (umol/(um^2*s))
        """
        flux = np.zeros_like(neuron.sol.C)

        # Get time-dependent amplitude
        amplitude = self.temporal_pattern.get_amplitude(t)

        if amplitude > 0:
            # Compute base flux from receptor
            base_flux = self.receptor_model.compute_flux({}, {})

            # Apply to specified nodes
            for node in self.nodes:
                flux[node] = amplitude * base_flux

        return flux


class IP3InfluxSynapse(Synapse):
    """Synapse that directly injects IP3 (single neuron).

    Used for synthetic IP3 stimulation to trigger IP3-mediated
    calcium release from ER.

    Example:
        from spine.synapse.temporal import LinearDecayPattern
        from spine.synapse.receptors import DirectIP3Release

        pattern = LinearDecayPattern(amplitude=1.0, duration=200e-3)
        receptor = DirectIP3Release(max_flux=5e-12)
        syn = IP3InfluxSynapse(nodes=[500], temporal_pattern=pattern,
                                receptor_model=receptor)
    """

    def __init__(self, nodes: Union[int, List[int]],
                 temporal_pattern: TemporalPattern,
                 receptor_model: ReceptorModel = None,
                 max_flux: float = None):
        """Initialize IP3 influx synapse.

        Args:
            nodes: Node index or list where IP3 is injected
            temporal_pattern: TemporalPattern defining time course
            receptor_model: ReceptorModel for flux (default: DirectIP3Release)
            max_flux: Maximum flux if receptor_model not provided (umol/(um^2*s))
        """
        # If no receptor provided, create default
        if receptor_model is None:
            from spine.synapse.receptors import DirectIP3Release
            flux = max_flux if max_flux is not None else 5e-12
            receptor_model = DirectIP3Release(max_flux=flux)

        super().__init__(nodes, temporal_pattern, receptor_model)
        self.domain = 'ip3'  # For backward compatibility with jSYN()

    def compute_current(self, t: float, neuron, comm=None, comm_iter=None) -> np.ndarray:
        """Compute IP3 flux at time t.

        Args:
            t: Current time (s)
            neuron: NeuronModel instance
            comm: Not used (single neuron)
            comm_iter: Not used

        Returns:
            flux: Array of IP3 flux values (umol/(um^2*s))
        """
        flux = np.zeros_like(neuron.sol.IP3)

        # Get time-dependent amplitude
        amplitude = self.temporal_pattern.get_amplitude(t)

        if amplitude > 0:
            # Compute base flux from receptor
            base_flux = self.receptor_model.compute_flux({}, {})

            # Apply to specified nodes
            for node in self.nodes:
                flux[node] = amplitude * base_flux

        return flux


class CalciumCoupledSynapse(Synapse):
    """Inter-neuron synapse with calcium-dependent release.

    Flux from presynaptic neuron's calcium concentration.
    Used for calcium-based inter-neuron communication via communicator.

    Requires SynapseCommunicator to access presynaptic calcium.

    Example:
        syn = CalciumCoupledSynapse(post_nodes=[1000], sensitivity=1e3)
        neuron2.synapse_instances = [syn]

        communicator = SynapseCommunicator([neuron1, neuron2])
        communicator.add_synapse_connection((0, 0), (1, 1000), weight=1200)
    """

    def __init__(self, post_nodes: Union[int, List[int]],
                 receptor_model: ReceptorModel = None,
                 sensitivity: float = None):
        """Initialize calcium-coupled synapse.

        Args:
            post_nodes: Postsynaptic node indices
            receptor_model: ReceptorModel (default: CalciumDependentRelease)
            sensitivity: Sensitivity if receptor_model not provided (s^-1)
        """
        # If no receptor provided, create default
        if receptor_model is None:
            from spine.synapse.receptors import CalciumDependentRelease
            sens = sensitivity if sensitivity is not None else 1e3
            receptor_model = CalciumDependentRelease(sensitivity=sens)

        # Always active (temporal pattern handled by presynaptic calcium)
        temporal = ConstantPattern(amplitude=1.0, duration=np.inf)

        super().__init__(post_nodes, temporal, receptor_model)
        self.domain = 'receptor'  # For backward compatibility with jSYN()

    def compute_current(self, t: float, neuron, comm=None, comm_iter=None) -> np.ndarray:
        """Compute calcium flux based on presynaptic calcium.

        Args:
            t: Current time (s)
            neuron: NeuronModel instance (postsynaptic)
            comm: SynapseCommunicator instance (required)
            comm_iter: Postsynaptic neuron index

        Returns:
            flux: Array of calcium flux values (umol/(um^2*s))
        """
        if comm is None:
            # No communicator - return zero
            return np.zeros_like(neuron.sol.C)

        flux = np.zeros_like(neuron.sol.C)

        # Get presynaptic calcium from communicator
        pre_C_list, post_nodes_list = comm.get_pre_C0(comm_iter)

        for pre_C, post_nodes in zip(pre_C_list, post_nodes_list):
            # Package data for receptor
            pre_data = {'C': pre_C}  # μM
            post_data = {'cceq': neuron.settings.cceq}  # umol/um^3

            # Compute flux from receptor model
            flux_val = self.receptor_model.compute_flux(pre_data, post_data)

            # Apply to postsynaptic nodes
            if isinstance(post_nodes, (list, np.ndarray)):
                for node in post_nodes:
                    flux[node] += flux_val
            else:
                flux[post_nodes] += flux_val

        return flux


class ChemicalSynapse(Synapse):
    """Neurotransmitter-based synapse (AMPA, NMDA, GABA).

    Used for voltage-based synaptic transmission between neurons.
    Receptors have gating kinetics that respond to presynaptic voltage.

    Requires SynapseCommunicator to access presynaptic voltage.

    Example:
        from spine.synapse.receptors import AMPAReceptor

        receptor = AMPAReceptor(g_max=2.7e-10, E_rev=0.0)
        syn = ChemicalSynapse(post_nodes=[1000], receptor_model=receptor, weight=1.0)

        neuron2.synapse_instances = [syn]

        communicator = SynapseCommunicator([neuron1, neuron2])
        communicator.add_synapse_connection((0, 0), (1, 1000), weight=1200)
    """

    def __init__(self, post_nodes: Union[int, List[int]],
                 receptor_model: ReceptorModel,
                 weight: float = 1.0):
        """Initialize chemical synapse.

        Args:
            post_nodes: Postsynaptic node indices
            receptor_model: ReceptorModel (AMPA, NMDA, GABA, etc.)
            weight: Synaptic weight (multiplier)
        """
        # Always active (temporal pattern handled by receptor kinetics)
        temporal = ConstantPattern(amplitude=1.0, duration=np.inf)

        super().__init__(post_nodes, temporal, receptor_model)
        self.weight = weight
        self.domain = 'voltage'  # Affects voltage dynamics

    def compute_current(self, t: float, neuron, comm=None, comm_iter=None) -> np.ndarray:
        """Compute synaptic current based on presynaptic voltage and calcium.

        Args:
            t: Current time (s)
            neuron: NeuronModel instance (postsynaptic)
            comm: SynapseCommunicator instance (required)
            comm_iter: Postsynaptic neuron index

        Returns:
            current: Array of current values (V/s for voltage equation)
        """
        if comm is None:
            # No communicator - return zero
            return np.zeros_like(neuron.sol.V)

        current = np.zeros_like(neuron.sol.V)

        # Get presynaptic voltage and calcium from communicator
        pre_V_list, post_nodes_list_V = comm.get_pre_V(comm_iter)
        pre_C_list, post_nodes_list_C = comm.get_pre_C0(comm_iter)

        # Both should have same structure, but verify
        for (pre_V, post_nodes_V), (pre_C, post_nodes_C) in zip(
            zip(pre_V_list, post_nodes_list_V),
            zip(pre_C_list, post_nodes_list_C)
        ):
            # Package data for receptor (includes both voltage and calcium)
            pre_data = {
                'V': 1e-3 * pre_V,  # Convert mV to V
                'C': pre_C  # Already in μM
            }

            # Postsynaptic data depends on node
            if isinstance(post_nodes_V, (list, np.ndarray)):
                for node in post_nodes_V:
                    post_data = {
                        'V': 1e-3 * neuron.sol.V[node],  # Convert mV to V
                        'C': neuron.sol.C[node],  # Postsynaptic calcium (μM)
                        'Cm': neuron.settings.Cm,
                        'cceq': neuron.settings.cceq  # Baseline concentration
                    }
                    current_val = self.receptor_model.compute_flux(pre_data, post_data)
                    current[node] += self.weight * current_val
            else:
                post_data = {
                    'V': 1e-3 * neuron.sol.V[post_nodes_V],
                    'C': neuron.sol.C[post_nodes_V],  # Postsynaptic calcium (μM)
                    'Cm': neuron.settings.Cm,
                    'cceq': neuron.settings.cceq
                }
                current_val = self.receptor_model.compute_flux(pre_data, post_data)
                current[post_nodes_V] += self.weight * current_val

        return current

    def update_state(self, dt: float, neuron, comm=None, comm_iter=None):
        """Update receptor gating variables with voltage and calcium data.

        Args:
            dt: Time step (s)
            neuron: NeuronModel instance
            comm: SynapseCommunicator instance
            comm_iter: Postsynaptic neuron index
        """
        if comm is None:
            return

        # Get presynaptic voltage and calcium
        pre_V_list, post_nodes_list_V = comm.get_pre_V(comm_iter)
        pre_C_list, post_nodes_list_C = comm.get_pre_C0(comm_iter)

        for (pre_V, post_nodes_V), (pre_C, post_nodes_C) in zip(
            zip(pre_V_list, post_nodes_list_V),
            zip(pre_C_list, post_nodes_list_C)
        ):
            # Package both voltage and calcium for receptor
            pre_data = {
                'V': 1e-3 * pre_V,  # Convert mV to V
                'C': pre_C  # Already in μM
            }

            # Postsynaptic data (for calcium-dependent receptor models)
            if isinstance(post_nodes_V, (list, np.ndarray)):
                # Use first node for update (synaptic state is shared)
                post_data = {
                    'V': 1e-3 * neuron.sol.V[post_nodes_V[0]] if hasattr(post_nodes_V, '__iter__') else 1e-3 * neuron.sol.V[post_nodes_V],
                    'C': neuron.sol.C[post_nodes_V[0]] if hasattr(post_nodes_V, '__iter__') else neuron.sol.C[post_nodes_V],
                    'cceq': neuron.settings.cceq
                }
            else:
                post_data = {
                    'V': 1e-3 * neuron.sol.V[post_nodes_V],
                    'C': neuron.sol.C[post_nodes_V],
                    'cceq': neuron.settings.cceq
                }

            # Update receptor state
            self.receptor_model.update_state(dt, pre_data, post_data)
