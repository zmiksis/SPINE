"""Factory methods for easy synapse creation.

Provides convenient methods to create common synapse types without
needing to manually instantiate temporal patterns and receptor models.

Usage:
    from spine.synapse import SynapseFactory

    # Simple calcium pulse
    syn = SynapseFactory.create_calcium_pulse(nodes=[500], amplitude=2.5e-12, duration=1e-3)

    # AMPA synapse
    ampa = SynapseFactory.create_AMPA(post_nodes=[1000], g_max=2.7e-10)
"""

import numpy as np
from typing import List, Union, Callable
from spine.synapse.temporal import (ConstantPattern, LinearDecayPattern,
                                        ExponentialDecayPattern, PulseTrainPattern,
                                        CustomPattern)
from spine.synapse.receptors import (DirectCalciumRelease, DirectIP3Release,
                                         CalciumDependentRelease, AMPAReceptor,
                                         NMDAReceptor, GABAReceptor,
                                         CalciumModulatedAMPA, CustomReceptor,
                                         AMPAVModulatedReceptor)
from spine.synapse.calcium_coupled import (CalciumInfluxSynapse, IP3InfluxSynapse,
                                               CalciumCoupledSynapse, ChemicalSynapse)


class SynapseFactory:
    """Factory for creating common synapse types."""

    # ==================== Single Neuron Calcium Synapses ====================

    @staticmethod
    def create_calcium_pulse(nodes: Union[int, List[int]], amplitude: float,
                             duration: float, start_time: float = 0.0) -> CalciumInfluxSynapse:
        """Create pulsed calcium injection synapse.

        Args:
            nodes: Node index or list where calcium is injected
            amplitude: Flux amplitude (umol/(um^2*s))
            duration: Pulse duration (s)
            start_time: Time when pulse starts (s)

        Returns:
            CalciumInfluxSynapse instance

        Example:
            syn = SynapseFactory.create_calcium_pulse(
                nodes=[500], amplitude=2.5e-12, duration=1e-3
            )
        """
        pattern = ConstantPattern(amplitude=1.0, duration=duration, start_time=start_time)
        receptor = DirectCalciumRelease(max_flux=amplitude)
        return CalciumInfluxSynapse(nodes, pattern, receptor)

    @staticmethod
    def create_calcium_linear(nodes: Union[int, List[int]], amplitude: float,
                              duration: float, start_time: float = 0.0) -> CalciumInfluxSynapse:
        """Create linearly decaying calcium injection synapse.

        Args:
            nodes: Node index or list where calcium is injected
            amplitude: Initial flux amplitude (umol/(um^2*s))
            duration: Decay duration (s)
            start_time: Time when decay starts (s)

        Returns:
            CalciumInfluxSynapse instance

        Example:
            syn = SynapseFactory.create_calcium_linear(
                nodes=[500], amplitude=2.5e-12, duration=10e-3
            )
        """
        pattern = LinearDecayPattern(amplitude=1.0, duration=duration, start_time=start_time)
        receptor = DirectCalciumRelease(max_flux=amplitude)
        return CalciumInfluxSynapse(nodes, pattern, receptor)

    @staticmethod
    def create_calcium_exponential(nodes: Union[int, List[int]], amplitude: float,
                                    tau: float, start_time: float = 0.0) -> CalciumInfluxSynapse:
        """Create exponentially decaying calcium injection synapse.

        Args:
            nodes: Node index or list where calcium is injected
            amplitude: Initial flux amplitude (umol/(um^2*s))
            tau: Time constant (s)
            start_time: Time when decay starts (s)

        Returns:
            CalciumInfluxSynapse instance

        Example:
            syn = SynapseFactory.create_calcium_exponential(
                nodes=[700], amplitude=3e-12, tau=10e-3
            )
        """
        pattern = ExponentialDecayPattern(amplitude=1.0, tau=tau, start_time=start_time)
        receptor = DirectCalciumRelease(max_flux=amplitude)
        return CalciumInfluxSynapse(nodes, pattern, receptor)

    @staticmethod
    def create_calcium_train(nodes: Union[int, List[int]], amplitude: float,
                             pulse_duration: float, frequency: float,
                             n_pulses: int = None) -> CalciumInfluxSynapse:
        """Create calcium injection with pulse train.

        Args:
            nodes: Node index or list where calcium is injected
            amplitude: Flux amplitude (umol/(um^2*s))
            pulse_duration: Duration of each pulse (s)
            frequency: Pulse frequency (Hz)
            n_pulses: Number of pulses (None = infinite)

        Returns:
            CalciumInfluxSynapse instance

        Example:
            # 10 Hz stimulation, 1 ms pulses, 5 times
            syn = SynapseFactory.create_calcium_train(
                nodes=[500], amplitude=2.5e-12,
                pulse_duration=1e-3, frequency=10.0, n_pulses=5
            )
        """
        pattern = PulseTrainPattern(amplitude=1.0, pulse_duration=pulse_duration,
                                     frequency=frequency, n_pulses=n_pulses)
        receptor = DirectCalciumRelease(max_flux=amplitude)
        return CalciumInfluxSynapse(nodes, pattern, receptor)
    
    @staticmethod
    def create_AMPA_voltage_modulated(nodes: Union[int, List[int]],
                                        g_O2: float = 9.0e-12, g_O3: float = 15.0e-12,
                                        g_O4: float = 21.0e-12,
                                        E_rev: float = 0.0, n_receptors: int = 437,
                                        rho: float = 0.0) -> CalciumInfluxSynapse:
        """Create AMPA receptor with calcium-dependent facilitation.

        Combines AMPA kinetics with presynaptic calcium-dependent modulation,
        enabling short-term synaptic plasticity (facilitation).

        Args:
            post_nodes: Postsynaptic node indices
            g_O2: Conductance of state O2 (S/um^2)
            g_O3: Conductance of state O3 (S/um^2)
            g_O4: Conductance of state O4 (S/um^2)
            E_rev: Reversal potential (V)
            n_receptors: Number of receptors per um^2
            rho: Initial synapse state (0: depressed, 1: potentiated)

        Returns:
            CalciumInfluxSynapse with AMPAVModulatedReceptor receptor
        """
        pattern = ConstantPattern(amplitude=1.0, duration=1e99, start_time=0.0)
        receptor = AMPAVModulatedReceptor(
            g_O2=g_O2, g_O3=g_O3, g_O4=g_O4,
            E_rev=E_rev, n_receptors=n_receptors, rho=rho
        )
        return CalciumInfluxSynapse(nodes, pattern, receptor)

    # ==================== Single Neuron IP3 Synapses ====================

    @staticmethod
    def create_ip3_pulse(nodes: Union[int, List[int]], amplitude: float,
                         duration: float, start_time: float = 0.0) -> IP3InfluxSynapse:
        """Create pulsed IP3 injection synapse.

        Args:
            nodes: Node index or list where IP3 is injected
            amplitude: Flux amplitude (umol/(um^2*s))
            duration: Pulse duration (s)
            start_time: Time when pulse starts (s)

        Returns:
            IP3InfluxSynapse instance

        Example:
            syn = SynapseFactory.create_ip3_pulse(
                nodes=[500], amplitude=5e-12, duration=200e-3
            )
        """
        pattern = ConstantPattern(amplitude=1.0, duration=duration, start_time=start_time)
        receptor = DirectIP3Release(max_flux=amplitude)
        return IP3InfluxSynapse(nodes, pattern, receptor)

    @staticmethod
    def create_ip3_linear(nodes: Union[int, List[int]], amplitude: float,
                          duration: float, start_time: float = 0.0) -> IP3InfluxSynapse:
        """Create linearly decaying IP3 injection synapse.

        Args:
            nodes: Node index or list where IP3 is injected
            amplitude: Initial flux amplitude (umol/(um^2*s))
            duration: Decay duration (s)
            start_time: Time when decay starts (s)

        Returns:
            IP3InfluxSynapse instance
        """
        pattern = LinearDecayPattern(amplitude=1.0, duration=duration, start_time=start_time)
        receptor = DirectIP3Release(max_flux=amplitude)
        return IP3InfluxSynapse(nodes, pattern, receptor)

    # ==================== Inter-Neuron Calcium-Coupled Synapses ====================

    @staticmethod
    def create_calcium_coupled(post_nodes: Union[int, List[int]],
                               sensitivity: float = 1e3) -> CalciumCoupledSynapse:
        """Create calcium-dependent inter-neuron synapse.

        Flux proportional to presynaptic calcium concentration.
        Requires SynapseCommunicator for inter-neuron coupling.

        Args:
            post_nodes: Postsynaptic node indices
            sensitivity: Sensitivity to presynaptic calcium (s^-1)

        Returns:
            CalciumCoupledSynapse instance

        Example:
            syn = SynapseFactory.create_calcium_coupled(
                post_nodes=[1000], sensitivity=1e3
            )
            neuron2.synapse_instances = [syn]

            communicator = SynapseCommunicator([neuron1, neuron2])
            communicator.add_synapse_connection((0, 0), (1, 1000), weight=1200)
        """
        receptor = CalciumDependentRelease(sensitivity=sensitivity)
        return CalciumCoupledSynapse(post_nodes, receptor)

    # ==================== Inter-Neuron Chemical Synapses ====================

    @staticmethod
    def create_AMPA(post_nodes: Union[int, List[int]], g_max: float = 2.7e-10,
                    E_rev: float = 0.0, tau: float = 5.26e-3,
                    weight: float = 1.0) -> ChemicalSynapse:
        """Create AMPA glutamate receptor synapse (fast excitatory).

        Args:
            post_nodes: Postsynaptic node indices
            g_max: Maximum conductance (S/um^2)
            E_rev: Reversal potential (V)
            tau: Time constant for activation decay (s)
            weight: Synaptic weight (multiplier)

        Returns:
            ChemicalSynapse with AMPA receptor

        Example:
            ampa = SynapseFactory.create_AMPA(
                post_nodes=[100, 101, 102], g_max=3e-10, weight=1.5
            )
        """
        receptor = AMPAReceptor(g_max=g_max, E_rev=E_rev, tau=tau)
        return ChemicalSynapse(post_nodes, receptor, weight)

    @staticmethod
    def create_calcium_modulated_AMPA(post_nodes: Union[int, List[int]],
                                       g_max: float = 2.7e-10, E_rev: float = 0.0,
                                       tau: float = 5.26e-3, ca_sensitivity: float = 5.0,
                                       baseline_release: float = 0.5,
                                       ca_baseline: float = 0.05,
                                       weight: float = 1.0) -> ChemicalSynapse:
        """Create AMPA receptor with calcium-dependent facilitation.

        Combines AMPA kinetics with presynaptic calcium-dependent modulation,
        enabling short-term synaptic plasticity (facilitation).

        Release probability increases with presynaptic calcium:
        P_release = baseline * (1 + sensitivity * (C_pre - C_baseline))

        Args:
            post_nodes: Postsynaptic node indices
            g_max: Maximum conductance (S/um^2)
            E_rev: Reversal potential (V)
            tau: Time constant for activation decay (s)
            ca_sensitivity: Sensitivity to presynaptic calcium (μM^-1)
            baseline_release: Baseline release probability (0-1)
            ca_baseline: Baseline calcium concentration (μM)
            weight: Synaptic weight (multiplier)

        Returns:
            ChemicalSynapse with CalciumModulatedAMPA receptor

        Example:
            # High facilitation synapse (5x increase per μM calcium)
            ampa_fac = SynapseFactory.create_calcium_modulated_AMPA(
                post_nodes=[100],
                ca_sensitivity=5.0,    # Strong facilitation
                baseline_release=0.3   # Low baseline, high dynamic range
            )

            # Requires SynapseCommunicator to access presynaptic calcium:
            comm = SynapseCommunicator([neuron1, neuron2])
            comm.add_synapse_connection((0, 0), (1, 100))
        """
        receptor = CalciumModulatedAMPA(
            g_max=g_max, E_rev=E_rev, tau=tau,
            ca_sensitivity=ca_sensitivity,
            baseline_release=baseline_release,
            ca_baseline=ca_baseline
        )
        return ChemicalSynapse(post_nodes, receptor, weight)

    @staticmethod
    def create_NMDA(post_nodes: Union[int, List[int]], g_max: float = 1e-10,
                    E_rev: float = 0.0, tau_rise: float = 2e-3,
                    tau_decay: float = 80e-3, Mg_conc: float = 1.0,
                    weight: float = 1.0) -> ChemicalSynapse:
        """Create NMDA glutamate receptor synapse (slow excitatory).

        NMDA receptors have Mg2+ voltage-dependent block and
        slower kinetics than AMPA.

        Args:
            post_nodes: Postsynaptic node indices
            g_max: Maximum conductance (S/um^2)
            E_rev: Reversal potential (V)
            tau_rise: Rise time constant (s)
            tau_decay: Decay time constant (s)
            Mg_conc: Extracellular magnesium concentration (mM)
            weight: Synaptic weight (multiplier)

        Returns:
            ChemicalSynapse with NMDA receptor

        Example:
            nmda = SynapseFactory.create_NMDA(
                post_nodes=[200], g_max=2e-10, Mg_conc=1.5
            )
        """
        receptor = NMDAReceptor(g_max=g_max, E_rev=E_rev,
                                tau_rise=tau_rise, tau_decay=tau_decay,
                                Mg_conc=Mg_conc)
        return ChemicalSynapse(post_nodes, receptor, weight)

    @staticmethod
    def create_GABA(post_nodes: Union[int, List[int]], g_max: float = 5e-11,
                    E_rev: float = -0.070, tau: float = 10e-3,
                    weight: float = 1.0) -> ChemicalSynapse:
        """Create GABA_A receptor synapse (fast inhibitory).

        Args:
            post_nodes: Postsynaptic node indices
            g_max: Maximum conductance (S/um^2)
            E_rev: Reversal potential (V), typically ~-70 mV (inhibitory)
            tau: Time constant for activation decay (s)
            weight: Synaptic weight (multiplier)

        Returns:
            ChemicalSynapse with GABA receptor

        Example:
            gaba = SynapseFactory.create_GABA(
                post_nodes=[300, 301], g_max=8e-11, E_rev=-0.075
            )
        """
        receptor = GABAReceptor(g_max=g_max, E_rev=E_rev, tau=tau)
        return ChemicalSynapse(post_nodes, receptor, weight)

    # ==================== Custom Synapses ====================

    @staticmethod
    def create_custom_calcium(nodes: Union[int, List[int]],
                              flux_callback: Callable,
                              **kwargs) -> CalciumInfluxSynapse:
        """Create calcium synapse with custom temporal pattern.

        Args:
            nodes: Node index or list where calcium is injected
            flux_callback: Function(t) -> flux amplitude
            **kwargs: Additional parameters passed to callback

        Returns:
            CalciumInfluxSynapse with custom pattern

        Example:
            def my_pattern(t):
                return 2.5e-12 * np.sin(2*np.pi*10*t) if t < 0.1 else 0.0

            syn = SynapseFactory.create_custom_calcium(
                nodes=[500], flux_callback=my_pattern
            )
        """
        pattern = CustomPattern(flux_callback)
        receptor = DirectCalciumRelease(max_flux=1.0)
        return CalciumInfluxSynapse(nodes, pattern, receptor)

    @staticmethod
    def create_custom_coupled(post_nodes: Union[int, List[int]],
                              coupling_callback: Callable,
                              **kwargs) -> CalciumCoupledSynapse:
        """Create inter-neuron synapse with custom coupling logic.

        Args:
            post_nodes: Postsynaptic node indices
            coupling_callback: Function(pre_data, post_data, **kwargs) -> flux
            **kwargs: Parameters passed to coupling_callback

        Returns:
            CalciumCoupledSynapse with custom receptor

        Example:
            def my_coupling(pre_data, post_data, threshold=100e-18, gain=1e3):
                pre_C = pre_data.get('C', 0.0)
                if 1e-15 * pre_C > threshold:
                    return gain * (1e-15 * pre_C - threshold)
                return 0.0

            syn = SynapseFactory.create_custom_coupled(
                post_nodes=[1000],
                coupling_callback=my_coupling,
                threshold=150e-18, gain=2e3
            )
        """
        receptor = CustomReceptor(flux_callback=coupling_callback, **kwargs)
        return CalciumCoupledSynapse(post_nodes, receptor)
