"""Factory methods for creating ion channels.

Similar to SynapseFactory, this provides convenient methods to create
common ion channel types without needing to manually instantiate classes.

Usage:
    from spine.cable_eq import ChannelFactory

    # Create standard Hodgkin-Huxley channels
    na_channel = ChannelFactory.create_Na()
    k_channel = ChannelFactory.create_K()
    ca_channel = ChannelFactory.create_Ca()
    leak_channel = ChannelFactory.create_leak()

    # Add to neuron
    neuron.CableSettings.add_channel(na_channel)
    neuron.CableSettings.add_channel(k_channel)
    # ... etc
"""

from typing import Type
from .channels import SodiumChannel, PotassiumChannel, CalciumChannel, LeakChannel
from .base import ChannelModel


class ChannelFactory:
    """Factory for creating ion channels with sensible defaults."""

    @staticmethod
    def create_Na(g_max=500e-12, E_rev=50e-3, Vt=-60e-3):
        """Create fast sodium channel (Hodgkin-Huxley m³h model).

        Args:
            g_max: Maximum conductance (S/um²)
            E_rev: Reversal potential (V)
            Vt: Threshold voltage for kinetics (V)

        Returns:
            SodiumChannel instance

        Example:
            na = ChannelFactory.create_Na(g_max=800e-12)  # High density
        """
        return SodiumChannel(g_max, E_rev, Vt)

    @staticmethod
    def create_K(g_max=50e-12, E_rev=-90e-3, Vt=-60e-3):
        """Create delayed rectifier potassium channel (HH n⁴ model).

        Args:
            g_max: Maximum conductance (S/um²)
            E_rev: Reversal potential (V)
            Vt: Threshold voltage for kinetics (V)

        Returns:
            PotassiumChannel instance

        Example:
            k = ChannelFactory.create_K(g_max=100e-12)  # High density
        """
        return PotassiumChannel(g_max, E_rev, Vt)

    @staticmethod
    def create_Ca(g_max=5.16*6e-12, K_Ca=140e-18):
        """Create calcium channel with Ca-dependent inactivation.

        Args:
            g_max: Maximum conductance (S/um²)
            K_Ca: Half-maximal calcium for inactivation (umol/um³ = μM * 1e-15)

        Returns:
            CalciumChannel instance

        Example:
            ca = ChannelFactory.create_Ca(g_max=8e-12)  # L-type like
        """
        return CalciumChannel(g_max, K_Ca)

    @staticmethod
    def create_leak(g_max=0.05e-12, V_rest=-70e-3):
        """Create passive leak channel.

        Reversal potential set to V_rest. The set_leak_reversal() method
        will balance this to ensure resting equilibrium with other channels.

        Args:
            g_max: Leak conductance (S/um²)
            V_rest: Resting potential (V), used as initial E_leak

        Returns:
            LeakChannel instance

        Example:
            leak = ChannelFactory.create_leak(g_max=0.1e-12)  # Higher leak
        """
        return LeakChannel(g_max, E_rev=V_rest)

    @staticmethod
    def create_custom(channel_class: Type[ChannelModel], **kwargs):
        """Create custom channel from user-defined class.

        Allows users to extend the channel system without modifying pyCalSim.

        Args:
            channel_class: Class extending ChannelModel
            **kwargs: Arguments passed to channel_class constructor

        Returns:
            Instance of channel_class

        Example:
            from spine.cable_eq.base import ChannelModel

            class HChannel(ChannelModel):
                '''H-current (HCN channel).'''
                def __init__(self, g_max, E_rev=-30e-3):
                    super().__init__('H', g_max, E_rev)
                    # ... implement required methods

            h_channel = ChannelFactory.create_custom(
                HChannel, g_max=5e-12
            )
        """
        return channel_class(**kwargs)
