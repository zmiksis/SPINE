"""Specific ion channel implementations.

This module contains concrete implementations of common ion channels:
- SodiumChannel: Fast Na channel (HH m³h model)
- PotassiumChannel: Delayed rectifier K (HH n⁴ model)
- CalciumChannel: Ca channel with Ca-dependent inactivation
- LeakChannel: Passive leak channel

Users can extend ChannelModel to create custom channels.
"""

import numpy as np
from .base import ChannelModel, GatingVariable


class SodiumChannel(ChannelModel):
    """Fast sodium channel (Hodgkin-Huxley m³h model).

    Voltage-gated sodium channel with activation (m) and inactivation (h) gates.
    Responsible for action potential upstroke.
    """

    def __init__(self, g_max=500e-12, E_rev=50e-3, Vt=-60e-3):
        """Initialize sodium channel.

        Args:
            g_max: Maximum conductance (S/um²)
            E_rev: Reversal potential (V)
            Vt: Threshold voltage for kinetics (V)
        """
        super().__init__('Na', g_max, E_rev)
        self.Vt = Vt
        self.m = GatingVariable('m', exponent=3)
        self.h = GatingVariable('h', exponent=1)

    def compute_current(self, V, C, Cm):
        """Compute I_Na = g_Na * m³ * h * (V - E_Na)."""
        return (self.g_max / Cm) * self.m.current**3 * self.h.current * (V - self.E_rev)

    def update_state(self, V, V0, dt):
        """Update m and h gating variables."""
        self.m.update_sbdf2(self._m_kinetics, V, V0, dt)
        self.h.update_sbdf2(self._h_kinetics, V, V0, dt)

    def _m_kinetics(self, m, V):
        """Activation gate kinetics: dm/dt = alpha_m * (1-m) - beta_m * m."""
        Vt = 1e3*self.Vt
        alpha = -1e3 * 0.32 * (V - Vt - 13.) / (np.exp(-(V - Vt - 13.)/4.) - 1.)
        beta = 1e3 * 0.28 * (V - Vt - 40.) / (np.exp((V - Vt - 40.)/5.) - 1.)
        return alpha * (1. - m) - beta * m

    def _h_kinetics(self, h, V):
        """Inactivation gate kinetics: dh/dt = alpha_h * (1-h) - beta_h * h."""
        Vt = 1e3*self.Vt
        alpha = 1e3 * 0.128 * np.exp(-(V - Vt - 17.)/18.)
        beta = 1e3 * 4.0 / (1. + np.exp(-(V - Vt - 40.)/5.))
        return alpha * (1. - h) - beta * h

    def initialize_state(self, V_rest, nnodes):
        """Find steady-state m∞ and h∞ at resting potential."""
        V_rest_mV = 1e3 * V_rest

        # Find m infinity using Newton-Raphson
        m_val = 0.5
        for _ in range(50):
            f = self._m_kinetics(m_val, V_rest_mV)
            if abs(f) < 1e-10:
                break
            df = (self._m_kinetics(m_val + 1e-6, V_rest_mV) - f) / 1e-6
            m_val = np.clip(m_val - f/df, 0, 1)

        # Find h infinity using Newton-Raphson
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
    """Delayed rectifier potassium channel (HH n⁴ model).

    Voltage-gated potassium channel with activation (n) gate.
    Responsible for action potential repolarization.
    """

    def __init__(self, g_max=50e-12, E_rev=-90e-3, Vt=-60e-3):
        """Initialize potassium channel.

        Args:
            g_max: Maximum conductance (S/um²)
            E_rev: Reversal potential (V)
            Vt: Threshold voltage for kinetics (V)
        """
        super().__init__('K', g_max, E_rev)
        self.Vt = Vt
        self.n = GatingVariable('n', exponent=4)

    def compute_current(self, V, C, Cm):
        """Compute I_K = g_K * n⁴ * (V - E_K)."""
        return (self.g_max / Cm) * self.n.current**4 * (V - self.E_rev)

    def update_state(self, V, V0, dt):
        """Update n gating variable."""
        self.n.update_sbdf2(self._n_kinetics, V, V0, dt)

    def _n_kinetics(self, n, V):
        """Activation gate kinetics: dn/dt = alpha_n * (1-n) - beta_n * n."""
        Vt = 1e3*self.Vt
        alpha = 1e3 * -0.032 * (V - Vt - 15.) / (np.exp(-(V - Vt - 15.)/5.) - 1.)
        beta = 1e3 * 0.5 * np.exp(-(V - Vt - 10.)/40.)
        return alpha * (1. - n) - beta * n

    def initialize_state(self, V_rest, nnodes):
        """Find steady-state n∞ at resting potential."""
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
    """Calcium channel with calcium-dependent inactivation.

    High-voltage activated calcium channel with q-gate activation
    and calcium-dependent inactivation (rho factor).
    """

    def __init__(self, g_max=5.16*6e-12, K_Ca=140e-18):
        """Initialize calcium channel.

        Args:
            g_max: Maximum conductance (S/um²)
            K_Ca: Half-maximal calcium for inactivation (umol/um³)
        """
        super().__init__('Ca', g_max, E_rev=None)  # Dynamic E_rev via Nernst
        self.K_Ca = K_Ca
        self.q = GatingVariable('q', exponent=1)

    def compute_current(self, V, C, Cm):
        """Compute I_Ca = g_Ca * q * ρ(C) * (V - E_Ca(C))."""
        # Calcium-dependent inactivation
        rho = self.K_Ca / (self.K_Ca + C)
        # Dynamic reversal potential
        E_Ca = self.get_reversal_potential(V, C)
        return (self.g_max / Cm) * self.q.current * rho * (V - E_Ca)

    def get_reversal_potential(self, V, C):
        """Compute Nernst potential for calcium.

        E_Ca = (RT/2F) * ln(Co/Ci)
        """
        co = 2000e-18  # External calcium (umol/um³)
        return 12.5e-3 * np.log(co / C)

    def update_state(self, V, V0, dt):
        """Update q gating variable."""
        self.q.update_sbdf2(self._q_kinetics, V, V0, dt)

    def _q_kinetics(self, q, V):
        """Activation gate kinetics: dq/dt = (σ(V) - q) / τ(V)."""
        tau = 7.8 / (np.exp((V + 6.)/16.) + np.exp(-(V + 6.)/16.))
        sigma = 1.0 / (1.0 + np.exp(-(V - 3)/8.))
        return (sigma - q) / tau

    def initialize_state(self, V_rest, nnodes):
        """Find steady-state q∞ at resting potential."""
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
    """Passive leak channel (no gating).

    Ohmic leak conductance that helps maintain resting potential.
    The reversal potential is balanced via set_leak_reversal().
    """

    def __init__(self, g_max=0.05e-12, E_rev=-70e-3):
        """Initialize leak channel.

        Args:
            g_max: Leak conductance (S/um²)
            E_rev: Reversal potential (V), typically set to V_rest
        """
        super().__init__('leak', g_max, E_rev)

    def compute_current(self, V, C, Cm):
        """Compute I_leak = g_leak * (V - E_leak)."""
        return (self.g_max / Cm) * (V - self.E_rev)

    def update_state(self, V, V0, dt):
        """No gating variables to update."""
        pass

    def initialize_state(self, V_rest, nnodes):
        """No state to initialize."""
        pass

    def contributes_to_leak_balance(self) -> bool:
        """Leak channel doesn't contribute to its own balancing."""
        return False
