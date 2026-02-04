"""Receptor model classes for synaptic transmission.

Receptor models define the biological mechanism of synaptic transmission.
They are independent of temporal patterns and can be combined with any
temporal pattern class.

Available receptors:
- DirectCalciumRelease: Simple calcium influx (for synthetic stimulation)
- DirectIP3Release: Simple IP3 influx (for synthetic stimulation)
- CalciumDependentRelease: Flux proportional to presynaptic calcium
- AMPAReceptor: AMPA glutamate receptor (fast excitatory)
- NMDAReceptor: NMDA glutamate receptor (slow excitatory, voltage-dependent)
- GABAReceptor: GABA_A receptor (fast inhibitory)
- CalciumModulatedAMPA: AMPA with calcium-dependent facilitation (plasticity)
- CustomReceptor: User-defined receptor via callback

Note: All chemical synapses (AMPA/NMDA/GABA) now have access to both voltage
and calcium data in pre_data and post_data dictionaries, enabling calcium-
dependent modulation of synaptic strength.
"""

import numpy as np
import copy
from spine.synapse.base import ReceptorModel


class DirectCalciumRelease(ReceptorModel):
    """Direct calcium influx for synthetic stimulation.

    Simple constant flux - used for injecting calcium into a single neuron
    to trigger waves without inter-neuron coupling.

    Example:
        receptor = DirectCalciumRelease(max_flux=2.5e-12)
    """

    def __init__(self, max_flux: float = 1e-11):
        """Initialize direct calcium release.

        Args:
            max_flux: Maximum flux (umol/(um^2*s))
        """
        self.max_flux = max_flux

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Return constant flux."""
        return self.max_flux

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """No state to update."""
        pass


class DirectIP3Release(ReceptorModel):
    """Direct IP3 influx for synthetic stimulation.

    Simple constant flux - used for injecting IP3 into a single neuron.

    Example:
        receptor = DirectIP3Release(max_flux=5e-12)
    """

    def __init__(self, max_flux: float = 5e-12):
        """Initialize direct IP3 release.

        Args:
            max_flux: Maximum flux (umol/(um^2*s))
        """
        self.max_flux = max_flux

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Return constant flux."""
        return self.max_flux

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """No state to update."""
        pass


class CalciumDependentRelease(ReceptorModel):
    """Calcium-dependent release for inter-neuron coupling.

    Flux proportional to presynaptic calcium concentration.
    Used for calcium-based inter-neuron communication.

    Flux = sensitivity * (C_pre - C_baseline)

    Example:
        receptor = CalciumDependentRelease(sensitivity=1e3)
    """

    def __init__(self, sensitivity: float = 1e3):
        """Initialize calcium-dependent release.

        Args:
            sensitivity: Sensitivity to presynaptic calcium (s^-1)
        """
        self.sensitivity = sensitivity

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Compute flux proportional to presynaptic calcium."""
        pre_C = pre_data.get('C', 0.0)  # μM
        baseline = post_data.get('cceq', 50e-18)  # umol/um^3

        # Flux proportional to difference from baseline
        flux = self.sensitivity * (1e-15 * pre_C - baseline)
        return flux

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """No state to update."""
        pass


class AMPAReceptor(ReceptorModel):
    """AMPA glutamate receptor with simple kinetics.

    Fast excitatory synapse with single gating variable.
    Used for rapid synaptic transmission.

    I = g_max * s * (V - E_rev)

    where s follows: ds/dt = alpha * f(V_pre) * (1-s) - s/tau

    Example:
        receptor = AMPAReceptor(g_max=2.7e-10, E_rev=0.0, tau=5e-3)
    """

    def __init__(self, g_max: float = 2.7e-10, E_rev: float = 0.0,
                 tau: float = 5.26e-3):
        """Initialize AMPA receptor.

        Args:
            g_max: Maximum conductance (S/um^2)
            E_rev: Reversal potential (V)
            tau: Time constant for activation decay (s)
        """
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau = tau
        self.s = 0.0    # Activation variable
        self.s0 = 0.0   # Previous activation (for SBDF2)

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Compute AMPA current.

        Args:
            pre_data: {'V': presynaptic voltage (V)}
            post_data: {'V': postsynaptic voltage (V), 'Cm': membrane capacitance}

        Returns:
            Current (V/s for voltage equation)
        """
        V_post = post_data.get('V', -0.072)  # V
        Cm = post_data.get('Cm', 0.01e-12)  # F/um^2

        # I = g * s * (V - E_rev), converted to V/s for cable equation
        return (self.g_max / Cm) * self.s * (V_post - self.E_rev)

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update activation variable with SBDF2.

        Args:
            dt: Time step (s)
            pre_data: {'V': presynaptic voltage (V)}
            post_data: Not used
        """
        V_pre = pre_data.get('V', -0.072)  # V

        # Activation based on presynaptic voltage (sigmoid)
        alpha = 0.55
        beta = 4.0e-3  # V
        s_inf = alpha * (1.0 + np.tanh(V_pre / beta))

        # SBDF2 time stepping
        ds = (s_inf - self.s) / self.tau
        ds0 = (s_inf - self.s0) / self.tau

        s_new = (4./3.)*self.s + (4./3.)*ds*dt - (1./3.)*self.s0 - (2./3.)*ds0*dt
        self.s0 = self.s
        self.s = np.clip(s_new, 0.0, 1.0)


class NMDAReceptor(ReceptorModel):
    """NMDA glutamate receptor with Mg2+ voltage dependence.

    Slow excitatory synapse with dual exponential kinetics and
    magnesium block that depends on postsynaptic voltage.

    I = g_max * (B - A) * B_Mg(V) * (V - E_rev)

    where:
    - A, B are fast/slow gating variables
    - B_Mg(V) = 1 / (1 + [Mg]exp(-0.062*V) / 3.57) is Mg block

    Example:
        receptor = NMDAReceptor(g_max=1e-10, E_rev=0.0, Mg_conc=1.0)
    """

    def __init__(self, g_max: float = 1e-10, E_rev: float = 0.0,
                 tau_rise: float = 2e-3, tau_decay: float = 80e-3,
                 Mg_conc: float = 1.0):
        """Initialize NMDA receptor.

        Args:
            g_max: Maximum conductance (S/um^2)
            E_rev: Reversal potential (V)
            tau_rise: Rise time constant (s)
            tau_decay: Decay time constant (s)
            Mg_conc: Extracellular magnesium concentration (mM)
        """
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.Mg_conc = Mg_conc
        self.A = 0.0  # Fast variable
        self.B = 0.0  # Slow variable

    def _mg_block(self, V):
        """Magnesium block factor (Jahr & Stevens, 1990)."""
        return 1.0 / (1.0 + (self.Mg_conc / 3.57) * np.exp(-0.062 * V * 1e3))

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Compute NMDA current with Mg2+ block.

        Args:
            pre_data: {'V': presynaptic voltage (V)}
            post_data: {'V': postsynaptic voltage (V), 'Cm': membrane capacitance}

        Returns:
            Current (V/s for voltage equation)
        """
        V_post = post_data.get('V', -0.072)  # V
        Cm = post_data.get('Cm', 0.01e-12)  # F/um^2

        g = self.g_max * (self.B - self.A) * self._mg_block(V_post)
        return (g / Cm) * (V_post - self.E_rev)

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update dual exponential kinetics.

        Args:
            dt: Time step (s)
            pre_data: {'V': presynaptic voltage (V)}
            post_data: Not used
        """
        V_pre = pre_data.get('V', -0.072)

        # Simple spike detection (threshold crossing)
        spike = 1.0 if V_pre > -0.02 else 0.0  # Spike if V > -20mV

        # Update fast component
        dA = -self.A / self.tau_rise + spike
        self.A += dA * dt
        self.A = max(0.0, self.A)  # Non-negative

        # Update slow component
        dB = -self.B / self.tau_decay + spike
        self.B += dB * dt
        self.B = max(0.0, self.B)  # Non-negative


class GABAReceptor(ReceptorModel):
    """GABA_A receptor (inhibitory).

    Fast inhibitory synapse with single gating variable.
    Reversal potential is typically near resting potential or below.

    I = g_max * s * (V - E_rev)

    Example:
        receptor = GABAReceptor(g_max=5e-11, E_rev=-0.070, tau=10e-3)
    """

    def __init__(self, g_max: float = 5e-11, E_rev: float = -0.070,
                 tau: float = 10e-3):
        """Initialize GABA receptor.

        Args:
            g_max: Maximum conductance (S/um^2)
            E_rev: Reversal potential (V), typically ~-70 mV (inhibitory)
            tau: Time constant for activation decay (s)
        """
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau = tau
        self.s = 0.0
        self.s0 = 0.0

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Compute GABA current.

        Args:
            pre_data: {'V': presynaptic voltage (V)}
            post_data: {'V': postsynaptic voltage (V), 'Cm': membrane capacitance}

        Returns:
            Current (V/s for voltage equation)
        """
        V_post = post_data.get('V', -0.072)
        Cm = post_data.get('Cm', 0.01e-12)

        return (self.g_max / Cm) * self.s * (V_post - self.E_rev)

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update activation variable with SBDF2.

        Args:
            dt: Time step (s)
            pre_data: {'V': presynaptic voltage (V)}
            post_data: Not used
        """
        V_pre = pre_data.get('V', -0.072)

        # Sigmoid activation
        s_inf = 1.0 / (1.0 + np.exp(-50.0 * (V_pre + 0.02)))

        # SBDF2 time stepping
        ds = (s_inf - self.s) / self.tau
        ds0 = (s_inf - self.s0) / self.tau

        s_new = (4./3.)*self.s + (4./3.)*ds*dt - (1./3.)*self.s0 - (2./3.)*ds0*dt
        self.s0 = self.s
        self.s = np.clip(s_new, 0.0, 1.0)


class CalciumModulatedAMPA(ReceptorModel):
    """AMPA receptor with calcium-dependent facilitation.

    Combines standard AMPA kinetics with presynaptic calcium-dependent
    facilitation, modeling short-term synaptic plasticity.

    Facilitation: release_prob = baseline * (1 + sensitivity * (C_pre - C_baseline))
    I = g_max * s * release_prob * (V - E_rev)

    This enables modeling of:
    - Paired-pulse facilitation
    - Presynaptic calcium buildup effects
    - Activity-dependent synaptic strength

    Example:
        receptor = CalciumModulatedAMPA(
            g_max=2.7e-10,
            E_rev=0.0,
            ca_sensitivity=5.0,  # Higher = more facilitation
            baseline_release=0.5
        )
    """

    def __init__(self, g_max: float = 2.7e-10, E_rev: float = 0.0,
                 tau: float = 5.26e-3, ca_sensitivity: float = 5.0,
                 baseline_release: float = 0.5, ca_baseline: float = 0.05):
        """Initialize calcium-modulated AMPA receptor.

        Args:
            g_max: Maximum conductance (S/um^2)
            E_rev: Reversal potential (V)
            tau: Time constant for activation decay (s)
            ca_sensitivity: Sensitivity to presynaptic calcium (μM^-1)
            baseline_release: Baseline release probability (0-1)
            ca_baseline: Baseline calcium concentration (μM)
        """
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau = tau
        self.ca_sensitivity = ca_sensitivity
        self.baseline_release = baseline_release
        self.ca_baseline = ca_baseline
        self.s = 0.0    # Activation variable
        self.s0 = 0.0   # Previous activation (for SBDF2)

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Compute AMPA current with calcium-dependent facilitation.

        Args:
            pre_data: {'V': presynaptic voltage (V), 'C': presynaptic calcium (μM)}
            post_data: {'V': postsynaptic voltage (V), 'Cm': membrane capacitance}

        Returns:
            Current (V/s for voltage equation)
        """
        V_post = post_data.get('V', -0.072)  # V
        Cm = post_data.get('Cm', 0.01e-12)  # F/um^2
        C_pre = pre_data.get('C', self.ca_baseline)  # μM

        # Calcium-dependent facilitation
        ca_modulation = 1.0 + self.ca_sensitivity * (C_pre - self.ca_baseline)
        ca_modulation = max(0.0, ca_modulation)  # Non-negative
        release_prob = self.baseline_release * ca_modulation

        # I = g * s * release_prob * (V - E_rev)
        return (self.g_max / Cm) * self.s * release_prob * (V_post - self.E_rev)

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update activation variable with SBDF2.

        Args:
            dt: Time step (s)
            pre_data: {'V': presynaptic voltage (V), 'C': calcium (μM)}
            post_data: Not used
        """
        V_pre = pre_data.get('V', -0.072)  # V

        # Activation based on presynaptic voltage (sigmoid)
        alpha = 0.55
        beta = 4.0e-3  # V
        s_inf = alpha * (1.0 + np.tanh(V_pre / beta))

        # SBDF2 time stepping
        ds = (s_inf - self.s) / self.tau
        ds0 = (s_inf - self.s0) / self.tau

        s_new = (4./3.)*self.s + (4./3.)*ds*dt - (1./3.)*self.s0 - (2./3.)*ds0*dt
        self.s0 = self.s
        self.s = np.clip(s_new, 0.0, 1.0)

class AMPAVModulatedReceptor(ReceptorModel):
    """AMPA receptor with voltage-dependent plasticity.

    Combines standard AMPA kinetics with postsynaptic voltage-dependent
    plasticity.

    This enables modeling of:
    - Paired-pulse facilitation
    - Activity-dependent synaptic strength

    Example:
        receptor = AMPAVModulatedReceptor(
            g_O2=9.0e-12,
            g_O3=15.0e-12,
            g_O4=21.0e-12,
            E_rev=0.0,
            n_receptors=20,
            rho=0.0
        )
    """

    def __init__(self, g_O2: float = 9.0e-12, g_O3: float = 15.0e-12,
                 g_O4: float = 21.0e-12,
                 E_rev: float = 0.0, n_receptors: int = 2.22e3,
                 rho: float = 0.0):
        """Initialize calcium-modulated AMPA receptor.

        Args:
            g_O2: Conductance for state O2 (S)
            g_O3: Conductance for state O3 (S)
            g_O4: Conductance for state O4 (S)
            E_rev: Reversal potential (V)
            n_receptors: Number of receptors in synapse
        """
        self.g_max = (g_O2 + g_O3 + g_O4) / 3.0  # Initial conductance
        self.g_max0 = self.g_max
        self.E_rev = E_rev
        self.n_receptors = n_receptors
        self.Ca_fraction = 0.037  # Calcium-dependent fraction

        a00 = 1.002                         # Cpre factor for thetad basal
        a01 = 1.954                         # Cpost factor for thetad basal
        a10 = 1.159                         # Cpre factor for thetap basal
        a11 = 2.483                         # Cpost factor for thetap basal

        Cpre = 0.07365765664151952          # units??
        Cpost = 0.0132951761734736          # units??

        self.thetad = a00 * Cpre + a01 * Cpost   # depression threshold
        self.thetap = a10 * Cpre + a11 * Cpost   # potentiation threshold

        self.gammap = 216.2                      # potentiation rate
        self.gammad = 101.5                      # depression rate
        self.taustar = 278.318                   # seconds
        self.cstar = 0.0                         # leaky calcium integrator, probably mM?
        self.cstar0 = 0.0                        # previous cstar

        self.tau = 70                            # seconds
        self.rho = rho                          # synapse state (0: depressed, 1: potentiated)
        self.rho0 = rho                           # previous synapse state

        self.tauGampa = 100.0                    # seconds

        if rho == 0.0:
            self.Gd = 1.0 * self.g_max
            self.Gp = 2.0 * self.g_max
        else:
            self.Gd = (1./2.) * self.g_max
            self.Gp = 1.0 * self.g_max

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Compute AMPA current with calcium-dependent facilitation.

        Args:
            post_data: {'V': postsynaptic voltage (V)}

        Returns:
            Current (V/s for voltage equation)
        """
        V_post = post_data.get('V', -0.072)  # V
        Far = 9.6485e-2 # Faradays constant (C/umol)
        z = 2.0 # Calcium valence

        flux = self.Ca_fraction * self.n_receptors * self.g_max * (V_post - self.E_rev)
        flux /= (z * Far)  # Convert to umol/s

        if isinstance(flux, np.ndarray):
            flux[flux < 0] = 0.0  # No negative flux
        else:
            flux = max(0.0, flux)

        return flux

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update activation variable with SBDF2.

        Args:
            dt: Time step (s)
            post_data: {'C': postsynaptic calcium (μM), 'co': baseline calcium (μM)}
        """

        C = post_data.get('C', 0.0)     # μM
        co = post_data.get('co', 50e-3)  # μM

        g_max_new = (4./3.)*self.g_max + (4./3.)*self.gampaF(self.g_max, self.rho)*dt - (1./3.)*self.g_max0 - (2./3.)*self.gampaF(self.g_max0, self.rho0)*dt
        self.g_max0 = self.g_max
        self.g_max = g_max_new

        rho_new = (4./3.)*self.rho + (4./3.)*self.rhoF(self.rho)*dt - (1./3.)*self.rho0 - (2./3.)*self.rhoF(self.rho0)*dt
        self.rho0 = self.rho
        self.rho = rho_new

        cstar_new = (4./3.)*self.cstar + (4./3.)*self.cstarF(self.cstar, C, co)*dt - (1./3.)*self.cstar0 - (2./3.)*self.cstarF(self.cstar0, C, co)*dt
        self.cstar0 = self.cstar
        self.cstar = cstar_new

        if isinstance(self.cstar, np.ndarray):
            print(self.cstar[500])
        else:
            print(self.cstar)

    # Helper functions for plasticity dynamics
    def rhoF(self, rho):

        F = -rho * (1.0 - rho) * (0.5 - rho)
        F += self.gammap * (1.0 - rho) * np.heaviside(self.cstar - self.thetap, 0.5)
        F -= self.gammad * rho * np.heaviside(self.cstar - self.thetad, 0.5)
        F /= self.tau

        return F

    def cstarF(self, cstar, ci, co):

        F = -cstar / self.taustar
        F += ci - co

        return F
    
    def gbar(self, rho):

        return self.Gd + rho * (self.Gp - self.Gd)

    def gampaF(self, Gampa, rho):

        F = self.gbar(rho) - Gampa
        F /= self.tauGampa

        return F

class CustomReceptor(ReceptorModel):
    """User-defined receptor via callback functions.

    Allows users to define arbitrary receptor models without
    modifying the codebase.

    Example:
        def my_flux(pre_data, post_data, sensitivity=1e3):
            pre_C = pre_data.get('C', 0.0)
            post_C = post_data.get('C', 0.0)
            return sensitivity * (pre_C - post_C)

        def my_update(state, dt, pre_data, post_data):
            # Update state dict if needed
            pass

        receptor = CustomReceptor(
            flux_callback=my_flux,
            update_callback=my_update,
            sensitivity=2e3
        )
    """

    def __init__(self, flux_callback, update_callback=None, **params):
        """Initialize custom receptor.

        Args:
            flux_callback: Function(pre_data, post_data, **params) -> flux
            update_callback: Optional function(state, dt, pre_data, post_data) -> None
            **params: Additional parameters passed to flux_callback
        """
        self.flux_callback = flux_callback
        self.update_callback = update_callback
        self.params = params
        self.state = {}  # User can store state here

    def compute_flux(self, pre_data: dict, post_data: dict) -> float:
        """Compute flux using user callback."""
        return self.flux_callback(pre_data, post_data, **self.params)

    def update_state(self, dt: float, pre_data: dict, post_data: dict):
        """Update state using user callback if provided."""
        if self.update_callback is not None:
            self.update_callback(self.state, dt, pre_data, post_data)
