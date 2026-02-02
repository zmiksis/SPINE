"""Temporal pattern classes for synaptic activity.

Temporal patterns define the time-dependent amplitude of synaptic activity.
They are independent of the biological mechanism (receptor type).

Available patterns:
- ConstantPattern: Fixed amplitude for specified duration
- LinearDecayPattern: Linear ramp down from initial to zero
- ExponentialDecayPattern: Exponential decay with time constant
- PulseTrainPattern: Periodic pulses at specified frequency
- CustomPattern: User-defined pattern via callback function
"""

import numpy as np
from spine.synapse.base import TemporalPattern


class ConstantPattern(TemporalPattern):
    """Constant amplitude for fixed duration.

    Amplitude = A for 0 <= t-t0 < duration, 0 otherwise

    Example:
        # 5 ms pulse starting at t=10 ms
        pattern = ConstantPattern(amplitude=1.0, duration=5e-3, start_time=10e-3)
    """

    def __init__(self, amplitude: float, duration: float, start_time: float = 0.0):
        """Initialize constant pattern.

        Args:
            amplitude: Amplitude value (dimensionless)
            duration: Duration of pulse (seconds)
            start_time: Time when pulse starts (seconds)
        """
        self.amplitude = amplitude
        self.duration = duration
        self.start_time = start_time

    def get_amplitude(self, t: float) -> float:
        """Return amplitude at time t."""
        elapsed = t - self.start_time
        if 0 <= elapsed < self.duration:
            return self.amplitude
        return 0.0

    def reset(self):
        """Reset to initial state."""
        self.start_time = 0.0


class LinearDecayPattern(TemporalPattern):
    """Linear decay from initial amplitude to zero.

    Amplitude = A * (1 - (t-t0)/duration) for 0 <= t-t0 < duration, 0 otherwise

    This is equivalent to the current system's 'linear' type.

    Example:
        # Linear ramp from 1.0 to 0.0 over 10 ms
        pattern = LinearDecayPattern(amplitude=1.0, duration=10e-3)
    """

    def __init__(self, amplitude: float, duration: float, start_time: float = 0.0):
        """Initialize linear decay pattern.

        Args:
            amplitude: Initial amplitude value
            duration: Duration of decay (seconds)
            start_time: Time when decay starts (seconds)
        """
        self.amplitude = amplitude
        self.duration = duration
        self.start_time = start_time

    def get_amplitude(self, t: float) -> float:
        """Return amplitude at time t."""
        elapsed = t - self.start_time
        if 0 <= elapsed < self.duration:
            return self.amplitude * (1.0 - elapsed / self.duration)
        elif elapsed >= self.duration:
            return 0.0
        else:
            return 0.0

    def reset(self):
        """Reset to initial state."""
        self.start_time = 0.0


class ExponentialDecayPattern(TemporalPattern):
    """Exponential decay: A * exp(-(t-t0)/tau).

    Amplitude = A * exp(-(t-t0)/tau) for t >= t0, 0 otherwise

    This is equivalent to the current system's 'exponential' type.

    Example:
        # Exponential decay with 10 ms time constant
        pattern = ExponentialDecayPattern(amplitude=1.0, tau=10e-3)
    """

    def __init__(self, amplitude: float, tau: float, start_time: float = 0.0):
        """Initialize exponential decay pattern.

        Args:
            amplitude: Initial amplitude value
            tau: Time constant (seconds)
            start_time: Time when decay starts (seconds)
        """
        self.amplitude = amplitude
        self.tau = tau
        self.start_time = start_time

    def get_amplitude(self, t: float) -> float:
        """Return amplitude at time t."""
        if t >= self.start_time:
            return self.amplitude * np.exp(-(t - self.start_time) / self.tau)
        return 0.0

    def reset(self):
        """Reset to initial state."""
        self.start_time = 0.0


class PulseTrainPattern(TemporalPattern):
    """Periodic pulses at specified frequency.

    Generates a train of rectangular pulses with specified frequency
    and duty cycle.

    Example:
        # 10 Hz stimulation, 1 ms pulse width, 5 pulses
        pattern = PulseTrainPattern(
            amplitude=1.0,
            pulse_duration=1e-3,
            frequency=10.0,
            n_pulses=5
        )
    """

    def __init__(self, amplitude: float, pulse_duration: float,
                 frequency: float, n_pulses: int = None,
                 start_time: float = 0.0):
        """Initialize pulse train pattern.

        Args:
            amplitude: Amplitude of each pulse
            pulse_duration: Duration of each pulse (seconds)
            frequency: Pulse frequency (Hz)
            n_pulses: Number of pulses (None = infinite)
            start_time: Time when train starts (seconds)
        """
        self.amplitude = amplitude
        self.pulse_duration = pulse_duration
        self.period = 1.0 / frequency
        self.n_pulses = n_pulses
        self.start_time = start_time

    def get_amplitude(self, t: float) -> float:
        """Return amplitude at time t."""
        if t < self.start_time:
            return 0.0

        elapsed = t - self.start_time
        pulse_idx = int(elapsed / self.period)

        # Check if we've exceeded pulse count
        if self.n_pulses is not None and pulse_idx >= self.n_pulses:
            return 0.0

        # Check if we're within a pulse
        t_in_pulse = elapsed % self.period
        if t_in_pulse < self.pulse_duration:
            return self.amplitude
        return 0.0

    def reset(self):
        """Reset to initial state."""
        self.start_time = 0.0


class CustomPattern(TemporalPattern):
    """User-defined pattern via callback function.

    Allows users to define arbitrary temporal patterns without
    modifying the codebase.

    Example:
        # Sine wave modulation
        def sine_pattern(t):
            return 0.5 * (1.0 + np.sin(2*np.pi*10*t))  # 10 Hz

        pattern = CustomPattern(sine_pattern)

        # Burst pattern
        def burst_pattern(t):
            burst_freq = 10.0  # Hz
            within_burst_freq = 100.0  # Hz
            burst_duration = 50e-3  # 50 ms

            t_in_cycle = t % (1.0/burst_freq)
            if t_in_cycle < burst_duration:
                # High frequency within burst
                t_in_burst = t_in_cycle % (1.0/within_burst_freq)
                if t_in_burst < 1e-3:  # 1 ms pulse
                    return 1.0
            return 0.0

        pattern = CustomPattern(burst_pattern)
    """

    def __init__(self, callback):
        """Initialize custom pattern.

        Args:
            callback: Function that takes time (seconds) and returns amplitude
                      Signature: callback(t: float) -> float
        """
        self.callback = callback

    def get_amplitude(self, t: float) -> float:
        """Return amplitude at time t."""
        return self.callback(t)

    def reset(self):
        """Reset to initial state (no-op for custom patterns)."""
        pass
