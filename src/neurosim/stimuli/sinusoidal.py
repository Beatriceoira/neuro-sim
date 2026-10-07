"""
Sinusoidal current stimulus for the biological neuron simulator.

Generates a current that follows a sine wave:
I(t) = amplitude * sin(2 * pi * frequency * t + phase) + offset
"""

import numpy as np
from .base import BaseStimulus


class SinusoidalCurrent(BaseStimulus):
    """
    Sinusoidal current stimulus.
    """

    def __init__(
        self,
        stimulus_id: str = "",
        amplitude: float = 1.0,    # Current amplitude (pA or µA/cm²)
        frequency: float = 10.0,   # Frequency (Hz)
        phase: float = 0.0,        # Initial phase (radians)
        offset: float = 0.0,       # DC offset (pA or µA/cm²)
        duration: float = 1000.0   # Duration (ms)
    ):
        """
        Initialize the sinusoidal current stimulus.

        Args:
            stimulus_id: Unique identifier for this stimulus
            amplitude: Peak amplitude (pA or µA/cm²)
            frequency: Frequency of oscillation (Hz)
            phase: Initial phase (radians)
            offset: DC offset (pA or µA/cm²)
            duration: Duration of stimulus (ms)
        """
        super().__init__(stimulus_id, amplitude, duration)
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    def get_current(self, t: float) -> float:
        """
        Get the sinusoidal current at time t.

        Args:
            t: Time (ms)

        Returns:
            Current amplitude at time t
        """
        if 0 <= t <= self.duration:
            # frequency is in Hz, t is in ms, so convert t to seconds
            return self.amplitude * np.sin(2 * np.pi * self.frequency * (t / 1000.0) + self.phase) + self.offset
        return 0.0

    def get_info(self) -> dict:
        info = super().get_info()
        info.update({
            "stimulus_type": "sinusoidal",
            "frequency_Hz": self.frequency,
            "phase_rad": self.phase,
            "offset": self.offset
        })
        return info

    def describe(self) -> str:
        return f"SinusoidalCurrent(id={self.stimulus_id}, amp={self.amplitude}, freq={self.frequency}Hz, offset={self.offset})"


# Factory function for easy stimulus creation
def create_sinusoidal_current(
    stimulus_id: str = "",
    **kwargs
) -> SinusoidalCurrent:
    """
    Factory function to create a sinusoidal current stimulus.

    Args:
        stimulus_id: Unique identifier for this stimulus
        **kwargs: Parameters to override defaults

    Returns:
        Configured SinusoidalCurrent instance
    """
    return SinusoidalCurrent(stimulus_id=stimulus_id, **kwargs)
