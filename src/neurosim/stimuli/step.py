"""
Step current stimulus for the biological neuron simulator.

Generates a constant current that turns on at a specified start time
and turns off at a specified end time.
"""

from typing import Optional
import numpy as np
from .base import BaseStimulus


class StepCurrent(BaseStimulus):
    """
    Step current stimulus.

    Implements a current that is zero before start_time, constant amplitude
    between start_time and end_time, and zero after end_time.
    """

    def __init__(
        self,
        stimulus_id: str = "",
        amplitude: float = 0.0,    # Current amplitude (pA or µA/cm²)
        start_time: float = 0.0,   # Start time (ms)
        end_time: float = 0.0      # End time (ms)
    ):
        """
        Initialize the step current stimulus.

        Args:
            stimulus_id: Unique identifier for this stimulus
            amplitude: Current amplitude during the step (pA or µA/cm²)
            start_time: Time when current turns on (ms)
            end_time: Time when current turns off (ms)
        """
        super().__init__(stimulus_id, amplitude, end_time - start_time)
        self.start_time = start_time
        self.end_time = end_time

    def get_current(self, t: float) -> float:
        """
        Get the step current at time t.

        Args:
            t: Time (ms)

        Returns:
            Current amplitude at time t
        """
        if self.start_time <= t < self.end_time:
            return self.amplitude
        else:
            return 0.0

    def get_info(self) -> dict:
        """
        Get information about this step current stimulus.

        Returns:
            Dictionary of stimulus parameters
        """
        info = super().get_info()
        info.update({
            "stimulus_type": "step",
            "start_time_ms": self.start_time,
            "end_time_ms": self.end_time
        })
        return info

    def describe(self) -> str:
        """
        Get a description of the step current stimulus.

        Returns:
            String description of the stimulus
        """
        return f"StepCurrent(id={self.stimulus_id}, amplitude={self.amplitude}, start={self.start_time}ms, end={self.end_time}ms)"


# Factory function for easy stimulus creation
def create_step_current(
    stimulus_id: str = "",
    **kwargs
) -> StepCurrent:
    """
    Factory function to create a step current stimulus.

    Args:
        stimulus_id: Unique identifier for the stimulus
        **kwargs: Parameters to override defaults

    Returns:
        Configured StepCurrent instance
    """
    return StepCurrent(stimulus_id=stimulus_id, **kwargs)