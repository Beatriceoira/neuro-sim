"""
Base stimulus interface for the biological neuron simulator.

This module defines the abstract base class that all stimulus models should inherit from.
It establishes the common interface for generating input currents to neurons.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseStimulus(ABC):
    """
    Abstract base class for all stimulus models.

    This class defines the interface that all concrete stimulus models must implement.
    It handles the generation of input currents over time for neuronal simulation.
    """

    def __init__(
        self,
        stimulus_id: str = "",
        amplitude: float = 0.0,    # Current amplitude (pA or µA/cm²)
        duration: float = 0.0      # Stimulus duration (ms)
    ):
        """
        Initialize the stimulus.

        Args:
            stimulus_id: Unique identifier for this stimulus
            amplitude: Current amplitude
            duration: Stimulus duration (ms)
        """
        self.stimulus_id = stimulus_id
        self.amplitude = amplitude
        self.duration = duration

    @abstractmethod
    def get_current(self, t: float) -> float:
        """
        Get the stimulus current at time t.

        Args:
            t: Time (ms)

        Returns:
            Current amplitude at time t
        """
        pass

    def get_info(self) -> dict:
        """
        Get information about this stimulus.

        Returns:
            Dictionary of stimulus parameters
        """
        return {
            "stimulus_id": self.stimulus_id,
            "amplitude": self.amplitude,
            "duration": self.duration
        }

    def describe(self) -> str:
        """
        Get a description of the stimulus.

        Returns:
            String description of the stimulus
        """
        return f"BaseStimulus(id={self.stimulus_id}, amplitude={self.amplitude}, duration={self.duration}ms)"