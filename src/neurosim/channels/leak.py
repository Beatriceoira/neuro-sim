"""
Leak ion channel model for the biological neuron simulator.

Implements a constant conductance leak channel, typically representing
the combined effect of many ion channels at rest.
"""

from typing import Dict
import numpy as np
from .base import IonChannel


class LeakChannel(IonChannel):
    """
    Leak ion channel.

    Implements a constant conductance channel:
    I_L = g_L * (V - E_L)
    """

    def __init__(
        self,
        channel_id: str = "leak",
        conductance: float = 0.3,      # mS/cm² (leak conductance)
        reversal_potential: float = -54.387  # mV (leak reversal potential)
    ):
        """
        Initialize the leak channel.

        Args:
            channel_id: Identifier for this channel type
            conductance: Leak conductance (mS/cm²)
            reversal_potential: Leak reversal potential (mV)
        """
        super().__init__(channel_id, conductance, reversal_potential)

    def current(self, voltage: float, state: Dict[str, float]) -> float:
        """
        Compute the leak current.

        Implements: I_L = g_L * (V - E_L)

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary of state variables (not used for leak channel)

        Returns:
            Leak current density (µA/cm²)
        """
        current = self.conductance * (voltage - self.reversal_potential)
        return current

    def derivatives(self, voltage: float, state: Dict[str, float]) -> Dict[str, float]:
        """
        Compute derivatives of leak channel state variables.

        Leak channel has no dynamic state variables, so returns empty dict.

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary of state variables

        Returns:
            Empty dictionary (no state variables to update)
        """
        return {}

    def get_info(self) -> Dict[str, float]:
        """
        Get information about this leak channel.

        Returns:
            Dictionary of channel parameters
        """
        info = super().get_info()
        info.update({
            "channel_type": "leak"
        })
        return info

    def describe(self) -> str:
        """
        Get a description of the leak channel.

        Returns:
            String description of the channel
        """
        return f"LeakChannel(id={self.channel_id}, g_L={self.conductance}mS/cm², E_L={self.reversal_potential}mV)"