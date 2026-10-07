"""
Potassium ion channel model for the biological neuron simulator.

Implements voltage-gated potassium channel dynamics with activation (n) gate.
Based on Hodgkin-Huxley formalism.
"""

from typing import Dict
import numpy as np
from .base import IonChannel


class PotassiumChannel(IonChannel):
    """
    Voltage-gated potassium ion channel (K+).

    Implements activation gate following Hodgkin-Huxley kinetics:
    - n gate: activation

    Current: I_K = g_K * n^4 * (V - E_K)
    """

    def __init__(
        self,
        channel_id: str = "k",
        conductance: float = 36.0,      # mS/cm² (maximum conductance)
        reversal_potential: float = -77.0  # mV (potassium reversal potential)
    ):
        """
        Initialize the potassium channel.

        Args:
            channel_id: Identifier for this channel type
            conductance: Maximum conductance (mS/cm²)
            reversal_potential: Reversal potential (mV)
        """
        super().__init__(channel_id, conductance, reversal_potential)

    def current(self, voltage: float, state: Dict[str, float]) -> float:
        """
        Compute the potassium current.

        Implements: I_K = g_K * n^4 * (V - E_K)

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary containing 'n' gating variable

        Returns:
            Potassium current density (µA/cm²)
        """
        n = state.get('n', 0.0)
        conductance = self.conductance * (n**4)
        current = conductance * (voltage - self.reversal_potential)
        return current

    def derivatives(self, voltage: float, state: Dict[str, float]) -> Dict[str, float]:
        """
        Compute derivatives of potassium channel gating variables.

        Implements Hodgkin-Huxley rate equation for n gate:
        - dn/dt = alpha_n(V) * (1 - n) - beta_n(V) * n

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary containing 'n' gating variable

        Returns:
            Dictionary with derivative for 'n'
        """
        n = state.get('n', 0.0)

        # Rate constants for n gate (activation)
        alpha_n = 0.01 * (voltage + 55.0) / (1.0 - np.exp(-(voltage + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(voltage + 65.0) / 80.0)

        # Compute derivative
        dn_dt = alpha_n * (1.0 - n) - beta_n * n

        return {
            'n': dn_dt
        }

    def get_info(self) -> Dict[str, float]:
        """
        Get information about this potassium channel.

        Returns:
            Dictionary of channel parameters
        """
        info = super().get_info()
        info.update({
            "channel_type": "potassium",
            "activation_gate": "n"
        })
        return info

    def describe(self) -> str:
        """
        Get a description of the potassium channel.

        Returns:
            String description of the channel
        """
        return f"PotassiumChannel(id={self.channel_id}, g_max={self.conductance}mS/cm², E_K={self.reversal_potential}mV)"