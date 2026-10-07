"""
Sodium ion channel model for the biological neuron simulator.

Implements voltage-gated sodium channel dynamics with activation (m) and inactivation (h) gates.
Based on Hodgkin-Huxley formalism.
"""

from typing import Dict
import numpy as np
from .base import IonChannel


class SodiumChannel(IonChannel):
    """
    Voltage-gated sodium ion channel (Na+).

    Implements activation and inactivation gates following Hodgkin-Huxley kinetics:
    - m gate: activation
    - h gate: inactivation

    Current: I_Na = g_Na * m^3 * h * (V - E_Na)
    """

    def __init__(
        self,
        channel_id: str = "na",
        conductance: float = 120.0,      # mS/cm² (maximum conductance)
        reversal_potential: float = 50.0  # mV (sodium reversal potential)
    ):
        """
        Initialize the sodium channel.

        Args:
            channel_id: Identifier for this channel type
            conductance: Maximum conductance (mS/cm²)
            reversal_potential: Reversal potential (mV)
        """
        super().__init__(channel_id, conductance, reversal_potential)

    def current(self, voltage: float, state: Dict[str, float]) -> float:
        """
        Compute the sodium current.

        Implements: I_Na = g_Na * m^3 * h * (V - E_Na)

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary containing 'm' and 'h' gating variables

        Returns:
            Sodium current density (µA/cm²)
        """
        m = state.get('m', 0.0)
        h = state.get('h', 0.0)
        conductance = self.conductance * (m**3) * h
        current = conductance * (voltage - self.reversal_potential)
        return current

    def derivatives(self, voltage: float, state: Dict[str, float]) -> Dict[str, float]:
        """
        Compute derivatives of sodium channel gating variables.

        Implements Hodgkin-Huxley rate equations for m and h gates:
        - dm/dt = alpha_m(V) * (1 - m) - beta_m(V) * m
        - dh/dt = alpha_h(V) * (1 - h) - beta_h(V) * h

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary containing 'm' and 'h' gating variables

        Returns:
            Dictionary with derivatives for 'm' and 'h'
        """
        m = state.get('m', 0.0)
        h = state.get('h', 0.0)

        # Rate constants for m gate (activation)
        alpha_m = 0.1 * (voltage + 40.0) / (1.0 - np.exp(-(voltage + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(voltage + 65.0) / 18.0)

        # Rate constants for h gate (inactivation)
        alpha_h = 0.07 * np.exp(-(voltage + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(voltage + 35.0) / 10.0))

        # Compute derivatives
        dm_dt = alpha_m * (1.0 - m) - beta_m * m
        dh_dt = alpha_h * (1.0 - h) - beta_h * h

        return {
            'm': dm_dt,
            'h': dh_dt
        }

    def get_info(self) -> Dict[str, float]:
        """
        Get information about this sodium channel.

        Returns:
            Dictionary of channel parameters
        """
        info = super().get_info()
        info.update({
            "channel_type": "sodium",
            "activation_gate": "m",
            "inactivation_gate": "h"
        })
        return info

    def describe(self) -> str:
        """
        Get a description of the sodium channel.

        Returns:
            String description of the channel
        """
        return f"SodiumChannel(id={self.channel_id}, g_max={self.conductance}mS/cm², E_Na={self.reversal_potential}mV)"