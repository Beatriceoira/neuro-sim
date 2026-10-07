"""
Base ion channel interface for the biological neuron simulator.

This module defines the abstract base class that all ion channel models should inherit from.
It establishes the common interface for computing ionic currents and state dynamics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np


class IonChannel(ABC):
    """
    Abstract base class for all ion channel models.

    This class defines the interface that all concrete ion channel models must implement.
    It handles the computation of ionic currents and state variable dynamics based on
    membrane potential and other state variables.
    """

    def __init__(
        self,
        channel_id: str = "",
        conductance: float = 0.0,      # Maximum conductance (mS/cm²)
        reversal_potential: float = 0.0 # Reversal potential (mV)
    ):
        """
        Initialize the ion channel.

        Args:
            channel_id: Unique identifier for this channel type
            conductance: Maximum conductance (mS/cm²)
            reversal_potential: Reversal potential (mV)
        """
        self.channel_id = channel_id
        self.conductance = conductance      # g_max in mS/cm²
        self.reversal_potential = reversal_potential  # E_ion in mV

    @abstractmethod
    def current(self, voltage: float, state: Dict[str, float]) -> float:
        """
        Compute the ionic current through this channel.

        Implements: I = g * (V - E_ion)
        where g is the instantaneous conductance determined by channel state.

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary of state variables (gating variables, concentrations, etc.)

        Returns:
            Current density (µA/cm²)
            Positive current is depolarizing (inward positive current)
        """
        pass

    @abstractmethod
    def derivatives(self, voltage: float, state: Dict[str, float]) -> Dict[str, float]:
        """
        Compute the derivatives of state variables for this channel.

        Implements: dy/dt = f(y, V) for each state variable y.

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary of state variables

        Returns:
            Dictionary mapping state variable names to their derivatives
        """
        pass

    def get_conductance(self, voltage: float, state: Dict[str, float]) -> float:
        """
        Compute the instantaneous conductance of this channel.

        This default implementation assumes the conductance is already stored
        in the state dictionary under the channel's conductance key.
        Subclasses can override for more complex conductance calculations.

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary of state variables

        Returns:
            Instantaneous conductance (mS/cm²)
        """
        # Default: look for conductance in state
        cond_key = f"g_{self.channel_id}"
        if cond_key in state:
            return state[cond_key]
        # If not found, return maximum conductance (should be overridden by subclasses)
        return self.conductance

    def get_info(self) -> Dict[str, float]:
        """
        Get information about this ion channel.

        Returns:
            Dictionary of channel parameters
        """
        return {
            "channel_id": self.channel_id,
            "conductance_mS_per_cm2": self.conductance,
            "reversal_potential_mV": self.reversal_potential
        }

    def describe(self) -> str:
        """
        Get a description of the ion channel.

        Returns:
            String description of the channel
        """
        return f"IonChannel(id={self.channel_id}, g_max={self.conductance}mS/cm², E={self.reversal_potential}mV)"