"""
Ion channel models and factory functions for the biological neuron simulator.

This module provides convenient factory functions for creating ion channels
and managing channel collections for neurons.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from .sodium import SodiumChannel
from .potassium import PotassiumChannel
from .leak import LeakChannel
from .base import IonChannel


def create_sodium_channel(
    conductance: float = 120.0,      # mS/cm²
    reversal_potential: float = 50.0  # mV
) -> SodiumChannel:
    """
    Factory function to create a sodium channel.

    Args:
        conductance: Maximum conductance (mS/cm²)
        reversal_potential: Reversal potential (mV)

    Returns:
        Configured SodiumChannel instance
    """
    return SodiumChannel(conductance=conductance, reversal_potential=reversal_potential)


def create_potassium_channel(
    conductance: float = 36.0,       # mS/cm²
    reversal_potential: float = -77.0 # mV
) -> PotassiumChannel:
    """
    Factory function to create a potassium channel.

    Args:
        conductance: Maximum conductance (mS/cm²)
        reversal_potential: Reversal potential (mV)

    Returns:
        Configured PotassiumChannel instance
    """
    return PotassiumChannel(conductance=conductance, reversal_potential=reversal_potential)


def create_leak_channel(
    conductance: float = 0.3,        # mS/cm²
    reversal_potential: float = -54.387  # mV
) -> LeakChannel:
    """
    Factory function to create a leak channel.

    Args:
        conductance: Leak conductance (mS/cm²)
        reversal_potential: Leak reversal potential (mV)

    Returns:
        Configured LeakChannel instance
    """
    return LeakChannel(conductance=conductance, reversal_potential=reversal_potential)


def create_hh_channels() -> Dict[str, IonChannel]:
    """
    Create standard Hodgkin-Huxley channel set.

    Returns:
        Dictionary mapping channel names to IonChannel instances
        Contains: 'na', 'k', 'leak' channels with standard HH parameters
    """
    return {
        'na': create_sodium_channel(),
        'k': create_potassium_channel(),
        'leak': create_leak_channel()
    }


def compute_total_current(
    channels: Dict[str, IonChannel],
    voltage: float,
    state: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total ionic current from multiple channels.

    Args:
        channels: Dictionary mapping channel names to IonChannel instances
        voltage: Membrane potential (mV)
        state: Dictionary of state variables

    Returns:
        Tuple of (total_current, individual_currents)
        where total_current is the sum of all channel currents (µA/cm²)
        and individual_currents is a dictionary mapping channel names to their currents
    """
    individual_currents = {}
    total_current = 0.0

    for channel_name, channel in channels.items():
        current = channel.current(voltage, state)
        individual_currents[channel_name] = current
        total_current += current

    return total_current, individual_currents


def compute_channel_derivatives(
    channels: Dict[str, IonChannel],
    voltage: float,
    state: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute derivatives of all channel state variables.

    Args:
        channels: Dictionary mapping channel names to IonChannel instances
        voltage: Membrane potential (mV)
        state: Dictionary of state variables

    Returns:
        Dictionary mapping state variable names to their derivatives
    """
    all_derivatives = {}

    for channel_name, channel in channels.items():
        derivatives = channel.derivatives(voltage, state)
        # Prefix derivative names with channel name to avoid conflicts
        for var_name, derivative in derivatives.items():
            prefixed_name = f"{channel_name}_{var_name}"
            all_derivatives[prefixed_name] = derivative

    return all_derivatives


class ChannelManager:
    """
    Manages a collection of ion channels for a neuron.

    This class simplifies working with multiple ion channels by providing
    methods to compute total currents and state derivatives.
    """

    def __init__(self, channels: Dict[str, IonChannel] = None):
        """
        Initialize the channel manager.

        Args:
            channels: Dictionary mapping channel names to IonChannel instances
        """
        self.channels = channels or {}

    def add_channel(self, name: str, channel: IonChannel):
        """
        Add a channel to the manager.

        Args:
            name: Name identifier for the channel
            channel: IonChannel instance to add
        """
        self.channels[name] = channel

    def remove_channel(self, name: str):
        """
        Remove a channel from the manager.

        Args:
            name: Name identifier for the channel to remove
        """
        if name in self.channels:
            del self.channels[name]

    def get_channel(self, name: str) -> Optional[IonChannel]:
        """
        Get a channel by name.

        Args:
            name: Name identifier for the channel

        Returns:
            IonChannel instance or None if not found
        """
        return self.channels.get(name)

    def compute_total_current(
        self,
        voltage: float,
        state: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total ionic current from all managed channels.

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary of state variables

        Returns:
            Tuple of (total_current, individual_currents)
        """
        return compute_total_current(self.channels, voltage, state)

    def compute_derivatives(
        self,
        voltage: float,
        state: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute derivatives of all managed channel state variables.

        Args:
            voltage: Membrane potential (mV)
            state: Dictionary of state variables

        Returns:
            Dictionary mapping state variable names to their derivatives
        """
        return compute_channel_derivatives(self.channels, voltage, state)

    def get_info(self) -> Dict[str, Dict]:
        """
        Get information about all managed channels.

        Returns:
            Dictionary mapping channel names to their info dictionaries
        """
        return {name: channel.get_info() for name, channel in self.channels.items()}

    def describe(self) -> str:
        """
        Get a description of the channel manager.

        Returns:
            String description of the channel collection
        """
        channel_descriptions = [channel.describe() for channel in self.channels.values()]
        return f"ChannelManager({len(self.channels)} channels: {', '.join(channel_descriptions)})"
