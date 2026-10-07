"""
Base synapse interface for the biological neuron simulator.

This module defines the abstract base class that all synapse models should inherit from.
It establishes the common interface for computing synaptic currents and handling
neurotransmitter release and receptor dynamics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np


class BaseSynapse(ABC):
    """
    Abstract base class for all synapse models.

    This class defines the interface that all concrete synapse models must implement.
    It handles the computation of synaptic currents and the dynamics of synaptic
    resources, neurotransmitter release, and receptor activation.
    """

    def __init__(
        self,
        synapse_id: str = "",
        weight: float = 1.0,           # Synaptic weight (dimensionless or scaled)
        delay: float = 0.0,            # Synaptic delay (ms)
        reversal_potential: float = 0.0 # Reversal potential (mV)
    ):
        """
        Initialize the synapse.

        Args:
            synapse_id: Unique identifier for this synapse
            weight: Synaptic weight (dimensionless, typically 0-1 or higher)
            delay: Synaptic delay (ms)
            reversal_potential: Reversal potential (mV)
        """
        self.synapse_id = synapse_id
        self.weight = weight            # Dimensionless synaptic weight
        self.delay = delay              # Synaptic delay in milliseconds
        self.reversal_potential = reversal_potential  # E_syn in mV

        # State variables for synaptic dynamics
        self.resources = {
            'R': 1.0,   # Fraction of available resources
            'u': 0.0,   # Fraction of resources utilized (release probability)
            'x': 0.0,   # Fraction of resources in active state
        }

        # For short-term plasticity tracking
        self.spike_times = []  # Times of pre-synaptic spikes
        self.last_update_time = 0.0  # Last time state was updated

    @abstractmethod
    def current(self, voltage: float, state: Dict[str, float]) -> float:
        """
        Compute the synaptic current.

        Args:
            voltage: Membrane potential of postsynaptic neuron (mV)
            state: Dictionary of synaptic state variables

        Returns:
            Synaptic current density (µA/cm²)
        """
        pass

    @abstractmethod
    def update_state(
        self,
        pre_synaptic_spike: bool,
        current_time: float,
        dt: float
    ) -> Dict[str, float]:
        """
        Update the synaptic state based on pre-synaptic activity.

        Args:
            pre_synaptic_spike: Whether a pre-synaptic spike occurred
            current_time: Current simulation time (ms)
            dt: Time step (ms)

        Returns:
            Dictionary of updated state variables
        """
        pass

    def activate(self, release_probability: float = 1.0):
        """
        Activate the synapse (trigger neurotransmitter release).

        Args:
            release_probability: Probability of vesicle release (0-1)
        """
        # This would be called when a pre-synaptic spike occurs
        # Implementation depends on specific synapse model
        pass

    def get_info(self) -> Dict[str, float]:
        """
        Get information about this synapse.

        Returns:
            Dictionary of synapse parameters
        """
        return {
            "synapse_id": self.synapse_id,
            "weight": self.weight,
            "delay_ms": self.delay,
            "reversal_potential_mV": self.reversal_potential
        }

    def describe(self) -> str:
        """
        Get a description of the synapse.

        Returns:
            String description of the synapse
        """
        return f"BaseSynapse(id={self.synapse_id}, weight={self.weight}, delay={self.delay}ms, E={self.reversal_potential}mV)"