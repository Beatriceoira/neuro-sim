"""
Base neuron class for the biological neuron simulator.

This module defines the abstract base class that all neuron models should inherit from.
It establishes the common interface for neuronal dynamics, spike generation, and
state management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..core.state import NeuronState


class BaseNeuron(ABC):
    """
    Abstract base class for all neuron models.

    This class defines the interface that all concrete neuron models must implement.
    It handles common functionality like state management, spike detection, and
    refractory periods, while leaving the specific dynamics to subclasses.
    """

    def __init__(
        self,
        neuron_id: int = 0,
        initial_state: Optional[NeuronState] = None
    ):
        """
        Initialize the base neuron.

        Args:
            neuron_id: Unique identifier for this neuron
            initial_state: Initial state of the neuron
        """
        self.neuron_id = neuron_id
        self.state = initial_state or NeuronState()

        # Default parameters (to be overridden by subclasses)
        self.spike_threshold = -40.0      # mV
        self.spike_reset = -65.0          # mV
        self.refractory_period = 2.0      # ms
        self.membrane_potential = -65.0   # mV (will sync with state)

        # Synchronize initial state
        self._sync_state()

    def _sync_state(self):
        """Synchronize internal attributes with the state object."""
        self.membrane_potential = self.state.membrane_potential
        # Other state variables can be synced as needed by subclasses

    def _update_state(self):
        """Update state object with current internal attributes."""
        self.state.membrane_potential = self.membrane_potential
        # Other state variables can be updated as needed by subclasses

    @abstractmethod
    def compute_derivatives(
        self,
        t: float,
        external_current: float = 0.0,
        synaptic_inputs: Dict[str, float] = None
    ) -> Tuple[float, Dict]:
        """
        Compute the derivatives of state variables.

        This is the core method that defines the neuron's dynamics.
        Subclasses must implement this to return dv/dt and any other
        state variable derivatives.

        Args:
            t: Current time (ms)
            external_current: External applied current (pA)
            synaptic_inputs: Dictionary of synaptic current inputs

        Returns:
            Tuple of (dv/dt, other_derivatives_dict)
            where dv/dt is the derivative of membrane potential (mV/ms)
            and other_derivatives_dict contains derivatives of other state variables
        """
        pass

    def update(
        self,
        t: float,
        dt: float,
        external_current: float = 0.0,
        synaptic_inputs: Dict[str, float] = None
    ) -> bool:
        """
        Update the neuron's state by one time step.

        Args:
            t: Current time (ms)
            dt: Time step (ms)
            external_current: External applied current (pA)
            synaptic_inputs: Dictionary of synaptic current inputs

        Returns:
            True if a spike occurred during this step, False otherwise
        """
        # Check if in refractory period
        if self.state.refractory_remaining > 0:
            self.state.refractory_remaining = max(0, self.state.refractory_remaining - dt)
            # During refractory period, membrane potential is typically clamped
            # to a value (often reset potential) - this can be customized
            return False

        # Compute derivatives
        dv_dt, other_derivs = self.compute_derivatives(
            t, external_current, synaptic_inputs or {}
        )

        # Update membrane potential using Euler integration
        # Subclasses can override this for more sophisticated integration
        self.membrane_potential += dv_dt * dt

        # Update other state variables
        for var_name, derivative in other_derivs.items():
            if hasattr(self.state, var_name):
                current_val = getattr(self.state, var_name)
                setattr(self.state, var_name, current_val + derivative * dt)
            else:
                # Store in gating_variables or similar dictionary
                if var_name not in self.state.gating_variables:
                    self.state.gating_variables[var_name] = 0.0
                self.state.gating_variables[var_name] += derivative * dt

        # Update state object
        self._update_state()

        # Check for spike threshold crossing
        spiked = False
        if self.membrane_potential >= self.spike_threshold:
            spiked = True
            self._handle_spike(t)

        return spiked

    def _handle_spike(self, t: float):
        """
        Handle the occurrence of a spike.

        Args:
            t: Time of the spike (ms)
        """
        # Record spike time
        self.state.spike_times.append(t)

        # Reset membrane potential
        self.membrane_potential = self.spike_reset

        # Initiate refractory period
        self.state.refractory_remaining = self.refractory_period

        # Synchronize state
        self._update_state()

    def get_state(self) -> NeuronState:
        """
        Get the current state of the neuron.

        Returns:
            Copy of the neuron's state
        """
        self._update_state()
        return self.state

    def set_parameters(self, **kwargs):
        """
        Set neuron parameters.

        Args:
            **kwargs: Parameter names and values to set
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Allow setting of arbitrary parameters
                setattr(self, key, value)

    def describe(self) -> str:
        """
        Get a description of the neuron model.

        Returns:
            String description of the neuron
        """
        return f"{self.__class__.__name__}(id={self.neuron_id})"