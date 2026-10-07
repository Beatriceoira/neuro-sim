"""
Izhikevich neuron model.

Implements the Izhikevich equations:
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)

When v >= 30 mV:
    v = c
    u = u + d

This model can reproduce various firing patterns including:
* regular spiking
* intrinsically bursting
* chattering
* fast spiking
* low-threshold spiking

Note: This is a phenomenological model that captures the spiking dynamics
of cortical neurons but does not explicitly model ion channels.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from .base import BaseNeuron
from ..core.state import NeuronState


class IzhikevichNeuron(BaseNeuron):
    """
    Izhikevich neuron model.

    Implements a system of two differential equations that can reproduce
    various spiking patterns observed in cortical neurons.
    This is a phenomenological model, not based on explicit ion channel dynamics.
    """

    def __init__(
        self,
        neuron_id: int = 0,
        a: float = 0.02,          # time scale of recovery variable u
        b: float = 0.2,           # sensitivity of recovery variable u
        c: float = -65.0,         # after-spike reset value of membrane potential v
        d: float = 8.0,           # after-spike reset of recovery variable u
        initial_state: Optional[NeuronState] = None
    ):
        """
        Initialize the Izhikevich neuron.

        Args:
            neuron_id: Unique identifier for this neuron
            a: Time scale of recovery variable u (ms^-1)
            b: Sensitivity of recovery variable u to membrane potential v (nS)
            c: After-spike reset value of membrane potential v (mV)
            d: After-spike reset of recovery variable u (pA)
            initial_state: Initial state of the neuron
        """
        super().__init__(neuron_id, initial_state)

        # Store Izhikevich parameters
        self.a = a          # ms^-1
        self.b = b          # nS
        self.c = c          # mV
        self.d = d          # pA

        # Override base class attributes (approximate values for spike detection)
        self.spike_threshold = 30.0   # mV (Izhikevich spike threshold)
        self.spike_reset = c          # mV
        self.refractory_period = 2.0  # ms (approximate)

        # Initialize state variables
        # v: membrane potential (mV)
        # u: recovery variable (pA)
        self.state.membrane_potential = -65.0  # Initial membrane potential (mV)
        self.state.recovery_variable = 0.0     # Initial recovery variable (pA)
        self.membrane_potential = -65.0
        self.recovery_variable = 0.0

    def compute_derivatives(
        self,
        t: float,
        external_current: float = 0.0,
        synaptic_inputs: Dict[str, float] = None
    ) -> Tuple[float, Dict]:
        """
        Compute the derivatives for the Izhikevich model.

        Implements:
        dv/dt = 0.04v² + 5v + 140 - u + I
        du/dt = a(bv - u)

        Args:
            t: Current time (ms)
            external_current: External applied current (pA)
            synaptic_inputs: Dictionary of synaptic current inputs

        Returns:
            Tuple of (dv/dt, other_derivatives_dict)
            where other_derivatives_dict contains du/dt
        """
        synaptic_inputs = synaptic_inputs or {}

        # Calculate total input current
        I_total = external_current + sum(synaptic_inputs.values())

        # Izhikevich dynamics
        # dv/dt = 0.04v² + 5v + 140 - u + I
        dv_dt = 0.04 * self.membrane_potential**2 + 5 * self.membrane_potential + 140 - self.recovery_variable + I_total

        # du/dt = a(bv - u)
        du_dt = self.a * (self.b * self.membrane_potential - self.recovery_variable)

        # Return derivatives
        other_derivs = {"recovery_variable": du_dt}

        return dv_dt, other_derivs

    def _handle_spike(self, t: float):
        """
        Handle spike occurrence for Izhikevich neuron.

        Args:
            t: Time of the spike (ms)
        """
        # Record spike time
        self.state.spike_times.append(t)

        # Izhikevich spike reset conditions
        self.membrane_potential = self.c
        self.recovery_variable += self.d

        # Update state object
        self._update_state()

        # Initiate refractory period (approximate)
        self.state.refractory_remaining = self.refractory_period

    def get_biophysical_info(self) -> Dict[str, float]:
        """
        Get Izhikevich parameters of the neuron.

        Returns:
            Dictionary of Izhikevich parameters
        """
        return {
            "a_ms^-1": self.a,
            "b_nS": self.b,
            "c_mV": self.c,
            "d_pA": self.d
        }

    def describe(self) -> str:
        """
        Get a description of the Izhikevich neuron.

        Returns:
            String description of the neuron
        """
        return f"IzhikevichNeuron(id={self.neuron_id}, a={self.a}, b={self.b}, c={self.c}, d={self.d})"


# Presets for different firing patterns
def create_regular_spiking_neuron(neuron_id: int = 0) -> IzhikevichNeuron:
    """
    Create a regular spiking Izhikevich neuron.

    Parameters: a=0.02, b=0.2, c=-65, d=8
    """
    return IzhikevichNeuron(
        neuron_id=neuron_id,
        a=0.02,
        b=0.2,
        c=-65.0,
        d=8.0
    )


def create_intrinsically_bursting_neuron(neuron_id: int = 0) -> IzhikevichNeuron:
    """
    Create an intrinsically bursting Izhikevich neuron.

    Parameters: a=0.02, b=0.2, c=-55, d=4
    """
    return IzhikevichNeuron(
        neuron_id=neuron_id,
        a=0.02,
        b=0.2,
        c=-55.0,
        d=4.0
    )


def create_chattering_neuron(neuron_id: int = 0) -> IzhikevichNeuron:
    """
    Create a chattering Izhikevich neuron.

    Parameters: a=0.02, b=0.2, c=-50, d=2
    """
    return IzhikevichNeuron(
        neuron_id=neuron_id,
        a=0.02,
        b=0.2,
        c=-50.0,
        d=2.0
    )


def create_fast_spiking_neuron(neuron_id: int = 0) -> IzhikevichNeuron:
    """
    Create a fast spiking Izhikevich neuron.

    Parameters: a=0.1, b=0.2, c=-65, d=2
    """
    return IzhikevichNeuron(
        neuron_id=neuron_id,
        a=0.1,
        b=0.2,
        c=-65.0,
        d=2.0
    )


def create_low_threshold_spiking_neuron(neuron_id: int = 0) -> IzhikevichNeuron:
    """
    Create a low-threshold spiking Izhikevich neuron.

    Parameters: a=0.02, b=0.25, c=-65, d=2
    """
    return IzhikevichNeuron(
        neuron_id=neuron_id,
        a=0.02,
        b=0.25,
        c=-65.0,
        d=2.0
    )


# Factory function for easy neuron creation
def create_izhikevich_neuron(
    neuron_id: int = 0,
    **kwargs
) -> IzhikevichNeuron:
    """
    Factory function to create an Izhikevich neuron with default parameters.

    Args:
        neuron_id: Unique identifier for the neuron
        **kwargs: Parameters to override defaults

    Returns:
        Configured IzhikevichNeuron instance
    """
    return IzhikevichNeuron(neuron_id=neuron_id, **kwargs)