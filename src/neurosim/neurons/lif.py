"""
Leaky Integrate-and-Fire (LIF) neuron model.

Implements the classic LIF model with dynamics:
dV/dt = [-(V - V_rest) + R_m * I] / tau_m

Where:
- V is membrane potential
- V_rest is resting potential
- R_m is membrane resistance
- I is input current
- tau_m is membrane time constant
"""

from typing import Dict, Tuple, Optional
import numpy as np
from .base import BaseNeuron
from ..core.state import NeuronState


class LIFNeuron(BaseNeuron):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    Implements subthreshold integration and spike-threshold reset dynamics.
    This is a simplified abstraction of neuronal integration and spike generation.
    It does not explicitly model ion channels or action-potential biophysics.
    """

    def __init__(
        self,
        neuron_id: int = 0,
        membrane_resistance: float = 10.0,      # MΩ
        membrane_time_constant: float = 20.0,   # ms
        resting_potential: float = -65.0,       # mV
        threshold_potential: float = -50.0,     # mV
        reset_potential: float = -65.0,         # mV
        refractory_period: float = 2.0,         # ms
        initial_state: Optional[NeuronState] = None
    ):
        """
        Initialize the LIF neuron.

        Args:
            neuron_id: Unique identifier for this neuron
            membrane_resistance: Membrane resistance (MΩ)
            membrane_time_constant: Membrane time constant (ms)
            resting_potential: Resting membrane potential (mV)
            threshold_potential: Spike threshold potential (mV)
            reset_potential: Reset potential after spike (mV)
            refractory_period: Refractory period duration (ms)
            initial_state: Initial state of the neuron
        """
        super().__init__(neuron_id, initial_state)

        # Store parameters
        self.membrane_resistance = membrane_resistance      # MΩ
        self.membrane_time_constant = membrane_time_constant # ms
        self.resting_potential = resting_potential          # mV
        self.threshold_potential = threshold_potential      # mV
        self.reset_potential = reset_potential              # mV
        self.refractory_period = refractory_period          # ms

        # Override base class attributes
        self.spike_threshold = threshold_potential
        self.spike_reset = reset_potential
        self.refractory_period = refractory_period

        # Initialize state with resting potential
        self.state.membrane_potential = resting_potential
        self.membrane_potential = resting_potential

    def compute_derivatives(
        self,
        t: float,
        external_current: float = 0.0,
        synaptic_inputs: Dict[str, float] = None
    ) -> Tuple[float, Dict]:
        """
        Compute the derivatives for the LIF model.

        Implements: dV/dt = [-(V - V_rest) + R_m * I] / tau_m

        Args:
            t: Current time (ms)
            external_current: External applied current (pA)
            synaptic_inputs: Dictionary of synaptic current inputs

        Returns:
            Tuple of (dv/dt, other_derivatives_dict)
            For LIF, other_derivatives_dict is empty as there are no additional state variables
        """
        synaptic_inputs = synaptic_inputs or {}

        # Calculate total input current
        I_total = external_current + sum(synaptic_inputs.values())

        # LIF dynamics: dV/dt = [-(V - V_rest) + R_m * I] / tau_m
        dv_dt = (-(self.membrane_potential - self.resting_potential) +
                 self.membrane_resistance * I_total) / self.membrane_time_constant

        # No additional state variables for basic LIF
        other_derivs = {}

        return dv_dt, other_derivs

    def _handle_spike(self, t: float):
        """
        Handle spike occurrence for LIF neuron.

        Args:
            t: Time of the spike (ms)
        """
        super()._handle_spike(t)
        # After spike, membrane potential is already reset to reset_potential
        # by the base class _handle_spike method

    def get_biophysical_info(self) -> Dict[str, float]:
        """
        Get biophysical parameters of the neuron.

        Returns:
            Dictionary of biophysical parameters
        """
        return {
            "membrane_resistance_MOhm": self.membrane_resistance,
            "membrane_time_constant_ms": self.membrane_time_constant,
            "resting_potential_mV": self.resting_potential,
            "threshold_potential_mV": self.threshold_potential,
            "reset_potential_mV": self.reset_potential,
            "refractory_period_ms": self.refractory_period
        }

    def describe(self) -> str:
        """
        Get a description of the LIF neuron.

        Returns:
            String description of the neuron
        """
        return (f"LIFNeuron(id={self.neuron_id}, "
                f"Rm={self.membrane_resistance}MΩ, "
                f"τm={self.membrane_time_constant}ms, "
                f"Vrest={self.resting_potential}mV, "
                f"Vth={self.threshold_potential}mV)")


# Factory function for easy neuron creation
def create_lif_neuron(
    neuron_id: int = 0,
    **kwargs
) -> LIFNeuron:
    """
    Factory function to create an LIF neuron with default parameters.

    Args:
        neuron_id: Unique identifier for the neuron
        **kwargs: Parameters to override defaults

    Returns:
        Configured LIFNeuron instance
    """
    return LIFNeuron(neuron_id=neuron_id, **kwargs)