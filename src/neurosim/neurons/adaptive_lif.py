"""
Adaptive Leaky Integrate-and-Fire (ALIF) neuron model.

Extends the LIF model with spike-frequency adaptation through an adaptation current.
Implements dynamics:
dV/dt = [-(V - V_rest) + R_m * I - w] / tau_m
dw/dt = (a * (V - V_rest) - w) / tau_w

Where w is the adaptation current, a is adaptation sensitivity, and tau_w is adaptation time constant.

Supports:
* spike-frequency adaptation
* adaptation current
* dynamic threshold
* refractory dynamics
"""

from typing import Dict, Tuple, Optional
import numpy as np
from .base import BaseNeuron
from ..core.state import NeuronState


class AdaptiveLIFNeuron(BaseNeuron):
    """
    Adaptive Leaky Integrate-and-Fire (ALIF) neuron model.

    Extends the LIF model with spike-frequency adaptation through an adaptation current.
    This model demonstrates adaptation, regular spiking, and burst-like behavior.
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
        adaptation_sensitivity: float = 0.0,    # nS (subthreshold adaptation)
        adaptation_time_constant: float = 40.0, # ms
        spike_triggered_increment: float = 0.0, # nA (spike-triggered adaptation)
        initial_state: Optional[NeuronState] = None
    ):
        """
        Initialize the ALIF neuron.

        Args:
            neuron_id: Unique identifier for this neuron
            membrane_resistance: Membrane resistance (MΩ)
            membrane_time_constant: Membrane time constant (ms)
            resting_potential: Resting membrane potential (mV)
            threshold_potential: Spike threshold potential (mV)
            reset_potential: Reset potential after spike (mV)
            refractory_period: Refractory period duration (ms)
            adaptation_sensitivity: Subthreshold adaptation sensitivity (nS)
            adaptation_time_constant: Adaptation time constant (ms)
            spike_triggered_increment: Spike-triggered adaptation increment (nA)
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
        self.adaptation_sensitivity = adaptation_sensitivity # nS
        self.adaptation_time_constant = adaptation_time_constant # ms
        self.spike_triggered_increment = spike_triggered_increment # nA

        # Override base class attributes
        self.spike_threshold = threshold_potential
        self.spike_reset = reset_potential
        self.refractory_period = refractory_period

        # Initialize state
        self.state.membrane_potential = resting_potential
        self.state.recovery_variable = 0.0  # Adaptation current w (nA)
        self.membrane_potential = resting_potential
        self.recovery_variable = 0.0

    def compute_derivatives(
        self,
        t: float,
        external_current: float = 0.0,
        synaptic_inputs: Dict[str, float] = None
    ) -> Tuple[float, Dict]:
        """
        Compute the derivatives for the ALIF model.

        Implements:
        dV/dt = [-(V - V_rest) + R_m * I - w] / tau_m
        dw/dt = (a * (V - V_rest) - w) / tau_w

        Args:
            t: Current time (ms)
            external_current: External applied current (pA)
            synaptic_inputs: Dictionary of synaptic current inputs

        Returns:
            Tuple of (dv/dt, other_derivatives_dict)
            where other_derivatives_dict contains dw/dt
        """
        synaptic_inputs = synaptic_inputs or {}

        # Calculate total input current
        I_total = external_current + sum(synaptic_inputs.values())

        # ALIF dynamics
        # dV/dt = [-(V - V_rest) + R_m * I - w] / tau_m
        dv_dt = (-(self.membrane_potential - self.resting_potential) +
                 self.membrane_resistance * I_total -
                 self.recovery_variable) / self.membrane_time_constant

        # dw/dt = (a * (V - V_rest) - w) / tau_w
        dw_dt = (self.adaptation_sensitivity * (self.membrane_potential - self.resting_potential) -
                 self.recovery_variable) / self.adaptation_time_constant

        # Return derivatives
        other_derivs = {"recovery_variable": dw_dt}

        return dv_dt, other_derivs

    def _handle_spike(self, t: float):
        """
        Handle spike occurrence for ALIF neuron.

        Args:
            t: Time of the spike (ms)
        """
        super()._handle_spike(t)
        # Add spike-triggered adaptation increment
        self.recovery_variable += self.spike_triggered_increment
        self.state.recovery_variable = self.recovery_variable

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
            "refractory_period_ms": self.refractory_period,
            "adaptation_sensitivity_nS": self.adaptation_sensitivity,
            "adaptation_time_constant_ms": self.adaptation_time_constant,
            "spike_triggered_increment_nA": self.spike_triggered_increment
        }

    def describe(self) -> str:
        """
        Get a description of the ALIF neuron.

        Returns:
            String description of the neuron
        """
        return (f"AdaptiveLIFNeuron(id={self.neuron_id}, "
                f"Rm={self.membrane_resistance}MΩ, "
                f"τm={self.membrane_time_constant}ms, "
                f"Vrest={self.resting_potential}mV, "
                f"Vth={self.threshold_potential}mV, "
                f"a={self.adaptation_sensitivity}nS, "
                f"τw={self.adaptation_time_constant}ms)")


# Factory function for easy neuron creation
def create_adaptive_lif_neuron(
    neuron_id: int = 0,
    **kwargs
) -> AdaptiveLIFNeuron:
    """
    Factory function to create an ALIF neuron with default parameters.

    Args:
        neuron_id: Unique identifier for the neuron
        **kwargs: Parameters to override defaults

    Returns:
        Configured AdaptiveLIFNeuron instance
    """
    return AdaptiveLIFNeuron(neuron_id=neuron_id, **kwargs)