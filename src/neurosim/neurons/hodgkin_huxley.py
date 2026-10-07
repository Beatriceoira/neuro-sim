"""
Hodgkin-Huxley neuron model.

Implements a full Hodgkin-Huxley neuron with dynamics for:
* membrane capacitance
* sodium current (I_Na)
* potassium current (I_K)
* leak current (I_L)
* gating variables m, h, n

Equations:
C_m * dV/dt = I_ext - I_Na - I_K - I_L

Where:
I_Na = g_Na * m^3 * h * (V - E_Na)
I_K = g_K * n^4 * (V - E_K)
I_L = g_L * (V - E_L)

Gating variables follow first-order kinetics:
dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x

for x in {m, h, n}

Note: This model explicitly models ion channel biophysics and action potential generation.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from .base import BaseNeuron
from ..core.state import NeuronState


class HodgkinHuxleyNeuron(BaseNeuron):
    """
    Hodgkin-Huxley neuron model.

    Implements the canonical model of action potential generation based on
    voltage-gated ion channels. This model explicitly represents the biophysics
    of sodium and potassium channels, as well as leak currents.
    """

    def __init__(
        self,
        neuron_id: int = 0,
        membrane_capacitance: float = 1.0,      # µF/cm²
        # Sodium channel parameters
        g_na: float = 120.0,                    # mS/cm²
        e_na: float = 50.0,                     # mV
        # Potassium channel parameters
        g_k: float = 36.0,                      # mS/cm²
        e_k: float = -77.0,                     # mV
        # Leak channel parameters
        g_l: float = 0.3,                       # mS/cm²
        e_l: float = -54.387,                   # mV
        initial_state: Optional[NeuronState] = None
    ):
        """
        Initialize the Hodgkin-Huxley neuron.

        Args:
            neuron_id: Unique identifier for this neuron
            membrane_capacitance: Membrane capacitance (µF/cm²)
            g_na: Maximum sodium conductance (mS/cm²)
            e_na: Sodium reversal potential (mV)
            g_k: Maximum potassium conductance (mS/cm²)
            e_k: Potassium reversal potential (mV)
            g_l: Leak conductance (mS/cm²)
            e_l: Leak reversal potential (mV)
            initial_state: Initial state of the neuron
        """
        super().__init__(neuron_id, initial_state)

        # Store biophysical parameters
        self.membrane_capacitance = membrane_capacitance  # µF/cm²
        self.g_na = g_na                                  # mS/cm²
        self.e_na = e_na                                  # mV
        self.g_k = g_k                                    # mS/cm²
        self.e_k = e_k                                    # mV
        self.g_l = g_l                                    # mS/cm²
        self.e_l = e_l                                    # mV

        # Override base class attributes for spike detection
        self.spike_threshold = -40.0   # mV (approximate for HH)
        self.spike_reset = -65.0       # mV
        self.refractory_period = 2.0   # ms (approximate)

        # Initialize state variables
        # V: membrane potential (mV)
        # m: sodium activation gate
        # h: sodium inactivation gate
        # n: potassium activation gate
        self.state.membrane_potential = -65.0  # Initial membrane potential (mV)
        self.state.gating_variables = {
            'm': 0.05,  # Initial sodium activation
            'h': 0.6,   # Initial sodium inactivation
            'n': 0.32   # Initial potassium activation
        }
        self.state.ionic_currents = {
            'I_Na': 0.0,  # Sodium current density (µA/cm²)
            'I_K': 0.0,   # Potassium current density (µA/cm²)
            'I_L': 0.0    # Leak current density (µA/cm²)
        }

        # Synchronize state
        self._sync_state()

    def _sync_state(self):
        """Synchronize internal attributes with the state object."""
        self.membrane_potential = self.state.membrane_potential
        self.gating_variables = self.state.gating_variables.copy()
        self.ionic_currents = self.state.ionic_currents.copy()

    def _update_state(self):
        """Update state object with current internal attributes.

        Gating variables and ionic currents are read back from the state
        object so that updates applied by the base class integration loop
        (which writes directly to ``self.state.gating_variables``) are
        reflected in the neuron's own working copies.
        """
        self.state.membrane_potential = self.membrane_potential
        self.gating_variables = self.state.gating_variables.copy()
        self.ionic_currents = self.state.ionic_currents.copy()

    def _alpha_m(self, V: float) -> float:
        """Alpha rate constant for m gate."""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def _beta_m(self, V: float) -> float:
        """Beta rate constant for m gate."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def _alpha_h(self, V: float) -> float:
        """Alpha rate constant for h gate."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def _beta_h(self, V: float) -> float:
        """Beta rate constant for h gate."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def _alpha_n(self, V: float) -> float:
        """Alpha rate constant for n gate."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def _beta_n(self, V: float) -> float:
        """Beta rate constant for n gate."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def compute_derivatives(
        self,
        t: float,
        external_current: float = 0.0,
        synaptic_inputs: Dict[str, float] = None
    ) -> Tuple[float, Dict]:
        """
        Compute the derivatives for the Hodgkin-Huxley model.

        Implements the full HH equations:
        C_m * dV/dt = I_ext - I_Na - I_K - I_L

        Args:
            t: Current time (ms)
            external_current: External applied current (µA/cm²)
            synaptic_inputs: Dictionary of synaptic current inputs (µA/cm²)

        Returns:
            Tuple of (dv/dt, other_derivatives_dict)
            where other_derivatives_dict contains dm/dt, dh/dt, dn/dt
            and current values
        """
        synaptic_inputs = synaptic_inputs or {}

        # Calculate total input current
        I_total = external_current + sum(synaptic_inputs.values())

        # Get current state variables
        V = self.membrane_potential
        m = self.gating_variables['m']
        h = self.gating_variables['h']
        n = self.gating_variables['n']

        # Calculate ionic currents
        # I_Na = g_Na * m^3 * h * (V - E_Na)
        I_Na = self.g_na * m**3 * h * (V - self.e_na)

        # I_K = g_K * n^4 * (V - E_K)
        I_K = self.g_k * n**4 * (V - self.e_k)

        # I_L = g_L * (V - E_L)
        I_L = self.g_l * (V - self.e_l)

        # Store currents in state
        self.ionic_currents['I_Na'] = I_Na
        self.ionic_currents['I_K'] = I_K
        self.ionic_currents['I_L'] = I_L

        # Membrane potential dynamics
        # C_m * dV/dt = I_ext - I_Na - I_K - I_L
        dv_dt = (I_total - I_Na - I_K - I_L) / self.membrane_capacitance

        # Gating variable dynamics
        # dm/dt = alpha_m(V) * (1 - m) - beta_m(V) * m
        dm_dt = self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m

        # dh/dt = alpha_h(V) * (1 - h) - beta_h(V) * h
        dh_dt = self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h

        # dn/dt = alpha_n(V) * (1 - n) - beta_n(V) * n
        dn_dt = self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n

        # Return derivatives
        other_derivs = {
            'm': dm_dt,
            'h': dh_dt,
            'n': dn_dt,
            'I_Na': I_Na,
            'I_K': I_K,
            'I_L': I_L
        }

        return dv_dt, other_derivs

    def _handle_spike(self, t: float):
        """
        Handle spike occurrence for Hodgkin-Huxley neuron.

        Args:
            t: Time of the spike (ms)
        """
        super()._handle_spike(t)
        # For HH model, spike handling is primarily done through the
        # integrate-and-fire mechanism in the base class
        # The actual spike shape emerges from the ion channel dynamics

    def get_biophysical_info(self) -> Dict[str, float]:
        """
        Get biophysical parameters of the neuron.

        Returns:
            Dictionary of biophysical parameters
        """
        return {
            "membrane_capacitance_uF_per_cm2": self.membrane_capacitance,
            "g_Na_mS_per_cm2": self.g_na,
            "E_Na_mV": self.e_na,
            "g_K_mS_per_cm2": self.g_k,
            "E_K_mV": self.e_k,
            "g_L_mS_per_cm2": self.g_l,
            "E_L_mV": self.e_l
        }

    def describe(self) -> str:
        """
        Get a description of the Hodgkin-Huxley neuron.

        Returns:
            String description of the neuron
        """
        return (f"HodgkinHuxleyNeuron(id={self.neuron_id}, "
                f"Cm={self.membrane_capacitance}µF/cm², "
                f"g_Na={self.g_na}mS/cm², E_Na={self.e_na}mV, "
                f"g_K={self.g_k}mS/cm², E_K={self.e_k}mV, "
                f"g_L={self.g_l}mS/cm², E_L={self.e_l}mV)")


# Factory function for easy neuron creation
def create_hodgkin_huxley_neuron(
    neuron_id: int = 0,
    **kwargs
) -> HodgkinHuxleyNeuron:
    """
    Factory function to create a Hodgkin-Huxley neuron with default parameters.

    Args:
        neuron_id: Unique identifier for the neuron
        **kwargs: Parameters to override defaults

    Returns:
        Configured HodgkinHuxleyNeuron instance
    """
    return HodgkinHuxleyNeuron(neuron_id=neuron_id, **kwargs)