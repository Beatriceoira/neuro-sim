"""
Short-term plasticity synapse model for the biological neuron simulator.

Implements Tsodyks-Markram model of short-term plasticity including
facilitation, depression, and recovery dynamics.
"""

from typing import Dict
import numpy as np
from .base import BaseSynapse


class ShortTermPlasticitySynapse(BaseSynapse):
    """
    Short-term plasticity synapse using Tsodyks-Markram model.

    Implements dynamic synapses with facilitation and depression:
    - Utilization of synaptic resources (u) increases with pre-synaptic spikes (facilitation)
    - Available resources (R) decrease with vesicle release (depression)
    - Resources recover with time constant tau_rec

    Equations:
    du/dt = -u/tau_facilitation + U*(1-u)*spike
    dR/dt = (1-R)/tau_recovery - u*R*spike
    Released resources = u*R

    where spike is 1 if pre-synaptic spike occurred, 0 otherwise
    """

    def __init__(
        self,
        synapse_id: str = "",
        weight: float = 1.0,           # Synaptic weight (dimensionless)
        delay: float = 0.0,            # Synaptic delay (ms)
        reversal_potential: float = 0.0, # Reversal potential (mV)
        # Tsodyks-Markram parameters
        U: float = 0.5,                # Utilization parameter (0-1)
        tau_facilitation: float = 0.0, # Facilitation time constant (ms)
        tau_recovery: float = 800.0,   # Recovery time constant (ms)
        # Base synaptic parameters
        conductance: float = 1.0,      # Peak conductance (µS)
        rise_time: float = 2.0,        # Rise time constant (ms)
        decay_time: float = 10.0       # Decay time constant (ms)
    ):
        """
        Initialize the short-term plasticity synapse.

        Args:
            synapse_id: Unique identifier for this synapse
            weight: Synaptic weight (dimensionless)
            delay: Synaptic delay (ms)
            reversal_potential: Reversal potential (mV)
            U: Utilization parameter (fraction of resources released per spike)
            tau_facilitation: Facilitation time constant (ms)
            tau_recovery: Recovery time constant (ms)
            conductance: Peak synaptic conductance (µS)
            rise_time: Rise time constant (ms)
            decay_time: Decay time constant (ms)
        """
        super().__init__(synapse_id, weight, delay, reversal_potential)

        # Tsodyks-Markram parameters
        self.U = U                        # Utilization parameter
        self.tau_facilitation = tau_facilitation  # ms
        self.tau_recovery = tau_recovery    # ms

        # Base synaptic parameters
        self.conductance = conductance      # g_peak in µS
        self.rise_time = rise_time          # τ_rise in ms
        self.decay_time = decay_time        # τ_decay in ms

        # State variables for synaptic dynamics
        self.state = {
            'u': self.U,        # Fraction of resources utilized (release probability)
            'R': 1.0,           # Fraction of available resources
            'x': 0.0,           # Rising phase variable for conductance
            'y': 0.0            # Decaying phase variable for conductance
        }

        # For tracking spike times (used in some plasticity models)
        self.spike_times = []

    def current(self, voltage: float, state: Dict[str, float]) -> float:
        """
        Compute the synaptic current with short-term plasticity.

        Implements: I_syn = g_syn * weight * (V - E_syn)
        where g_syn depends on available resources and utilization

        Args:
            voltage: Membrane potential of postsynaptic neuron (mV)
            state: Dictionary containing state variables

        Returns:
            Synaptic current density (µA/cm²)
        """
        # Get state variables
        u = state.get('u', self.U)
        R = state.get('R', 1.0)
        g_syn = state.get('g', 0.0)

        # Effective synaptic strength incorporates short-term plasticity
        effective_weight = self.weight * u * R
        current = g_syn * effective_weight * (voltage - self.reversal_potential)
        return current

    def update_state(
        self,
        pre_synaptic_spike: bool,
        current_time: float,
        dt: float
    ) -> Dict[str, float]:
        """
        Update the short-term plasticity synapse state.

        Implements Tsodyks-Markram dynamics:
        du/dt = -u/tau_facilitation + U*(1-u)*spike
        dR/dt = (1-R)/tau_recovery - u*R*spike

        Also updates synaptic conductance using dual-exponential model.

        Args:
            pre_synaptic_spike: Whether a pre-synaptic spike occurred
            current_time: Current simulation time (ms)
            dt: Time step (ms)

        Returns:
            Dictionary of updated state variables
        """
        # Get current state values
        u = self.state['u']
        R = self.state['R']
        x = self.state['x']
        y = self.state['y']

        # Update short-term plasticity variables
        # du/dt = -u/tau_facilitation
        u_dt = -u / self.tau_facilitation if self.tau_facilitation > 0 else 0.0

        # dR/dt = (1-R)/tau_recovery
        R_dt = (1.0 - R) / self.tau_recovery

        # Add spike effects if pre-synaptic spike occurred
        if pre_synaptic_spike:
            # Facilitation: increase utilization
            u_spike_term = self.U * (1.0 - u)
            u_dt += u_spike_term / dt  # Scale by dt for integration

            # Depression: decrease available resources
            R_spike_term = u * R
            R_dt -= R_spike_term / dt  # Scale by dt for integration

            # Record spike time
            self.spike_times.append(current_time)

        # Update state variables
        u += u_dt * dt
        R += R_dt * dt

        # Ensure bounds
        u = max(0.0, min(1.0, u))
        R = max(0.0, min(1.0, R))

        # Update synaptic conductance (dual-exponential model)
        # dx/dt = -x/τ_rise
        # dy/dt = -y/τ_decay
        x_dt = -x / self.rise_time
        y_dt = -y / self.decay_time

        x += x_dt * dt
        y += y_dt * dt

        # Add spike contribution if pre-synaptic spike occurred
        if pre_synaptic_spike:
            x += 1.0  # Instantaneous rise in x variable

        # Compute instantaneous conductance
        # g_syn = g_peak * (y - x)
        g_syn = self.conductance * (y - x)
        # Ensure conductance is non-negative
        g_syn = max(0.0, g_syn)

        # Update state
        self.state['u'] = u
        self.state['R'] = R
        self.state['x'] = x
        self.state['y'] = y
        self.state['g'] = g_syn

        return self.state.copy()

    def get_info(self) -> Dict[str, float]:
        """
        Get information about this short-term plasticity synapse.

        Returns:
            Dictionary of synapse parameters
        """
        info = super().get_info()
        info.update({
            "synapse_type": "short_term_plasticity",
            "U": self.U,
            "tau_facilitation_ms": self.tau_facilitation,
            "tau_recovery_ms": self.tau_recovery,
            "conductance_µS": self.conductance,
            "rise_time_ms": self.rise_time,
            "decay_time_ms": self.decay_time,
            "reversal_potential_mV": self.reversal_potential
        })
        return info

    def describe(self) -> str:
        """
        Get a description of the short-term plasticity synapse.

        Returns:
            String description of the synapse
        """
        return (f"ShortTermPlasticitySynapse(id={self.synapse_id}, "
                f"weight={self.weight}, delay={self.delay}ms, "
                f"U={self.U}, τ_fac={self.tau_facilitation}ms, "
                f"τ_rec={self.tau_recovery}ms, "
                f"g_peak={self.conductance}µS, E_syn={self.reversal_potential}mV)")


# Factory function for easy synapse creation
def create_stp_synapse(
    synapse_id: str = "",
    **kwargs
) -> ShortTermPlasticitySynapse:
    """
    Factory function to create a short-term plasticity synapse.

    Args:
        synapse_id: Unique identifier for the synapse
        **kwargs: Parameters to override defaults

    Returns:
        Configured ShortTermPlasticitySynapse instance
    """
    return ShortTermPlasticitySynapse(synapse_id=synapse_id, **kwargs)