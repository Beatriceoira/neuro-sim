"""
Excitatory synapse model for the biological neuron simulator.

Implements AMPA-like excitatory synaptic currents with conductance-based dynamics.
"""

from typing import Dict
import numpy as np
from .base import BaseSynapse


class ExcitatorySynapse(BaseSynapse):
    """
    Excitatory synapse model (AMPA-like).

    Implements conductance-based excitatory synaptic currents:
    I_syn = g_syn * weight * (V - E_syn)
    where g_syn follows alpha-function or double-exponential kinetics.

    Represents glutamatergic excitatory synapses.
    """

    def __init__(
        self,
        synapse_id: str = "",
        weight: float = 1.0,           # Synaptic weight (dimensionless)
        delay: float = 0.0,            # Synaptic delay (ms)
        reversal_potential: float = 0.0, # Reversal potential (mV, typically 0 mV for AMPA)
        conductance: float = 1.0,      # Peak conductance (µS)
        rise_time: float = 2.0,        # Rise time constant (ms)
        decay_time: float = 10.0       # Decay time constant (ms)
    ):
        """
        Initialize the excitatory synapse.

        Args:
            synapse_id: Unique identifier for this synapse
            weight: Synaptic weight (dimensionless)
            delay: Synaptic delay (ms)
            reversal_potential: Reversal potential (mV)
            conductance: Peak synaptic conductance (µS)
            rise_time: Rise time constant (ms)
            decay_time: Decay time constant (ms)
        """
        super().__init__(synapse_id, weight, delay, reversal_potential)

        self.conductance = conductance      # g_peak in µS
        self.rise_time = rise_time          # τ_rise in ms
        self.decay_time = decay_time        # τ_decay in ms

        # State variables for synaptic conductance dynamics
        self.state = {
            'g': 0.0,       # Instantaneous conductance (µS)
            'x': 0.0,       # Rising phase variable
            'y': 0.0        # Decaying phase variable
        }

    def current(self, voltage: float, state: Dict[str, float]) -> float:
        """
        Compute the excitatory synaptic current.

        Implements: I_syn = g_syn * weight * (V - E_syn)

        Args:
            voltage: Membrane potential of postsynaptic neuron (mV)
            state: Dictionary containing 'g' conductance variable

        Returns:
            Synaptic current density (µA/cm²)
            Positive current is depolarizing (inward positive current)
        """
        g_syn = state.get('g', 0.0)
        # Convert conductance from µS to appropriate units if needed
        # For simplicity, assuming conductance is already in correct units
        current = g_syn * self.weight * (voltage - self.reversal_potential)
        return current

    def update_state(
        self,
        pre_synaptic_spike: bool,
        current_time: float,
        dt: float
    ) -> Dict[str, float]:
        """
        Update the excitatory synapse state.

        Implements dual-exponential model for synaptic conductance:
        dx/dt = -x/τ_rise + spike_input
        dy/dt = -y/τ_decay
        g_syn = conductance * (x - y)

        Args:
            pre_synaptic_spike: Whether a pre-synaptic spike occurred
            current_time: Current simulation time (ms)
            dt: Time step (ms)

        Returns:
            Dictionary of updated state variables
        """
        # Update state variables
        x = self.state['x']
        y = self.state['y']

        # Alpha-function approximation using dual exponential
        # dx/dt = -x/τ_rise + spike_input
        # dy/dt = -y/τ_decay
        x_dt = -x / self.rise_time
        y_dt = -y / self.decay_time

        x += x_dt * dt
        y += y_dt * dt

        # Add spike contribution if pre-synaptic spike occurred
        if pre_synaptic_spike:
            # Spike causes instantaneous rise in both x and y variables
            # x represents the activated state (jumps on spike, decays with τ_rise)
            # y represents the desensitized state (jumps on spike, decays with τ_decay)
            x += 1.0
            y += 1.0
            # Record spike time for plasticity calculations
            if not hasattr(self, 'spike_times'):
                self.spike_times = []
            self.spike_times.append(current_time)

        # Compute instantaneous conductance
        # g_syn = g_peak * (y - x)
        # Since x decays faster than y (τ_rise < τ_decay), y > x initially after spike
        # This creates a transient conductance peak
        g_syn = self.conductance * (y - x)
        # Ensure conductance is non-negative
        g_syn = max(0.0, g_syn)

        # Update state
        self.state['x'] = x
        self.state['y'] = y
        self.state['g'] = g_syn

        return self.state

    def get_info(self) -> Dict[str, float]:
        """
        Get information about this excitatory synapse.

        Returns:
            Dictionary of synapse parameters
        """
        info = super().get_info()
        info.update({
            "synapse_type": "excitatory",
            "conductance_µS": self.conductance,
            "rise_time_ms": self.rise_time,
            "decay_time_ms": self.decay_time,
            "reversal_potential_mV": self.reversal_potential  # Typically 0 mV for AMPA
        })
        return info

    def describe(self) -> str:
        """
        Get a description of the excitatory synapse.

        Returns:
            String description of the synapse
        """
        return f"ExcitatorySynapse(id={self.synapse_id}, weight={self.weight}, delay={self.delay}ms, g_peak={self.conductance}µS, E_syn={self.reversal_potential}mV)"