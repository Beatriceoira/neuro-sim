"""
Spike-timing dependent plasticity (STDP) synapse model for the biological neuron simulator.

Implements STDP plasticity where synaptic weight changes depend on the timing
of pre- and post-synaptic spikes:

If pre-synaptic spike occurs before post-synaptic spike: LTP (potentiation)
If post-synaptic spike occurs before pre-synaptic spike: LTD (depression)

The weight change follows an exponential learning window:
Δw = A_plus * exp(-Δt/tau_plus)   for Δt > 0 (pre before post)
Δw = -A_minus * exp(Δt/tau_minus)  for Δt < 0 (post before pre)

where Δt = t_post - t_pre
"""

from typing import Dict, List
import numpy as np
from .base import BaseSynapse


class STDSynapse(BaseSynapse):
    """
    Spike-timing dependent plasticity (STDP) synapse.

    Implements Hebbian learning rule where synaptic strength changes based on
    the relative timing of pre- and post-synaptic spikes:
    - If pre-synaptic neuron fires before post-synaptic neuron: Long-Term Potentiation (LTP)
    - If post-synaptic neuron fires before pre-synaptic neuron: Long-Term Depression (LTD)

    Uses exponential learning windows:
    Δw = A+ * exp(-Δt/τ+)   for Δt > 0
    Δw = -A- * exp(Δt/τ-)   for Δt < 0
    """

    def __init__(
        self,
        synapse_id: str = "",
        weight: float = 1.0,           # Initial synaptic weight (dimensionless)
        delay: float = 0.0,            # Synaptic delay (ms)
        reversal_potential: float = 0.0, # Reversal potential (mV)
        # STDP parameters
        A_plus: float = 0.01,          # LTP learning rate
        A_minus: float = 0.012,        # LTD learning rate (usually slightly larger than A_plus)
        tau_plus: float = 20.0,        # LTP time constant (ms)
        tau_minus: float = 20.0,       # LTD time constant (ms)
        weight_min: float = 0.0,       # Minimum synaptic weight
        weight_max: float = 4.0,       # Maximum synaptic weight
        # Base synaptic parameters
        conductance: float = 1.0,      # Peak conductance (µS)
        rise_time: float = 2.0,        # Rise time constant (ms)
        decay_time: float = 10.0       # Decay time constant (ms)
    ):
        """
        Initialize the STDP synapse.

        Args:
            synapse_id: Unique identifier for this synapse
            weight: Initial synaptic weight (dimensionless)
            delay: Synaptic delay (ms)
            reversal_potential: Reversal potential (mV)
            A_plus: LTP learning rate
            A_minus: LTD learning rate
            tau_plus: LTP time constant (ms)
            tau_minus: LTD time constant (ms)
            weight_min: Minimum synaptic weight
            weight_max: Maximum synaptic weight
            conductance: Peak synaptic conductance (µS)
            rise_time: Rise time constant (ms)
            decay_time: Decay time constant (ms)
        """
        super().__init__(synapse_id, weight, delay, reversal_potential)

        # STDP parameters
        self.A_plus = A_plus           # LTP learning rate
        self.A_minus = A_minus         # LTD learning rate
        self.tau_plus = tau_plus       # LTP time constant (ms)
        self.tau_minus = tau_minus     # LTD time constant (ms)
        self.weight_min = weight_min   # Minimum weight
        self.weight_max = weight_max   # Maximum weight

        # Base synaptic parameters
        self.conductance = conductance      # g_peak in µS
        self.rise_time = rise_time          # τ_rise in ms
        self.decay_time = decay_time        # τ_decay in ms

        # State variables for synaptic dynamics
        self.state = {
            'g': 0.0,       # Instantaneous conductance (µS)
            'x': 0.0,       # Rising phase variable
            'y': 0.0        # Decaying phase variable
        }

        # Spike timing tracking for STDP
        self.pre_synaptic_spike_times = []  # Times of pre-synaptic spikes
        self.post_synaptic_spike_times = [] # Times of post-synaptic spikes

    def current(self, voltage: float, state: Dict[str, float]) -> float:
        """
        Compute the synaptic current.

        Implements: I_syn = g_syn * weight * (V - E_syn)

        Args:
            voltage: Membrane potential of postsynaptic neuron (mV)
            state: Dictionary containing 'g' conductance variable

        Returns:
            Synaptic current density (µA/cm²)
        """
        g_syn = state.get('g', 0.0)
        current = g_syn * self.weight * (voltage - self.reversal_potential)
        return current

    def update_state(
        self,
        pre_synaptic_spike: bool,
        post_synaptic_spike: bool,
        current_time: float,
        dt: float
    ) -> Dict[str, float]:
        """
        Update the STDP synapse state.

        Updates synaptic conductance and applies STDP learning rule based on
        spike timing.

        Args:
            pre_synaptic_spike: Whether a pre-synaptic spike occurred
            post_synaptic_spike: Whether a post-synaptic spike occurred
            current_time: Current simulation time (ms)
            dt: Time step (ms)

        Returns:
            Dictionary of updated state variables
        """
        # Update synaptic conductance (dual-exponential model)
        x = self.state['x']
        y = self.state['y']

        # dx/dt = -x/τ_rise
        # dy/dt = -y/τ_decay
        x_dt = -x / self.rise_time
        y_dt = -y / self.decay_time

        x += x_dt * dt
        y += y_dt * dt

        # Add spike contributions if spikes occurred
        if pre_synaptic_spike:
            x += 1.0  # Instantaneous rise in x variable
            self.pre_synaptic_spike_times.append(current_time)

        # Compute instantaneous conductance
        # g_syn = g_peak * (y - x)
        g_syn = self.conductance * (y - x)
        # Ensure conductance is non-negative
        g_syn = max(0.0, g_syn)

        # Update state variables
        self.state['x'] = x
        self.state['y'] = y
        self.state['g'] = g_syn

        # Apply STDP learning rule
        self._apply_stdp_rule(pre_synaptic_spike, post_synaptic_spike, current_time)

        # Track post-synaptic spikes
        if post_synaptic_spike:
            self.post_synaptic_spike_times.append(current_time)

        # Clean up old spike times unconditionally so the lists stay bounded
        # even if the learning-rule path above is skipped or short-circuited.
        self._cleanup_spike_times(current_time)

        return self.state.copy()

    def _apply_stdp_rule(
        self,
        pre_synaptic_spike: bool,
        post_synaptic_spike: bool,
        current_time: float
    ) -> None:
        """
        Apply the STDP learning rule to update synaptic weight.

        Args:
            pre_synaptic_spike: Whether a pre-synaptic spike occurred at current_time
            post_synaptic_spike: Whether a post-synaptic spike occurred at current_time
            current_time: Current simulation time (ms)
        """
        if post_synaptic_spike:
            # Post-synaptic spike occurred: check for recent pre-synaptic spikes
            # For each recent pre-synaptic spike, apply LTP
            for t_pre in self.pre_synaptic_spike_times:
                # Only consider spikes within a reasonable time window
                if current_time - t_pre > 5 * self.tau_plus:  # 5 time constants cutoff
                    continue

                delta_t = current_time - t_pre  # t_post - t_pre
                if delta_t > 0:  # Pre before post -> LTP
                    dw = self.A_plus * np.exp(-delta_t / self.tau_plus)
                    self.weight = min(self.weight_max, self.weight + dw)

        if pre_synaptic_spike:
            # Pre-synaptic spike occurred: check for recent post-synaptic spikes
            # For each recent post-synaptic spike, apply LTD
            for t_post in self.post_synaptic_spike_times:
                # Only consider spikes within a reasonable time window
                if abs(current_time - t_post) > 5 * self.tau_minus:  # 5 time constants cutoff
                    continue

                delta_t = t_post - current_time  # t_post - t_pre
                if delta_t < 0:  # Post before pre -> LTD
                    dw = -self.A_minus * np.exp(delta_t / self.tau_minus)  # delta_t is negative
                    self.weight = max(self.weight_min, self.weight + dw)

    def _cleanup_spike_times(self, current_time: float):
        """
        Remove old spike times to prevent unbounded memory growth.

        Args:
            current_time: Current simulation time (ms)
        """
        # Keep only spikes within 5 time constants of the current time
        cutoff_time = current_time - 5 * max(self.tau_plus, self.tau_minus)

        self.pre_synaptic_spike_times = [
            t for t in self.pre_synaptic_spike_times if t > cutoff_time
        ]
        self.post_synaptic_spike_times = [
            t for t in self.post_synaptic_spike_times if t > cutoff_time
        ]

    def get_info(self) -> Dict[str, float]:
        """
        Get information about this STDP synapse.

        Returns:
            Dictionary of synapse parameters
        """
        info = super().get_info()
        info.update({
            "synapse_type": "stdp",
            "A_plus": self.A_plus,
            "A_minus": self.A_minus,
            "tau_plus_ms": self.tau_plus,
            "tau_minus_ms": self.tau_minus,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "conductance_µS": self.conductance,
            "rise_time_ms": self.rise_time,
            "decay_time_ms": self.decay_time,
            "reversal_potential_mV": self.reversal_potential,
            "current_weight": self.weight
        })
        return info

    def describe(self) -> str:
        """
        Get a description of the STDP synapse.

        Returns:
            String description of the synapse
        """
        return (f"STDSynapse(id={self.synapse_id}, weight={self.weight:.3f}, "
                f"delay={self.delay}ms, A+={self.A_plus}, A-={self.A_minus}, "
                f"τ+={self.tau_plus}ms, τ-={self.tau_minus}ms, "
                f"g_peak={self.conductance}µS, E_syn={self.reversal_potential}mV)")


# Factory function for easy synapse creation
def create_stdp_synapse(
    synapse_id: str = "",
    **kwargs
) -> STDSynapse:
    """
    Factory function to create an STDP synapse.

    Args:
        synapse_id: Unique identifier for the synapse
        **kwargs: Parameters to override defaults

    Returns:
        Configured STDSynapse instance
    """
    return STDSynapse(synapse_id=synapse_id, **kwargs)