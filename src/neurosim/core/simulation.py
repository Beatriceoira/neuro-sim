"""
Main simulation engine for the biological neuron simulator.

This module implements the core simulation loop that integrates neuronal
dynamics over time, handles events (spikes), and manages data recording.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from .state import SimulationState, NetworkState, NeuronState
from .integrators import euler_step, rk4_step, exponential_euler_step


class Simulation:
    """
    Main simulation class that manages the integration of neural dynamics.
    """

    def __init__(
        self,
        network_state: NetworkState = None,
        deriv_function: Callable = None,
        integrator: str = "euler",
        dt: float = 0.01,
        spike_threshold: float = -40.0,
        spike_reset: float = -65.0,
        refractory_period: float = 2.0
    ):
        """
        Initialize the simulation.
        """
        self.state = SimulationState()
        self.state.network_state = network_state or NetworkState()
        self.state.network_state.dt = dt

        self.deriv_function = deriv_function
        self.integrator = integrator
        self.dt = dt
        self.spike_threshold = spike_threshold
        self.spike_reset = spike_reset
        self.refractory_period = refractory_period

    def _detect_spikes(self, current_time: float) -> List[int]:
        """
        Detect spikes in the neuron population.
        """
        spiked_neurons = []
        for i, neuron_state in enumerate(self.state.network_state.neuron_states):
            if (neuron_state.refractory_remaining <= 0 and
                neuron_state.membrane_potential >= self.spike_threshold):

                neuron_state.spike_times.append(current_time)
                neuron_state.refractory_remaining = self.refractory_period
                spiked_neurons.append(i)
        return spiked_neurons

    def _apply_spike_effects(self, spiked_neurons: List[int]):
        """
        Apply effects of spikes (reset membrane potential, etc.).
        """
        for i in spiked_neurons:
            neuron_state = self.state.network_state.neuron_states[i]
            neuron_state.membrane_potential = self.spike_reset

    def _update_refractory(self, dt: float):
        """
        Update refractory period counters for all neurons.
        """
        for neuron_state in self.state.network_state.neuron_states:
            if neuron_state.refractory_remaining > 0:
                neuron_state.refractory_remaining = max(0, neuron_state.refractory_remaining - dt)

    def step(self, current_time: float) -> SimulationState:
        """
        Perform a single simulation step.
        """
        # 1. Detect spikes
        spiked_neurons = self._detect_spikes(current_time)

        # 2. Apply spike effects
        self._apply_spike_effects(spiked_neurons)

        # 3. Update refractory periods
        self._update_refractory(self.dt)

        # 4. Update state via deriv_function (for continuous dynamics)
        # Pass spiked neuron info so the derivative function can handle synaptic transmission
        if self.deriv_function:
            self.deriv_function(current_time, self.state.network_state, self.state, spiked_neurons)

        # Update global time
        self.state.network_state.global_time += self.dt
        return self.state

    def run(
        self,
        duration: float,
        dt: Optional[float] = None,
        record_variables: List[str] = None
    ) -> SimulationState:
        """
        Run the simulation for a specified duration.
        """
        if dt is not None:
            self.dt = dt
            self.state.network_state.dt = dt

        if record_variables:
            for var in record_variables:
                if var not in self.state.recorded_variables:
                    self.state.recorded_variables[var] = []

        n_steps = int(duration / self.dt)
        for step in range(n_steps):
            current_time = step * self.dt
            self.step(current_time)

            if record_variables:
                for var in record_variables:
                    if var == "voltage" and self.state.network_state.neuron_states:
                        # Record voltage of first neuron (could be averaged or selected)
                        self.state.record_variable(
                            var,
                            self.state.network_state.neuron_states[0].membrane_potential
                        )
        return self.state
