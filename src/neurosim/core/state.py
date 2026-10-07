"""
State management for the biological neuron simulator.

This module defines the core state containers for storing simulation variables
such as membrane potential, ion channel states, synaptic conductances, etc.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np


@dataclass
class NeuronState:
    """
    State variables for a single neuron.

    Attributes:
        membrane_potential: Membrane potential (mV)
        recovery_variable: Recovery variable for adaptive models (dimensionless)
        gating_variables: Dictionary of ion channel gating variables
        ionic_currents: Dictionary of ionic current densities (µA/cm²)
        synaptic_conductances: Dictionary of synaptic conductances (mS/cm²)
        spike_times: List of spike times (ms)
        refractory_remaining: Time remaining in refractory period (ms)
    """
    membrane_potential: float = -65.0  # mV, resting potential
    external_current: float = 0.0      # pA, externally applied current
    recovery_variable: float = 0.0     # dimensionless
    gating_variables: Dict[str, float] = field(default_factory=dict)
    ionic_currents: Dict[str, float] = field(default_factory=dict)
    synaptic_conductances: Dict[str, float] = field(default_factory=dict)
    spike_times: List[float] = field(default_factory=list)
    refractory_remaining: float = 0.0  # ms


@dataclass
class NetworkState:
    """
    State variables for a network of neurons.

    Attributes:
        neuron_states: List of NeuronState objects for each neuron
        connection_weights: Matrix of synaptic weights between neurons
        connection_delays: Matrix of synaptic delays between neurons (ms)
        global_time: Current simulation time (ms)
        dt: Integration time step (ms)
    """
    neuron_states: List[NeuronState] = field(default_factory=list)
    connection_weights: Optional[np.ndarray] = None
    connection_delays: Optional[np.ndarray] = None
    global_time: float = 0.0  # ms
    dt: float = 0.01          # ms


@dataclass
class SimulationState:
    """
    Overall simulation state.

    Attributes:
        network_state: State of the neural network
        recorded_variables: Dictionary of recorded time series data
        event_queue: Queue of upcoming events (spikes, etc.)
        parameters: Dictionary of model parameters
    """
    network_state: NetworkState = field(default_factory=NetworkState)
    recorded_variables: Dict[str, List[float]] = field(default_factory=dict)
    event_queue: List[tuple] = field(default_factory=list)  # (time, type, data)
    parameters: Dict[str, Union[float, int, str]] = field(default_factory=dict)

    def record_variable(self, name: str, value: float):
        """Record a variable value at the current time step."""
        if name not in self.recorded_variables:
            self.recorded_variables[name] = []
        self.recorded_variables[name].append(value)

    def get_recorded_variable(self, name: str) -> np.ndarray:
        """Get recorded variable as numpy array."""
        if name not in self.recorded_variables:
            raise KeyError(f"Variable '{name}' not recorded")
        return np.array(self.recorded_variables[name])