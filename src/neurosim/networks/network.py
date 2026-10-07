"""
Network simulation component for the biological neuron simulator.

This module implements the main Network class that manages populations of neurons,
their connections, and coordinates the simulation of multi-neuron systems.
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from ..core.state import NetworkState, SimulationState, NeuronState
from ..core.simulation import Simulation
from ..neurons.base import BaseNeuron
from ..synapses.base import BaseSynapse
from ..synapses.excitatory import ExcitatorySynapse
from ..synapses.inhibitory import InhibitorySynapse
from ..optimization.jit import detect_spikes, accumulate_synaptic_currents


class Network:
    """
    Biological neural network simulation.

    Manages populations of neurons and their synaptic connections.
    Supports various connectivity patterns and coordinate simulation.
    """

    def __init__(
        self,
        dt: float = 0.01,
        spike_threshold: float = -40.0,
        spike_reset: float = -65.0,
        refractory_period: float = 2.0
    ):
        """
        Initialize the network.

        Args:
            dt: Integration time step (ms)
            spike_threshold: Membrane potential threshold for spike detection (mV)
            spike_reset: Membrane potential reset value after spike (mV)
            refractory_period: Refractory period duration (ms)
        """
        self.dt = dt
        self.spike_threshold = spike_threshold
        self.spike_reset = spike_reset
        self.refractory_period = refractory_period

        # Network components
        self.populations: Dict[str, List[BaseNeuron]] = {}
        self.connections: List[Tuple[str, str, List['Connection']]] = []  # (source_pop, target_pop, connections)
        self.neuron_id_counter = 0
        # Fast lookup indexes (built lazily in add_population / connect)
        self._neuron_index: Dict[int, BaseNeuron] = {}
        self._incoming_index: Dict[int, List['Connection']] = {}

        # Simulation state
        self.network_state = NetworkState()
        self.simulation = Simulation(
            network_state=self.network_state,
            integrator="euler",
            dt=dt,
            spike_threshold=spike_threshold,
            spike_reset=spike_reset,
            refractory_period=refractory_period
        )

        # Set the derivative function for the simulation
        self.simulation.deriv_function = self._compute_network_derivatives

    def add_population(
        self,
        name: str,
        neuron_model: Callable[[int], BaseNeuron],
        size: int,
        **neuron_params
    ) -> List[BaseNeuron]:
        """
        Add a population of neurons to the network.

        Args:
            name: Name identifier for the population
            neuron_model: Function that creates a neuron instance (takes neuron_id)
            size: Number of neurons in the population
            **neuron_params: Parameters to pass to neuron_model

        Returns:
            List of created neurons
        """
        if name in self.populations:
            raise ValueError(f"Population '{name}' already exists")

        neurons = []
        for i in range(size):
            neuron_id = self.neuron_id_counter
            self.neuron_id_counter += 1
            neuron = neuron_model(neuron_id, **neuron_params)
            neurons.append(neuron)
            # Also add the neuron's state to the network state for simulation
            self.network_state.neuron_states.append(neuron.state)
            # Build fast lookup index incrementally: O(N) per population add
            self._neuron_index[len(self.network_state.neuron_states) - 1] = neuron

        self.populations[name] = neurons
        return neurons

    def get_population(self, name: str) -> List[BaseNeuron]:
        """
        Get a population by name.

        Args:
            name: Name of the population

        Returns:
            List of neurons in the population
        """
        if name not in self.populations:
            raise ValueError(f"Population '{name}' not found")
        return self.populations[name]

    def connect(
        self,
        source_population: str,
        target_population: str,
        connection_rule: Callable[[int, int], bool],
        synapse_factory: Callable[[], BaseSynapse],
        weight_range: Tuple[float, float] = (0.5, 1.0),
        delay_range: Tuple[float, float] = (1.0, 2.0)
    ) -> List['Connection']:
        """
        Create connections between two populations.

        Args:
            source_population: Name of source population
            target_population: Name of target population
            connection_rule: Function that takes (source_idx, target_idx) and returns True if connection should exist
            synapse_factory: Function that creates synapse instances
            weight_range: Tuple of (min_weight, max_weight) for random weights
            delay_range: Tuple of (min_delay, max_delay) for random delays

        Returns:
            List of created Connection objects
        """
        if source_population not in self.populations:
            raise ValueError(f"Source population '{source_population}' not found")
        if target_population not in self.populations:
            raise ValueError(f"Target population '{target_population}' not found")

        source_neurons = self.populations[source_population]
        target_neurons = self.populations[target_population]

        connections = []

        for i, source_neuron in enumerate(source_neurons):
            for j, target_neuron in enumerate(target_neurons):
                if connection_rule(i, j):
                    # Create connection with random parameters within ranges
                    weight = np.random.uniform(weight_range[0], weight_range[1])
                    delay = np.random.uniform(delay_range[0], delay_range[1])
                    synapse = synapse_factory()

                    connection = Connection(
                        source_neuron=source_neuron,
                        target_neuron=target_neuron,
                        weight=weight,
                        delay=delay,
                        synapse=synapse
                    )
                    connections.append(connection)

        # Store connections
        self.connections.append((source_population, target_population, connections))
        self._build_connection_index()
        return connections

    def all_to_all_connect(
        self,
        source_population: str,
        target_population: str,
        synapse_factory: Callable[[], BaseSynapse],
        weight: float = 1.0,
        delay: float = 1.0
    ) -> List['Connection']:
        """
        Create all-to-all connections between two populations.

        Args:
            source_population: Name of source population
            target_population: Name of target population
            synapse_factory: Function that creates synapse instances
            weight: Synaptic weight for all connections
            delay: Synaptic delay for all connections

        Returns:
            List of created Connection objects
        """
        def connection_rule(source_idx: int, target_idx: int) -> bool:
            return True  # Connect all source to all target

        return self.connect(
            source_population=source_population,
            target_population=target_population,
            connection_rule=connection_rule,
            synapse_factory=synapse_factory,
            weight_range=(weight, weight),
            delay_range=(delay, delay)
        )

    def random_connect(
        self,
        source_population: str,
        target_population: str,
        connection_probability: float,
        synapse_factory: Callable[[], BaseSynapse],
        weight_range: Tuple[float, float] = (0.5, 1.0),
        delay_range: Tuple[float, float] = (1.0, 2.0)
    ) -> List['Connection']:
        """
        Create random connections between two populations.

        Args:
            source_population: Name of source population
            target_population: Name of target population
            connection_probability: Probability of connection between any pair (0-1)
            synapse_factory: Function that creates synapse instances
            weight_range: Tuple of (min_weight, max_weight) for random weights
            delay_range: Tuple of (min_delay, max_delay) for random delays

        Returns:
            List of created Connection objects
        """
        def connection_rule(source_idx: int, target_idx: int) -> bool:
            return np.random.random() < connection_probability

        return self.connect(
            source_population=source_population,
            target_population=target_population,
            connection_rule=connection_rule,
            synapse_factory=synapse_factory,
            weight_range=weight_range,
            delay_range=delay_range
        )

    def _build_connection_index(self):
        """
        Build a fast lookup index mapping each neuron to its incoming connections.

        Uses the pre-built _neuron_index (network index -> neuron object) to map each
        connection's target neuron to its network index in O(1) per connection,
        avoiding the O(N_conn × N_neurons) linear scan.
        """
        # Build a reverse index from state object id to network index if needed.
        # NeuronState is unhashable (dataclass), so use id() as the key.
        if not hasattr(self, '_state_id_to_index') or not self._state_id_to_index:
            self._state_id_to_index = {
                id(ns): i for i, ns in enumerate(self.network_state.neuron_states)
            }

        self._incoming_index = {}
        for source_pop_name, target_pop_name, connections in self.connections:
            for connection in connections:
                target = connection.target_neuron
                target_idx = self._state_id_to_index.get(id(target.state))
                if target_idx is None:
                    continue
                self._incoming_index.setdefault(target_idx, []).append(connection)

    def _compute_network_derivatives(
        self,
        t: float,
        network_state: NetworkState,
        sim_state: SimulationState,
        spiked_neurons: List[int] = None
    ) -> None:
        """
        Compute derivatives for the entire network.

        This function is called by the simulation engine to update neuron states.
        It computes synaptic inputs from all connections and applies them to neurons.
        Modifies the network_state in-place.

        Args:
            t: Current time (ms)
            network_state: Current network state (modified in-place)
            sim_state: Overall simulation state
            spiked_neurons: Optional list of neuron indices that spiked at this time step
        """
        # Process all connections to compute synaptic inputs for each neuron
        # We'll accumulate synaptic currents for each neuron (in pA)
        synaptic_currents = [0.0] * len(network_state.neuron_states)

        spike_transmissions = 0
        total_synaptic_current_exc = 0.0
        total_synaptic_current_inh = 0.0

        # Use pre-built incoming index: O(N_conn) instead of O(N_conn × N_neurons)
        # Build state_id->index map once per call if needed
        if not hasattr(self, '_state_id_cache') or self._state_id_cache.get('timestamp') != t:
            self._state_id_cache = {
                'timestamp': t,
                'mapping': {
                    id(ns): i for i, ns in enumerate(network_state.neuron_states)
                }
            }
        state_id_to_index = self._state_id_cache['mapping']

        for target_idx, connections in self._incoming_index.items():
            v = network_state.neuron_states[target_idx].membrane_potential
            for connection in connections:
                # Check if source neuron spiked at this time step
                source_spiked = False
                if spiked_neurons is not None:
                    # Find the index of the source neuron in the network state
                    source_state_id = id(connection.source_neuron.state)
                    if source_state_id in state_id_to_index and state_id_to_index[source_state_id] in spiked_neurons:
                        source_spiked = True
                else:
                    # Fallback: check spike_times
                    if (connection.source_neuron.state.refractory_remaining > 0 and
                        connection.source_neuron.state.spike_times and
                        abs(connection.source_neuron.state.spike_times[-1] - t) < self.dt):
                        source_spiked = True

                # Update connection and check for spike transmission
                spike_transmitted, synaptic_current_exc, synaptic_current_inh = connection.update(
                    t, self.dt, source_spiked=source_spiked
                )

                if spike_transmitted:
                    spike_transmissions += 1

                # Compute synaptic current using the synapse's conductance
                # I_syn = g_syn * weight * (V - E_syn)
                syn_state = connection.synapse.state
                g_syn = syn_state.get('g', 0.0)

                if isinstance(connection.synapse, ExcitatorySynapse):
                    # E_syn ≈ 0 mV for AMPA
                    # Excitatory current: I = g * (E_syn - V) which is positive when V < E_syn
                    e_syn = connection.synapse.reversal_potential
                    current = g_syn * connection.weight * (e_syn - v)
                    synaptic_currents[target_idx] += current
                    total_synaptic_current_exc += current
                elif isinstance(connection.synapse, InhibitorySynapse):
                    # E_syn ≈ -70 mV for GABA
                    # Inhibitory current: I = g * (V - E_syn) which is negative when V > E_syn
                    e_syn = connection.synapse.reversal_potential
                    current = g_syn * connection.weight * (v - e_syn)
                    synaptic_currents[target_idx] += current
                    total_synaptic_current_inh += abs(current)
                else:
                    # Default to excitatory
                    e_syn = connection.synapse.reversal_potential
                    current = g_syn * connection.weight * (e_syn - v)
                    synaptic_currents[target_idx] += current
                    total_synaptic_current_exc += current

        # Now update each neuron's state by computing its derivatives
        dt = self.dt  # Use the network's time step

        for i, neuron_state in enumerate(network_state.neuron_states):
            # Look up neuron object via pre-built index (O(1) per neuron)
            neuron_obj = self._neuron_index.get(i)
            if neuron_obj is None:
                continue

            # Combine the neuron's own external current with synaptic currents
            neuron_external = getattr(neuron_state, 'external_current', 0.0)
            total_current = neuron_external + synaptic_currents[i]

            dv_dt, other_derivs = neuron_obj.compute_derivatives(
                t,
                external_current=total_current,  # External stimulus + synaptic current
                synaptic_inputs={}  # No additional synaptic inputs dict needed
            )

            # Update the neuron's membrane potential using Euler integration
            neuron_state.membrane_potential += dv_dt * dt

            # Update any other state variables from other_derivs
            for var_name, var_deriv in other_derivs.items():
                if hasattr(neuron_state, var_name):
                    current_val = getattr(neuron_state, var_name)
                    setattr(neuron_state, var_name, current_val + var_deriv * dt)
                else:
                    # Store in gating_variables or similar dictionary
                    if var_name not in neuron_state.gating_variables:
                        neuron_state.gating_variables[var_name] = 0.0
                    neuron_state.gating_variables[var_name] += var_deriv * dt

        # Store diagnostic information in sim_state for accessibility
        sim_state.parameters['last_diagnostic'] = {
            'num_neurons': len(network_state.neuron_states),
            'num_connections': sum(len(conns) for _, _, conns in self.connections),
            'spike_transmissions': spike_transmissions,
            'total_synaptic_current_exc': total_synaptic_current_exc,
            'total_synaptic_current_inh': total_synaptic_current_inh,
            'time': t
        }

    def step(self, current_time: float) -> SimulationState:
        """
        Perform a single network simulation step.

        Uses the pre-built connection index and neuron index to avoid
        O(N²) lookups, then delegates to the generic simulation engine
        for spike detection and state integration.

        Args:
            current_time: Current simulation time (ms)

        Returns:
            Updated simulation state
        """
        return self.simulation.step(current_time)

    def run(
        self,
        duration: float,
        dt: Optional[float] = None
    ) -> SimulationState:
        """
        Run the network simulation for a specified duration.

        Args:
            duration: Simulation duration (ms)
            dt: Time step (ms). If None, uses the network's dt

        Returns:
            Final simulation state
        """
        return self.simulation.run(duration=duration, dt=dt)

    def get_population_spikes(self, population_name: str) -> List[np.ndarray]:
        """
        Get spike trains for all neurons in a population.

        Args:
            population_name: Name of the population

        Returns:
            List of numpy arrays containing spike times for each neuron
        """
        neurons = self.get_population(population_name)
        return [np.array(neuron.state.spike_times) for neuron in neurons]

    def get_network_info(self) -> Dict:
        """
        Get information about the network structure.

        Returns:
            Dictionary containing network information
        """
        info = {
            'populations': {},
            'connections': [],
            'total_neurons': 0,
            'total_connections': 0
        }

        for pop_name, neurons in self.populations.items():
            info['populations'][pop_name] = len(neurons)
            info['total_neurons'] += len(neurons)

        for source_pop, target_pop, conns in self.connections:
            info['connections'].append({
                'source': source_pop,
                'target': target_pop,
                'num_connections': len(conns)
            })
            info['total_connections'] += len(conns)

        return info


class Connection:
    """
    Represents a synaptic connection between two neurons.

    Attributes:
        source_neuron: The pre-synaptic neuron
        target_neuron: The post-synaptic neuron
        weight: Synaptic weight (dimensionless)
        delay: Synaptic delay (ms)
        synapse: Synapse model instance
    """

    def __init__(
        self,
        source_neuron: BaseNeuron,
        target_neuron: BaseNeuron,
        weight: float = 1.0,
        delay: float = 1.0,
        synapse: Optional[BaseSynapse] = None
    ):
        """
        Initialize a connection.

        Args:
            source_neuron: Pre-synaptic neuron
            target_neuron: Post-synaptic neuron
            weight: Synaptic weight
            delay: Synaptic delay (ms)
            synapse: Synapse model instance (if None, creates a basic synapse)
        """
        self.source_neuron = source_neuron
        self.target_neuron = target_neuron
        self.weight = weight
        self.delay = delay
        self.synapse = synapse

        # For delayed synaptic transmission
        self.spike_queue: List[float] = []  # Queued spike times accounting for delay
        # Track the last time we updated the synapse state
        self.last_update_time = 0.0

    def update(
        self,
        current_time: float,
        dt: float,
        source_spiked: bool = False
    ) -> Tuple[bool, float, float]:
        """
        Update the connection state and check for transmitted spikes.

        Args:
            current_time: Current simulation time (ms)
            dt: Time step (ms)
            source_spiked: Whether the source neuron spiked at this time step

        Returns:
            Tuple of (spike_transmitted, excitatory_current, inhibitory_current)
            where spike_transmitted is True if a spike was transmitted this step
            and the currents are the synaptic inputs to target neuron
        """
        # Check if source neuron spiked - use the provided flag or check spike_times
        if not source_spiked:
            # Fallback: check if source neuron just spiked by looking at spike_times
            if (self.source_neuron.state.refractory_remaining > 0 and
                self.source_neuron.state.spike_times and
                abs(self.source_neuron.state.spike_times[-1] - current_time) < dt):
                source_spiked = True

        if source_spiked:
            # Add spike to queue with delay
            self.spike_queue.append(current_time + self.delay)

        # Process delayed spikes - transmit any spikes whose delay time has passed
        synaptic_current_exc = 0.0
        synaptic_current_inh = 0.0
        spike_transmitted = False

        # Remove spikes that are ready to be transmitted (those with time <= current_time)
        ready_spikes = [t for t in self.spike_queue if t <= current_time]
        self.spike_queue = [t for t in self.spike_queue if t > current_time]

        if ready_spikes:
            spike_transmitted = True
            # For each transmitted spike, update the synapse state
            # This creates a conductance that persists over time
            for _ in ready_spikes:
                self.synapse.update_state(
                    pre_synaptic_spike=True,
                    current_time=current_time,
                    dt=dt
                )
        else:
            # Update synapse state for conductance decay (no spikes)
            self.synapse.update_state(
                pre_synaptic_spike=False,
                current_time=current_time,
                dt=dt
            )

        # Compute the synaptic current based on the synapse's current state
        # Get the synapse's conductance
        syn_state = self.synapse.state
        g_syn = syn_state.get('g', 0.0)

        # Compute current based on synapse type
        # Use a simple model where the synaptic current is proportional to the conductance
        # Scale by weight and a factor to match typical LIF neuron dynamics
        if isinstance(self.synapse, ExcitatorySynapse):
            # Excitatory current: positive (depolarizing)
            # Scale to match LIF neuron expectations (pA)
            synaptic_current_exc = g_syn * self.weight * 10.0  # Scaling factor
        elif isinstance(self.synapse, InhibitorySynapse):
            # Inhibitory current: negative (hyperpolarizing)
            synaptic_current_inh = g_syn * self.weight * 10.0  # Scaling factor
        else:
            # Default to excitatory
            synaptic_current_exc = g_syn * self.weight * 10.0

        return spike_transmitted, synaptic_current_exc, synaptic_current_inh

    def describe(self) -> str:
        """
        Get a description of the connection.

        Returns:
            String description of the connection
        """
        return (f"Connection({self.source_neuron.describe()} -> "
                f"{self.target_neuron.describe()}, "
                f"weight={self.weight:.3f}, delay={self.delay:.1f}ms)")