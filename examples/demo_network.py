#!/usr/bin/env python3
"""
Network simulation demonstration.

Builds a small excitatory/inhibitory network, runs it, and reports
per-population spike statistics.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from neurosim.neurons.lif import LIFNeuron
from neurosim.networks.network import Network
from neurosim.synapses.excitatory import ExcitatorySynapse
from neurosim.synapses.inhibitory import InhibitorySynapse


def demo_network_simulation():
    """Demonstrate network simulation features."""
    print("=" * 60)
    print("Demo: Network Simulation Features")
    print("=" * 60)

    # Create a simple network
    net = Network(dt=0.5, spike_threshold=-50.0, spike_reset=-65.0)

    # Add different types of neurons
    exc_neurons = net.add_population('exc', LIFNeuron, size=5)
    inh_neurons = net.add_population('inh', LIFNeuron, size=2)

    # Inject different currents
    for neuron in exc_neurons:
        neuron.state.external_current = 8.0
    for neuron in inh_neurons:
        neuron.state.external_current = 4.0

    # Create heterogeneous connections
    # All-to-all excitation
    net.all_to_all_connect('exc', 'exc', lambda: ExcitatorySynapse(),
                           weight=0.5, delay=1.0)

    # Random excitation to inhibition
    net.random_connect('exc', 'inh', connection_probability=0.7,
                       synapse_factory=lambda: ExcitatorySynapse(),
                       weight_range=(0.3, 0.8), delay_range=(1.0, 3.0))

    # All-to-all inhibition back to excitation
    net.all_to_all_connect('inh', 'exc', lambda: InhibitorySynapse(),
                           weight=0.8, delay=2.0)

    # Run simulation
    print("\nSimulating network with:")
    print(f"  - {len(exc_neurons)} excitatory neurons")
    print(f"  - {len(inh_neurons)} inhibitory neurons")
    print(f"  - {len(net.connections)} connection types")

    for step in range(200):  # 200 time steps
        t = float(step) * net.dt
        net.step(t)

    # Analyze network activity
    print("\nNetwork Activity Summary:")
    for pop_name in ['exc', 'inh']:
        neurons = net.get_population(pop_name)
        spike_counts = [len(n.state.spike_times) for n in neurons]
        print(f"  {pop_name.capitalize()} neurons:")
        print(f"    Total spikes: {sum(spike_counts)}")
        print(f"    Mean spikes per neuron: {np.mean(spike_counts):.1f}")
        print(f"    Max spikes per neuron: {np.max(spike_counts)}")


if __name__ == "__main__":
    demo_network_simulation()