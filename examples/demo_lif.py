#!/usr/bin/env python3
"""
LIF neuron demonstration.

Runs a Leaky Integrate-and-Fire neuron with step current and reports
spike count and mean inter-spike interval.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from neurosim.neurons.lif import LIFNeuron, create_lif_neuron


def demo_lif_features():
    """Demonstrate LIF neuron features with varying parameters."""
    print("=" * 60)
    print("Demo: LIF Neuron Features")
    print("=" * 60)

    # Create LIF neurons with different parameters
    neurons = [
        create_lif_neuron(neuron_id=0, membrane_resistance=10.0, resting_potential=-65.0),
        create_lif_neuron(neuron_id=1, membrane_resistance=20.0, resting_potential=-60.0),
        create_lif_neuron(neuron_id=2, membrane_resistance=5.0, resting_potential=-70.0),
    ]

    # Simulate with different input currents
    for i, neuron in enumerate(neurons):
        neuron.state.external_current = 5.0 * (i + 1)  # 5, 10, 15 pA

        spike_times = []
        for step in range(500):  # 500 ms simulation
            t = float(step) * 0.1  # 0.1 ms dt
            if neuron.update(t, 0.1, neuron.state.external_current):
                spike_times.append(t)

        print(f"\nNeuron {i}:")
        print(f"  Parameters: Rm={neuron.membrane_resistance} MΩ, V_rest={neuron.resting_potential} mV")
        print(f"  Input current: {neuron.state.external_current} pA")
        print(f"  Spike count: {len(spike_times)}")
        print(f"  Mean ISI: {np.mean(np.diff(spike_times)):.1f} ms (if >1 spike)")


if __name__ == "__main__":
    demo_lif_features()