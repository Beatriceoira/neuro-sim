#!/usr/bin/env python3
"""
Izhikevich neuron demonstration.

Runs Izhikevich neurons with different parameter presets and reports
spike counts, mean ISIs, and ISI coefficients of variation.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from neurosim.neurons.izhikevich import IzhikevichNeuron


def demo_izhikevich_features():
    """Demonstrate Izhikevich neuron features."""
    print("=" * 60)
    print("Demo: Izhikevich Neuron Features")
    print("=" * 60)

    # Create Izhikevich neurons with different parameters
    presets = [
        (0.02, 0.2, -65.0, 8.0),    # Regular spiking
        (0.02, 0.2, -50.0, 2.0),    # Intrinsic bursting
        (0.1, -0.1, -65.0, 0.0),    # Chattering
    ]

    for i, (a, b, c, d) in enumerate(presets):
        neuron = IzhikevichNeuron(neuron_id=i, a=a, b=b, c=c, d=d)

        spike_times = []
        for step in range(500):  # 500 ms
            t = float(step) * 0.1
            if neuron.update(t, 0.1, 10.0):
                spike_times.append(t)

        print(f"\nIzhikevich Neuron {i}:")
        print(f"  Parameters: a={a}, b={b}, c={c}, d={d}")
        print(f"  Input current: 10 pA")
        print(f"  Spike count: {len(spike_times)}")
        if len(spike_times) > 1:
            mean_isi = np.mean(np.diff(spike_times))
            cv_isi = np.std(np.diff(spike_times)) / mean_isi
            print(f"  Mean ISI: {mean_isi:.1f} ms, CV: {cv_isi:.2f}")


if __name__ == "__main__":
    demo_izhikevich_features()