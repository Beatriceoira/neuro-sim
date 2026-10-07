#!/usr/bin/env python3
"""
Hodgkin-Huxley neuron demonstration.

Runs a multi-compartment HH neuron with step current and reports
spike count and firing rate.
"""

import sys
sys.path.insert(0, 'src')

from neurosim.neurons.multi_compartment import MultiCompartmentNeuron
from neurosim.neurons.morphology import create_ball_and_stick
from neurosim.channels.channel_models import create_hh_channels
from neurosim.channels.leak import LeakChannel


def demo_hh_features():
    """Demonstrate Hodgkin-Huxley neuron features."""
    print("=" * 60)
    print("Demo: Hodgkin-Huxley Neuron Features")
    print("=" * 60)

    # Create multi-compartment HH neuron
    morph = create_ball_and_stick(soma_diameter=20, dendrite_length=200)
    neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
    soma = morph.get_soma()

    # Add HH channels
    hh_channels = create_hh_channels()
    for ch in hh_channels.values():
        neuron.add_channel_to_compartment(soma.compartment_id, ch)
    neuron.add_channel_to_compartment(soma.compartment_id, LeakChannel())

    # Simulate with step current
    step_duration = 200.0  # ms
    step_amplitude = 20.0  # pA

    for step in range(int(step_duration / 0.1)):
        t = float(step) * 0.1
        if t > 50 and t < 150:  # Step from 50-150 ms
            soma.external_current = step_amplitude
        else:
            soma.external_current = 0.0
        neuron.step(t)

    # Analyze results
    spike_times = neuron.state.spike_times
    print(f"\nHH Neuron:")
    print(f"  Morphology: Ball-and-stick, soma={morph.get_soma().compartment_id}")
    print(f"  Step current: {step_amplitude} pA from 50-150 ms")
    print(f"  Spike count: {len(spike_times)}")
    if len(spike_times) > 1:
        firing_rate = len(spike_times) / (step_duration / 1000.0)
        print(f"  Firing rate: {firing_rate:.1f} Hz")


if __name__ == "__main__":
    demo_hh_features()