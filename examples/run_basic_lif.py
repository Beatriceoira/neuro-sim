"""
Example script to run a basic LIF neuron simulation.

Demonstrates:
- Creating an LIF neuron with typical parameters
- Applying a step current stimulus
- Running the Simulation engine
- Recording the voltage trace
- Analyzing the resulting spike train
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt

from neurosim.neurons.lif import LIFNeuron
from neurosim.stimuli.step import StepCurrent
from neurosim.core.simulation import Simulation
from neurosim.analysis.spikes import analyze_spike_train


def run_basic_lif_experiment():
    """Run a basic LIF neuron experiment with step current."""

    # Create LIF neuron
    neuron = LIFNeuron(
        membrane_resistance=10.0,      # MΩ
        membrane_time_constant=20.0,   # ms
        resting_potential=-65.0,       # mV
        threshold_potential=-50.0,     # mV
        reset_potential=-65.0,         # mV
        refractory_period=2.0          # ms
    )

    # Create step current stimulus
    stimulus = StepCurrent(
        amplitude=10.0,    # pA
        start_time=50.0,   # ms
        end_time=150.0     # ms
    )

    # Set up simulation - the deriv_func updates the neuron's state directly
    def deriv_func(t, net_state, sim_state, spiked_neurons=None):
        # Get external current from stimulus
        I_ext = stimulus.get_current(t)
        # Compute neuron derivatives
        dv_dt, other_derivs = neuron.compute_derivatives(t, external_current=I_ext)
        # Apply Euler integration to update the membrane potential
        neuron.state.membrane_potential += dv_dt * net_state.dt

    sim = Simulation(
        deriv_function=deriv_func,
        dt=0.01,
        spike_threshold=-50.0,
        spike_reset=-65.0,
        refractory_period=2.0
    )

    # The Simulation engine manages NeuronState objects, so we register the
    # neuron's state (not the neuron itself) in the network state.
    sim.state.network_state.neuron_states = [neuron.state]

    # Run simulation, recording the voltage trace
    print("Running LIF neuron simulation...")
    sim.run(duration=200.0, record_variables=['voltage'])

    # Extract results
    voltage_trace = np.array(sim.state.get_recorded_variable('voltage'))
    spike_times = np.array(neuron.state.spike_times)
    time_points = np.arange(0, 200.0, 0.01)  # ms

    # Analyze spike train
    if len(spike_times) > 0:
        spike_analysis = analyze_spike_train(spike_times, duration=200.0)
        print(f"Number of spikes: {spike_analysis['n_spikes']}")
        print(f"Mean firing rate: {spike_analysis['mean_firing_rate_Hz']:.2f} Hz")
        print(f"CV of ISI: {spike_analysis['cv_isi']:.3f}")
    else:
        print("No spikes detected")
        spike_analysis = {'n_spikes': 0}

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Voltage trace
    axes[0].plot(time_points, voltage_trace, 'b-', linewidth=1)
    axes[0].axhline(y=neuron.threshold_potential, color='r', linestyle='--', alpha=0.7, label='Threshold')
    axes[0].axhline(y=neuron.reset_potential, color='g', linestyle='--', alpha=0.7, label='Reset')
    axes[0].set_ylabel('Membrane Potential (mV)')
    axes[0].set_title('LIF Neuron Response to Step Current')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark spikes
    if len(spike_times) > 0:
        axes[0].vlines(spike_times, -80, -40, color='red', alpha=0.7, linewidth=2, label='Spikes')

    # Stimulus
    stimulus_values = [stimulus.get_current(t) for t in time_points]
    axes[1].plot(time_points, stimulus_values, 'k-', linewidth=2)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Input Current (pA)')
    axes[1].set_title('Step Current Stimulus')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lif_example.png', dpi=150, bbox_inches='tight')
    print("Plot saved as lif_example.png")

    return spike_times, spike_analysis


if __name__ == "__main__":
    run_basic_lif_experiment()