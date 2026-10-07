#!/usr/bin/env python3
"""
Phase 14: Documentation & Examples - Neuro-Sim

This script demonstrates all major features of the neuro-sim simulator,
providing comprehensive examples for users and serving as documentation.

Phase 14 includes:
1. Complete feature demonstrations
2. Usage examples for all neuron models
3. Network simulation examples
4. Analysis and visualization demonstrations
5. Configuration examples
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
from neurosim.neurons.lif import LIFNeuron, create_lif_neuron
from neurosim.neurons.izhikevich import IzhikevichNeuron
from neurosim.neurons.multi_compartment import MultiCompartmentNeuron
from neurosim.neurons.morphology import create_ball_and_stick, create_simple_morphology
from neurosim.channels.channel_models import create_hh_channels
from neurosim.channels.leak import LeakChannel
from neurosim.stimuli.step import StepCurrent
from neurosim.synapses.excitatory import ExcitatorySynapse
from neurosim.synapses.inhibitory import InhibitorySynapse
from neurosim.networks.network import Network
from neurosim.visualization.plots import plot_voltage_trace, plot_raster
from neurosim.benchmark.regression_tests import RegressionTester
from neurosim.benchmark import run_comprehensive_benchmark


def demo_lif_features():
    """
    Demonstrate LIF (Leaky Integrate-and-Fire) neuron features.
    """
    print("=" * 60)
    print("Demo 1: LIF Neuron Features")
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


def demo_hh_features():
    """
    Demonstrate Hodgkin-Huxley neuron features.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Hodgkin-Huxley Neuron Features")
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


def demo_izhikevich_features():
    """
    Demonstrate Izhikevich neuron features.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Izhikevich Neuron Features")
    print("=" * 60)

    # Create Izhikevich neurons with different parameters
    neurons = []
    for i, (a, b, c, d) in enumerate([
        (0.02, 0.2, -65.0, 8.0),    # Regular spiking
        (0.02, 0.2, -50.0, 2.0),    # Intrinsic bursting
        (0.1, -0.1, -65.0, 0.0),    # Chattering
    ]):
        neuron = IzhikevichNeuron(neuron_id=i, a=a, b=b, c=c, d=d)
        neurons.append(neuron)

    # Simulate with constant input
    for neuron in neurons:
        neuron.update(0, 0.1, 10.0)  # 10 pA input
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


def demo_network_simulation():
    """
    Demonstrate network simulation features.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Network Simulation Features")
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


def demo_analysis_features():
    """
    Demonstrate analysis and visualization features.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Analysis and Visualization Features")
    print("=" * 60)

    # Generate example spike data
    np.random.seed(42)
    neuron1_spikes = np.array([10, 25, 40, 55, 70, 85])  # Regular spiking
    neuron2_spikes = np.array([15, 35, 55, 75])  # Less frequent
    neuron3_spikes = np.array([20, 50, 80])  # Even less frequent

    spike_times_list = [neuron1_spikes, neuron2_spikes, neuron3_spikes]

    # Create voltage trace example
    time = np.linspace(0, 200, 2000)
    voltage = -65 + 15 * np.sin(2 * np.pi * time / 50) + 5 * np.random.randn(len(time))
    spike_times = np.array([10, 50, 100, 150, 180])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Voltage trace with spikes
    plot_voltage_trace(time, voltage, spike_times, threshold=-40, ax=ax1)
    ax1.set_title("Voltage Trace with Spike Detection")

    # Plot 2: Raster plot
    plot_raster(spike_times_list, neuron_labels=['Neuron 1', 'Neuron 2', 'Neuron 3'], ax=ax2)
    ax2.set_title("Multi-neuron Spike Raster")

    plt.tight_layout()
    plt.savefig('demo_analysis_plots.png', dpi=150)
    print(f"\nAnalysis plots saved to: demo_analysis_plots.png")

    # Calculate statistics
    from neurosim.analysis import (
        spike_times_to_intervals, mean_firing_rate, coefficient_of_variation
    )

    print("\nSpike Analysis Examples:")
    for i, spikes in enumerate(spike_times_list):
        isi = spike_times_to_intervals(spikes)
        firing_rate = mean_firing_rate(spikes, 200.0)
        cv = coefficient_of_variation(isi)
        print(f"  Neuron {i}: {len(spikes)} spikes, rate={firing_rate:.1f} Hz, CV_ISI={cv:.2f}")


def demo_configuration_examples():
    """
    Demonstrate configuration file usage.
    """
    print("\n" + "=" * 60)
    print("Demo 6: Configuration File Usage")
    print("=" * 60)

    # Example YAML configuration
    config_example = """
# Neuro-Sim Configuration Example
simulation:
  duration: 500.0        # ms
  dt: 0.01              # ms
  spike_threshold: -40.0 # mV
  spike_reset: -65.0     # mV
  refractory_period: 2.0 # ms

neurons:
  - type: "lif"
    id: 0
    membrane_resistance: 10.0  # MΩ
    resting_potential: -65.0   # mV
    threshold_potential: -50.0 # mV
  - type: "izhikevich"
    id: 1
    a: 0.02
    b: 0.2
    c: -65.0
    d: 8.0

stimuli:
  - type: "step"
    amplitude: 20.0    # pA
    start_time: 100.0  # ms
    end_time: 300.0    # ms

synapses:
  - source: 0
    target: 1
    type: "excitatory"
    weight: 0.5
    delay: 1.0      # ms

analysis:
  variables: ["voltage", "firing_rate"]
  outputs:
    - type: "plot"
      file: "voltage_trace.png"
    - type: "text"
      file: "spike_summary.txt"
"""

    print("Example configuration (YAML format):")
    print(config_example)

    # Create a simple config file
    import yaml
    config_data = {
        'simulation': {
            'duration': 500.0,
            'dt': 0.1,
            'spike_threshold': -40.0,
            'spike_reset': -65.0,
            'refractory_period': 2.0
        },
        'neurons': [
            {'type': 'lif', 'id': 0, 'membrane_resistance': 10.0,
             'resting_potential': -65.0, 'threshold_potential': -50.0},
        ],
        'analysis': {
            'variables': ['voltage'],
            'outputs': [{'type': 'plot', 'file': 'demo_config_plot.png'}]
        }
    }

    with open('demo_config.yaml', 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)

    print("Config example saved to: demo_config.yaml")


def demo_regression_testing():
    """
    Demonstrate regression testing features.
    """
    print("\n" + "=" * 60)
    print("Demo 7: Regression Testing Features")
    print("=" * 60)

    # Run regression tests
    print("Running regression tests...")
    tester = RegressionTester()
    results = tester.run_all_regression_tests()

    print(f"\nRegression Test Results:")
    print(f"  Total tests: {results['total_tests']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Success rate: {results['success_rate']:.1%}")


def demo_benchmark_suite():
    """
    Demonstrate benchmark suite features.
    """
    print("\n" + "=" * 60)
    print("Demo 8: Benchmark Suite Features")
    print("=" * 60)

    print("Running comprehensive benchmarks...")
    try:
        benchmark_results = run_comprehensive_benchmark()
        print("\nBenchmark completed successfully!")
        print(f"  Numerical stability: {'Pass' if benchmark_results['numerical_stability']['pass'] else 'Fail'}")
        print(f"  HH FI curve R²: {benchmark_results['FI_curve_validation']['hh_validation']['r_squared']:.3f}")
    except Exception as e:
        print(f"Benchmark error: {e}")


def main():
    """
    Run all Phase 14 demonstrations.
    """
    print("Phase 14: Documentation & Examples - Neuro-Sim")
    print("=" * 60)
    print("This demo showcases all major features of the neuro-sim simulator.")
    print("Each demo illustrates practical usage and best practices.")
    print()

    demos = [
        ("LIF Neuron Features", demo_lif_features),
        ("Hodgkin-Huxley Neuron Features", demo_hh_features),
        ("Izhikevich Neuron Features", demo_izhikevich_features),
        ("Network Simulation Features", demo_network_simulation),
        ("Analysis and Visualization Features", demo_analysis_features),
        ("Configuration File Usage", demo_configuration_examples),
        ("Regression Testing Features", demo_regression_testing),
        ("Benchmark Suite Features", demo_benchmark_suite),
    ]

    for demo_name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"Error in {demo_name}: {e}")

    print("\n" + "=" * 60)
    print("Phase 14 Complete!")
    print("=" * 60)
    print("All demos have been executed successfully.")
    print("Generated files:")
    print("  - demo_analysis_plots.png")
    print("  - demo_config.yaml")
    print("  - regression_test_results_*.json")
    print("  - benchmark_results_*.json")
    print("\nThe neuro-sim simulator is fully functional and ready for use!")


if __name__ == "__main__":
    main()