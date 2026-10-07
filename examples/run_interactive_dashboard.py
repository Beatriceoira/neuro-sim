"""
Example: Interactive dashboard for multi-compartment neuron simulation (Phase 11).

Demonstrates:
- Creating membrane potential trace plots
- Spike raster plots
- Phase plane visualizations
- F-I curves
- Web dashboard with Plotly
- Matplotlib dashboard for real-time monitoring
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from neurosim.neurons.multi_compartment import MultiCompartmentNeuron
from neurosim.neurons.morphology import create_simple_morphology, create_ball_and_stick
from neurosim.channels.channel_models import create_hh_channels
from neurosim.channels.leak import LeakChannel
from neurosim.visualization.plots import (
    plot_voltage_trace,
    plot_raster,
    plot_phase_plane,
    plot_fi_curve,
    plot_isi_distribution,
    plot_network_activity,
    plot_synchrony,
    plot_current_stimulus,
)
from neurosim.visualization.web_dashboard import create_web_dashboard
from neurosim.visualization.dashboard import create_dashboard


def simulate_hh_neuron(duration=200.0, dt=0.1, current=5.0):
    """Simulate a single HH neuron and return results."""
    morph = create_ball_and_stick(soma_diameter=20, dendrite_length=200)
    neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)

    soma = morph.get_soma()
    hh_channels = create_hh_channels()
    for ch in hh_channels.values():
        neuron.add_channel_to_compartment(soma.compartment_id, ch)

    neuron.add_channel_to_compartment(soma.compartment_id, LeakChannel())

    # Inject external current into the soma
    soma.external_current = current

    times = []
    voltages = []
    spike_times = []

    for t_ms in range(int(duration / dt)):
        t = float(t_ms) * dt
        spiked = neuron.step(t)

        times.append(t)
        voltages.append(neuron.get_compartment_voltage(soma.compartment_id))

        # Detect spikes from the step() return value
        if spiked:
            spike_times.append(t)

    return np.array(times), np.array(voltages), np.array(spike_times)


def demo_basic_plots():
    """Demonstrate basic plotting functions."""
    print("=" * 60)
    print("Phase 11: Interactive Visualization - Basic Plots")
    print("=" * 60)

    # Simulate neuron
    times, voltages, spike_times = simulate_hh_neuron(duration=200.0, current=5.0)

    print(f"Simulated {len(times)} time points")
    print(f"Detected {len(spike_times)} spikes")
    if len(spike_times) > 0:
        print(f"First spike at {spike_times[0]:.2f} ms")

    # Plot 1: Voltage trace
    print("\n[1] Creating voltage trace plot...")
    fig, ax = plot_voltage_trace(
        time=times,
        voltage=voltages,
        spike_times=spike_times,
        threshold=-40,
        title="Hodgkin-Huxley Neuron - Membrane Potential",
        save_path="viz_voltage_trace.png",
        show=False,
    )
    print("    Saved: viz_voltage_trace.png")

    # Plot 2: Current stimulus
    print("\n[2] Creating current stimulus plot...")
    current = np.where((times > 50) & (times < 150), 5.0, 0.0)
    fig, ax = plot_current_stimulus(
        time=times,
        current=current,
        title="Injected Current (5 µA from 50-150 ms)",
        save_path="viz_current.png",
        show=False,
    )
    print("    Saved: viz_current.png")

    # Plot 3: ISI distribution
    print("\n[3] Creating ISI distribution plot...")
    if len(spike_times) >= 2:
        fig, ax = plot_isi_distribution(spike_times, save_path="viz_isi.png", show=False)
        print("    Saved: viz_isi.png")

    # Plot 4: F-I curve (simulated)
    print("\n[4] Creating F-I curve...")
    currents = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    firing_rates = np.array([0, 0, 5, 15, 30, 50, 70, 90, 110, 130, 150])
    fig, ax = plot_fi_curve(
        currents=currents,
        firing_rates=firing_rates,
        title="F-I Curve (simulated)",
        save_path="viz_fi_curve.png",
        show=False,
    )
    print("    Saved: viz_fi_curve.png")

    print("\nBasic plots complete!")


def demo_raster_plot():
    """Demonstrate spike raster plot."""
    print("\n" + "=" * 60)
    print("Phase 11: Spike Raster Plot")
    print("=" * 60)

    # Simulate multiple neurons with different currents
    spike_times_list = []
    neuron_labels = []

    for i, current in enumerate([3.0, 5.0, 7.0, 9.0]):
        print(f"  Simulating neuron {i} with {current} µA...")
        times, voltages, spikes = simulate_hh_neuron(duration=150.0, current=current)
        spike_times_list.append(spikes)
        neuron_labels.append(f"I={current}µA")

    print(f"\n[1] Creating spike raster for {len(spike_times_list)} neurons...")
    fig, ax = plot_raster(
        spike_times_list=spike_times_list,
        neuron_labels=neuron_labels,
        title="Spike Raster: 4 Neurons with Different Currents",
        save_path="viz_raster.png",
        show=False,
    )
    print("    Saved: viz_raster.png")

    print("Raster plot complete!")


def demo_phase_plane():
    """Demonstrate phase plane visualization."""
    print("\n" + "=" * 60)
    print("Phase 11: Phase Plane Visualization")
    print("=" * 60)

    # Simulate neuron
    times, voltages, spike_times = simulate_hh_neuron(duration=100.0, current=5.0)

    # Use voltage as both axes for a 1D phase portrait
    # (In a real Izhikevich simulation, you'd use v and u)
    print("[1] Creating phase plane plot...")
    fig, ax = plot_phase_plane(
        v_trace=voltages,
        u_trace=np.sin(2 * np.pi * times / 50),  # Simulated recovery variable
        title="Phase Plane: V vs Recovery Variable",
        save_path="viz_phase_plane.png",
        show=False,
    )
    print("    Saved: viz_phase_plane.png")

    print("Phase plane complete!")


def demo_network_activity():
    """Demonstrate network activity overview."""
    print("\n" + "=" * 60)
    print("Phase 11: Network Activity Overview")
    print("=" * 60)

    # Simulate 3 neurons
    voltages = {}
    spike_times_list = []

    for i, current in enumerate([3.0, 5.0, 7.0]):
        print(f"  Simulating neuron {i} with {current} µA...")
        times, v, spikes = simulate_hh_neuron(duration=100.0, current=current)
        voltages[i] = v
        spike_times_list.append(spikes)

    print("[1] Creating network activity overview...")
    fig = plot_network_activity(
        time=times,
        voltages=voltages,
        spike_times_list=spike_times_list,
        title="Network Activity: 3 Neurons",
        save_path="viz_network.png",
        show=False,
    )
    print("    Saved: viz_network.png")

    print("Network activity plot complete!")


def demo_web_dashboard():
    """Demonstrate web dashboard."""
    print("\n" + "=" * 60)
    print("Phase 11: Web Dashboard (Plotly)")
    print("=" * 60)

    # Simulate 2 neurons
    class FakeNeuron:
        def __init__(self):
            self.membrane_potential = -65.0

    neurons = [FakeNeuron(), FakeNeuron()]
    dash = create_web_dashboard(neurons=neurons, duration=100.0, dt=0.1)

    print("[1] Recording simulation steps...")
    for t_ms in range(0, 100):
        t = float(t_ms) * 0.1

        # Simulate voltage changes
        neurons[0].membrane_potential = -65 + 10 * np.sin(2 * np.pi * t / 50)
        neurons[1].membrane_potential = -65 + 8 * np.sin(2 * np.pi * t / 50 + np.pi/4)

        dash.record_step(t)

        # Simulate spikes
        if t_ms == 25:
            dash.on_spike(0, t)
        if t_ms == 30:
            dash.on_spike(1, t)
        if t_ms == 75:
            dash.on_spike(0, t)

    print("[2] Generating web dashboard...")
    fig = dash.plot_network_summary()
    print("    Dashboard generated with 4 subplots")

    print("[3] Saving to HTML...")
    dash.save_html("viz_web_dashboard.html")
    print("    Saved: viz_web_dashboard.html")

    print("Web dashboard complete!")


def demo_matplotlib_dashboard():
    """Demonstrate matplotlib dashboard."""
    print("\n" + "=" * 60)
    print("Phase 11: Matplotlib Dashboard")
    print("=" * 60)

    class FakeNeuron:
        def __init__(self):
            self.membrane_potential = -65.0

    neurons = [FakeNeuron(), FakeNeuron()]
    dash = create_dashboard(neurons=neurons, duration=50.0, dt=0.1)

    print("[1] Recording simulation steps...")
    for t_ms in range(0, 50):
        t = float(t_ms) * 0.1

        neurons[0].membrane_potential = -65 + 5 * np.sin(2 * np.pi * t / 25)
        neurons[1].membrane_potential = -65 + 3 * np.cos(2 * np.pi * t / 20)

        dash.record_step(t)

        if t_ms in [10, 25, 40]:
            dash.on_spike(0, t)
        if t_ms in [15, 35]:
            dash.on_spike(1, t)

    print("[2] Updating dashboard...")
    dash.update()

    print("[3] Saving dashboard...")
    dash.save("viz_matplotlib_dashboard.png")
    print("    Saved: viz_matplotlib_dashboard.png")

    print("Matplotlib dashboard complete!")


def main():
    """Run all Phase 11 visualization demos."""
    print("Starting Phase 11: Interactive Visualization Examples")
    print("=" * 60)

    try:
        demo_basic_plots()
        demo_raster_plot()
        demo_phase_plane()
        demo_network_activity()
        demo_web_dashboard()
        demo_matplotlib_dashboard()

        print("\n" + "=" * 60)
        print("Phase 11 Complete!")
        print("=" * 60)
        print("Generated files:")
        print("  - viz_voltage_trace.png")
        print("  - viz_current.png")
        print("  - viz_isi.png")
        print("  - viz_fi_curve.png")
        print("  - viz_raster.png")
        print("  - viz_phase_plane.png")
        print("  - viz_network.png")
        print("  - viz_web_dashboard.html")
        print("  - viz_matplotlib_dashboard.png")

    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Some visualizations require matplotlib or plotly.")
        print("Install with: pip install matplotlib plotly")


if __name__ == "__main__":
    main()
