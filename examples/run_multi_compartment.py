"""
Example: Multi-compartment neuron simulation (Phase 10).

Demonstrates:
- Creating morphologies (simple, ball-and-stick, branching)
- Adding ion channels to specific compartments
- Simulating with external current
- Observing dendritic integration
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt

from neurosim.neurons.multi_compartment import MultiCompartmentNeuron
from neurosim.neurons.morphology import (
    create_simple_morphology,
    create_ball_and_stick,
    create_branching_dendrite,
)
from neurosim.channels.channel_models import create_hh_channels
from neurosim.channels.leak import LeakChannel


def simulate_neuron(neuron: MultiCompartmentNeuron, duration: float = 200.0, dt: float = 0.1):
    """Run simulation and record voltages."""
    n_steps = int(duration / dt)
    times = np.arange(n_steps) * dt

    # Record voltages for all compartments
    voltages = {comp.compartment_id: [] for comp in neuron.compartments}
    spike_times = []

    for i in range(n_steps):
        t = times[i]

        # Apply step current to soma from 50-150 ms
        soma = neuron.morphology.get_soma()
        if 50 <= t < 150:
            soma.external_current = 20.0  # µA
        else:
            soma.external_current = 0.0

        # Step the neuron
        spiked = neuron.step(t)
        if spiked:
            spike_times.append(t)

        # Record voltages
        for comp in neuron.compartments:
            voltages[comp.compartment_id].append(neuron.get_compartment_voltage(comp.compartment_id))

    return times, voltages, spike_times


def plot_results(times, voltages, spike_times, morphology_name: str, neuron):
    """Plot simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot voltage traces
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(voltages)))
    for i, (comp_id, v_list) in enumerate(voltages.items()):
        comp_name = next(c.name for c in [neuron.morphology.get_soma()] +
                        [child for child in neuron.morphology.get_all_compartments()
                         if child.compartment_id == comp_id])
        ax1.plot(times, v_list, color=colors[i], label=f'Comp {comp_id}')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title(f'{morphology_name}: Voltage Traces')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot all voltages at final time
    ax2 = axes[0, 1]
    comp_names = []
    final_voltages = []
    for comp_id, v_list in voltages.items():
        comp = next(c for c in neuron.morphology.compartments.values() if c.compartment_id == comp_id)
        comp_names.append(comp.name)
        final_voltages.append(v_list[-1])

    bars = ax2.bar(range(len(comp_names)), final_voltages, color=colors)
    ax2.set_xlabel('Compartment')
    ax2.set_ylabel('Final Voltage (mV)')
    ax2.set_title(f'{morphology_name}: Final Voltages')
    ax2.set_xticks(range(len(comp_names)))
    ax2.set_xticklabels(comp_names, rotation=45)
    ax2.axhline(y=-65, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Plot spike raster
    ax3 = axes[1, 0]
    if spike_times:
        ax3.scatter(spike_times, [0] * len(spike_times), s=50, color='red', alpha=0.7)
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Spike')
        ax3.set_title(f'{morphology_name}: Spike Times (n={len(spike_times)})')
        ax3.set_ylim(-0.5, 0.5)
    else:
        ax3.text(0.5, 0.5, 'No spikes detected', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'{morphology_name}: No Spikes')

    ax3.grid(True, alpha=0.3)

    # Plot morphology diagram
    ax4 = axes[1, 1]
    _plot_morphology(ax4, neuron.morphology)
    ax4.set_title(f'{morphology_name}: Morphology')

    plt.tight_layout()
    plt.savefig(f'multi_compartment_{morphology_name.lower().replace(" ", "_")}.png', dpi=150)
    print(f"Saved plot: multi_compartment_{morphology_name.lower().replace(' ', '_')}.png")
    plt.close()


def _plot_morphology(ax, morphology):
    """Plot a simple 2D representation of the morphology."""
    def plot_compartment(comp, x, y, scale=1.0):
        size = comp.diameter * scale / 20.0
        circle = plt.Circle((x, y), size, color='lightblue', ec='black', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, comp.name, ha='center', va='center', fontsize=8)

    def dfs(comp, x, y, angle, length_scale):
        plot_compartment(comp, x, y)
        for child in comp.children:
            child_angle = angle + np.random.uniform(-0.3, 0.3)
            child_x = x + child.length * length_scale * np.cos(child_angle)
            child_y = y + child.length * length_scale * np.sin(child_angle)
            dfs(child, child_x, child_y, child_angle, length_scale)

    soma = morphology.get_soma()
    dfs(soma, 0, 0, -np.pi/2, 0.01)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    print("=" * 60)
    print("Multi-Compartment Neuron Simulation (Phase 10)")
    print("=" * 60)

    # --- Experiment 1: Simple morphology with HH channels ---
    print("\n[1] Creating simple morphology (soma + 2 dendrites)...")
    morph1 = create_simple_morphology(
        num_dendrites=2,
        dendrite_length=100.0,
        dendrite_diameter=2.0,
        soma_length=20.0,
        soma_diameter=20.0,
    )
    print(f"    Compartments: {len(morph1.compartments)}")

    neuron1 = MultiCompartmentNeuron(neuron_id=0, morphology=morph1)

    # Add HH channels to soma
    soma1 = morph1.get_soma()
    hh_channels = create_hh_channels()
    for name, ch in hh_channels.items():
        neuron1.add_channel_to_compartment(soma1.compartment_id, ch)
        print(f"    Added {name} channel to soma")

    # Add leak to dendrites
    for comp in morph1.compartments.values():
        if comp.name.startswith("dendrite"):
            leak = LeakChannel(conductance=0.1, reversal_potential=-70.0)
            neuron1.add_channel_to_compartment(comp.compartment_id, leak)
            print(f"    Added leak channel to {comp.name}")

    # Simulate
    print("    Running simulation...")
    times, voltages, spikes = simulate_neuron(neuron1, duration=300.0, dt=0.1)
    print(f"    Spikes detected: {len(spikes)}")
    if spikes:
        print(f"    First spike at: {spikes[0]:.2f} ms")
        print(f"    Firing rate: {len(spikes) / (spikes[-1] - spikes[0]) * 1000:.1f} Hz")

    # Plot
    plot_results(times, voltages, spikes, "Simple", neuron1)

    # --- Experiment 2: Ball-and-stick ---
    print("\n[2] Creating ball-and-stick morphology...")
    morph2 = create_ball_and_stick(soma_diameter=20, dendrite_length=200)
    neuron2 = MultiCompartmentNeuron(neuron_id=1, morphology=morph2)

    soma2 = morph2.get_soma()
    for name, ch in hh_channels.items():
        neuron2.add_channel_to_compartment(soma2.compartment_id, ch)

    dendrite = soma2.children[0]
    leak = LeakChannel(conductance=0.1, reversal_potential=-70.0)
    neuron2.add_channel_to_compartment(dendrite.compartment_id, leak)

    times2, voltages2, spikes2 = simulate_neuron(neuron2, duration=300.0, dt=0.1)
    print(f"    Spikes detected: {len(spikes2)}")
    plot_results(times2, voltages2, spikes2, "Ball-and-Stick", neuron2)

    # --- Experiment 3: Branching dendrite ---
    print("\n[3] Creating branching dendrite morphology...")
    morph3 = create_branching_dendrite(branch_order=2)
    neuron3 = MultiCompartmentNeuron(neuron_id=2, morphology=morph3)

    soma3 = morph3.get_soma()
    for name, ch in hh_channels.items():
        neuron3.add_channel_to_compartment(soma3.compartment_id, ch)

    # Add leak to all dendrites
    for comp in morph3.compartments.values():
        if not comp.is_soma:
            leak = LeakChannel(conductance=0.05, reversal_potential=-70.0)
            neuron3.add_channel_to_compartment(comp.compartment_id, leak)

    times3, voltages3, spikes3 = simulate_neuron(neuron3, duration=300.0, dt=0.1)
    print(f"    Spikes detected: {len(spikes3)}")
    plot_results(times3, voltages3, spikes3, "Branching", neuron3)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Simple morphology:      {len(neuron1.compartments)} compartments, {len(spikes)} spikes")
    print(f"Ball-and-stick:         {len(neuron2.compartments)} compartments, {len(spikes2)} spikes")
    print(f"Branching dendrite:     {len(neuron3.compartments)} compartments, {len(spikes3)} spikes")
    print("\nPlots saved as PNG files.")


if __name__ == "__main__":
    main()
