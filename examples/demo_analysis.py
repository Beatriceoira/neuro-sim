#!/usr/bin/env python3
"""
Analysis and visualization demonstration.

Generates example spike data and voltage traces, renders a voltage trace
with spike markers and a multi-neuron raster, and prints spike-train
statistics.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
from neurosim.visualization.plots import plot_voltage_trace, plot_raster
from neurosim.analysis import (
    spike_times_to_intervals,
    mean_firing_rate,
    coefficient_of_variation,
)


def demo_analysis_features():
    """Demonstrate analysis and visualization features."""
    print("=" * 60)
    print("Demo: Analysis and Visualization Features")
    print("=" * 60)

    # Generate example spike data
    np.random.seed(42)
    neuron1_spikes = np.array([10, 25, 40, 55, 70, 85])  # Regular spiking
    neuron2_spikes = np.array([15, 35, 55, 75])          # Less frequent
    neuron3_spikes = np.array([20, 50, 80])               # Even less frequent

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
    print("\nSpike Analysis Examples:")
    for i, spikes in enumerate(spike_times_list):
        isi = spike_times_to_intervals(spikes)
        firing_rate = mean_firing_rate(spikes, 200.0)
        cv = coefficient_of_variation(isi)
        print(f"  Neuron {i}: {len(spikes)} spikes, rate={firing_rate:.1f} Hz, CV_ISI={cv:.2f}")


if __name__ == "__main__":
    demo_analysis_features()