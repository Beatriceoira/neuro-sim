"""
Interactive dashboard for biological neuron simulations (Phase 11).

The Dashboard class provides real-time plotting of membrane potential,
spike rasters, phase planes, and stimulus waveforms with optional
matplotlib widget integration.
"""

from typing import Dict, List, Optional, Callable, Any
import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    """
    Interactive simulation dashboard with real-time visualization.

    Manages multiple subplots and updates them at each simulation
    step. Intended for use with the NeuroSim simulation engine.

    Parameters
    ----------
    neurons : list
        List of neuron objects to monitor.
    duration : float
        Simulation duration (ms).
    dt : float
        Time step (ms).
    n_cols : int
        Number of subplot columns (default 2).
    """

    def __init__(
        self,
        neurons: List[Any],
        duration: float,
        dt: float = 0.1,
        n_cols: int = 2,
    ):
        self.neurons = neurons
        self.duration = duration
        self.dt = dt
        self.n_cols = n_cols
        self.n_rows = int(np.ceil(len(neurons) / n_cols)) + 1  # +1 for raster

        self.fig, self.axes = self._setup_subplots()
        self.recorded = {i: {'time': [], 'voltage': []} for i in range(len(neurons))}
        self.all_spike_times = [[] for _ in neurons]

    def _setup_subplots(self):
        """Create subplot grid."""
        fig, axes = plt.subplots(self.n_rows, self.n_cols, figsize=(14, 4 * self.n_rows))
        if self.n_rows == 1:
            axes = axes.reshape(1, -1)
        return fig, axes

    def record_step(self, t: float):
        """Record current state of all neurons."""
        for i, neuron in enumerate(self.neurons):
            self.recorded[i]['time'].append(t)
            self.recorded[i]['voltage'].append(neuron.membrane_potential)

    def on_spike(self, neuron_idx: int, t: float):
        """Record a spike event."""
        self.all_spike_times[neuron_idx].append(t)

    def update(self):
        """Redraw all subplots with recorded data."""
        # Clear all axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()

        row = 0
        # Voltage traces
        for i in range(len(self.neurons)):
            ax = self.axes[row, i % self.n_cols]
            rec = self.recorded[i]
            if rec['time']:
                ax.plot(rec['time'], rec['voltage'], 'b-', linewidth=1.2, label='V(t)')
                ax.set_ylabel('mV', fontsize=9)
            ax.set_title(f'Neuron {i}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7)
        row += 1

        # Raster plot (last row, all columns)
        ax_raster = self.axes[row - 1, 0] if self.n_cols == 1 else self.axes[row - 1, 0]
        for i, spikes in enumerate(self.all_spike_times):
            if spikes:
                ax_raster.vlines(spikes, i + 0.5, i + 1.5, color='black', linewidth=1)
        ax_raster.set_xlabel('Time (ms)')
        ax_raster.set_ylabel('Neuron')
        ax_raster.set_title('Spike Raster')
        ax_raster.grid(True, alpha=0.3)

        plt.tight_layout()

    def show(self):
        """Display the dashboard."""
        plt.show()

    def save(self, path: str):
        """Save dashboard to file."""
        self.fig.savefig(path, dpi=150)


def create_dashboard(
    neurons: List[Any],
    duration: float,
    dt: float = 0.1,
    n_cols: int = 2,
) -> Dashboard:
    """Factory function to create a dashboard."""
    return Dashboard(neurons=neurons, duration=duration, dt=dt, n_cols=n_cols)
