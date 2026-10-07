"""
Plotting utilities for the biological neuron simulator.

Provides static and interactive plot generators using matplotlib and plotly
for membrane potential traces, spike rasters, phase planes, F-I curves,
and network activity.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


def plot_voltage_trace(
    time: np.ndarray,
    voltage: np.ndarray,
    spike_times: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    title: str = "Membrane Potential Trace",
    xlabel: str = "Time (ms)",
    ylabel: str = "Membrane Potential (mV)",
    figsize: Tuple[float, float] = (10, 4),
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot membrane potential over time with optional spike markers and threshold line.

    Args:
        time: Array of time points (ms)
        voltage: Array of membrane potential values (mV)
        spike_times: Optional array of spike times (ms)
        threshold: Optional spike threshold potential (mV)
        title: Chart title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figsize: Figure dimensions (width, height)
        ax: Optional existing matplotlib Axes object
        show: If True, calls plt.show()
        save_path: Optional file path to save image

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(time, voltage, 'b-', linewidth=1.5, label='V(t)')

    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({threshold} mV)')

    if spike_times is not None and len(spike_times) > 0:
        # Interpolate voltage at spike times for markers
        spike_v = np.interp(spike_times, time, voltage) if len(voltage) == len(time) else [threshold or -40]*len(spike_times)
        ax.scatter(spike_times, spike_v, color='red', s=40, zorder=5, label=f'Spikes (n={len(spike_times)})')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def plot_raster(
    spike_times_list: List[np.ndarray],
    neuron_labels: Optional[List[str]] = None,
    title: str = "Spike Raster Plot",
    xlabel: str = "Time (ms)",
    ylabel: str = "Neuron Index",
    time_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 5),
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spike raster for a population of neurons.

    Args:
        spike_times_list: List of 1D arrays containing spike times for each neuron
        neuron_labels: Optional labels for each neuron
        title: Chart title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        time_range: Optional (t_min, t_max) tuple to crop x-axis
        figsize: Figure dimensions
        ax: Optional existing matplotlib Axes object
        show: If True, calls plt.show()
        save_path: Optional file path to save image

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_neurons = len(spike_times_list)

    for i, spike_times in enumerate(spike_times_list):
        if len(spike_times) > 0:
            ax.vlines(spike_times, i + 0.6, i + 1.4, color='black', linewidth=1.2)

    ax.set_ylim(0.5, n_neurons + 0.5)
    ax.set_yticks(range(1, n_neurons + 1))

    if neuron_labels and len(neuron_labels) == n_neurons:
        ax.set_yticklabels(neuron_labels)

    if time_range:
        ax.set_xlim(time_range)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def plot_phase_plane(
    v_trace: np.ndarray,
    u_trace: np.ndarray,
    v_nullcline: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    u_nullcline: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Phase Plane Trajectory (V vs U)",
    xlabel: str = "Voltage V (mV)",
    ylabel: str = "Recovery U / Gate",
    figsize: Tuple[float, float] = (7, 6),
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot phase plane trajectory (e.g., V vs recovery variable or gating variable).

    Args:
        v_trace: Membrane potential values (mV)
        u_trace: Second state variable values (e.g. u in Izhikevich or m/h/n in HH)
        v_nullcline: Optional tuple (v_vals, u_vals) for dV/dt = 0 nullcline
        u_nullcline: Optional tuple (v_vals, u_vals) for du/dt = 0 nullcline
        title: Chart title
        xlabel: x-axis label
        ylabel: y-axis label
        figsize: Figure dimensions
        ax: Optional existing matplotlib Axes object
        show: If True, calls plt.show()
        save_path: Optional file path to save image

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot nullclines if provided
    if v_nullcline is not None:
        ax.plot(v_nullcline[0], v_nullcline[1], 'r--', linewidth=1.5, label='dV/dt = 0')

    if u_nullcline is not None:
        ax.plot(u_nullcline[0], u_nullcline[1], 'g--', linewidth=1.5, label='du/dt = 0')

    # Trajectory
    ax.plot(v_trace, u_trace, 'b-', linewidth=1.0, alpha=0.8, label='Trajectory')
    ax.scatter(v_trace[0], u_trace[0], color='green', s=50, marker='o', zorder=5, label='Start')
    ax.scatter(v_trace[-1], u_trace[-1], color='red', s=50, marker='s', zorder=5, label='End')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def plot_current_stimulus(
    time: np.ndarray,
    current: np.ndarray,
    title: str = "Input Current Stimulus",
    xlabel: str = "Time (ms)",
    ylabel: str = "Current (pA)",
    figsize: Tuple[float, float] = (10, 3),
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot current stimulus waveform over time.

    Args:
        time: Time array (ms)
        current: Current amplitude array (pA or µA)
        title: Chart title
        xlabel: x-axis label
        ylabel: y-axis label
        figsize: Figure dimensions
        ax: Optional existing matplotlib Axes object
        show: If True, calls plt.show()
        save_path: Optional file path to save image

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(time, current, 'k-', linewidth=1.5)
    ax.fill_between(time, 0, current, alpha=0.2, color='gray')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def plot_network_activity(
    time: np.ndarray,
    voltages: Dict[int, np.ndarray],
    spike_times_list: List[np.ndarray],
    title: str = "Network Activity Overview",
    figsize: Tuple[float, float] = (12, 8),
    show: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Combined overview figure with voltage traces and spike raster plot.

    Args:
        time: Time array (ms)
        voltages: Dict mapping neuron_id -> voltage trace array
        spike_times_list: List of spike time arrays per neuron
        title: Figure super-title
        figsize: Figure dimensions
        show: If True, calls plt.show()
        save_path: Optional file path to save image

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # 1. Voltage traces
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(voltages))))
    for i, (nid, v) in enumerate(voltages.items()):
        ax1.plot(time, v, color=colors[i % len(colors)], linewidth=1.2, label=f'Neuron {nid}')

    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title(f"{title}: Membrane Potentials", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # 2. Raster plot
    plot_raster(
        spike_times_list=spike_times_list,
        title="Spike Raster",
        xlabel="Time (ms)",
        ylabel="Neuron Index",
        ax=ax2,
        show=False
    )

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig


def plot_fi_curve(
    currents: np.ndarray,
    firing_rates: np.ndarray,
    title: str = "F-I Curve (Frequency vs Injected Current)",
    xlabel: str = "Injected Current (pA)",
    ylabel: str = "Firing Rate (Hz)",
    figsize: Tuple[float, float] = (8, 5),
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot firing frequency as a function of injected current amplitude (F-I curve).

    Args:
        currents: Array of injected current levels
        firing_rates: Array of resulting firing rates (Hz)
        title: Chart title
        xlabel: x-axis label
        ylabel: y-axis label
        figsize: Figure dimensions
        ax: Optional existing matplotlib Axes object
        show: If True, calls plt.show()
        save_path: Optional file path to save image

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(currents, firing_rates, 'bo-', linewidth=1.8, markersize=5)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def plot_isi_distribution(
    spike_times: np.ndarray,
    bins: int = 20,
    title: str = "Inter-Spike Interval (ISI) Distribution",
    xlabel: str = "ISI (ms)",
    ylabel: str = "Count",
    figsize: Tuple[float, float] = (8, 4),
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of inter-spike intervals (ISIs).

    Args:
        spike_times: Array of spike times (ms)
        bins: Number of histogram bins
        title: Chart title
        xlabel: x-axis label
        ylabel: y-axis label
        figsize: Figure dimensions
        ax: Optional existing matplotlib Axes object
        show: If True, calls plt.show()
        save_path: Optional file path to save image

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if len(spike_times) >= 2:
        isis = np.diff(spike_times)
        ax.hist(isis, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        mean_isi = np.mean(isis)
        cv = np.std(isis) / mean_isi if mean_isi > 0 else 0
        ax.axvline(mean_isi, color='red', linestyle='--', label=f'Mean = {mean_isi:.1f} ms (CV={cv:.2f})')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Insufficient spikes (<2)", ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax


def plot_synchrony(
    time: np.ndarray,
    synchrony_index: np.ndarray,
    title: str = "Population Synchrony Index over Time",
    xlabel: str = "Time (ms)",
    ylabel: str = "Synchrony R(t)",
    figsize: Tuple[float, float] = (10, 4),
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot population synchrony / order parameter over time.

    Args:
        time: Time array (ms)
        synchrony_index: Array of order parameter values (0 to 1)
        title: Chart title
        xlabel: x-axis label
        ylabel: y-axis label
        figsize: Figure dimensions
        ax: Optional existing matplotlib Axes object
        show: If True, calls plt.show()
        save_path: Optional file path to save image

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(time, synchrony_index, 'm-', linewidth=1.5)
    ax.set_ylim(-0.05, 1.05)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, ax
