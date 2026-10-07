"""
Spike train analysis tools for the biological neuron simulator.

This module provides functions for analyzing spike trains, including
inter-spike interval analysis, firing rate calculation, and spike statistics.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq


def spike_times_to_intervals(spike_times: np.ndarray) -> np.ndarray:
    """
    Convert spike times to inter-spike intervals (ISIs).

    Args:
        spike_times: Array of spike times (ms)

    Returns:
        Array of inter-spike intervals (ms)
        Returns empty array if fewer than 2 spikes
    """
    if len(spike_times) < 2:
        return np.array([])
    return np.diff(spike_times)


def instantaneous_firing_rate(spike_times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculate instantaneous firing rate using kernel density estimation.

    Args:
        spike_times: Array of spike times (ms)
        t: Time points at which to calculate rate (ms)

    Returns:
        Instantaneous firing rate (Hz) at each time point
    """
    if len(spike_times) == 0:
        return np.zeros_like(t)

    # Use a Gaussian kernel with width of 50 ms
    sigma = 50.0  # ms
    rates = np.zeros_like(t)

    for spike in spike_times:
        # Gaussian kernel centered at each spike
        rates += np.exp(-0.5 * ((t - spike) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    # Convert to Hz (spikes per second)
    rates *= 1000.0

    return rates


def mean_firing_rate(spike_times: np.ndarray, duration: float) -> float:
    """
    Calculate mean firing rate over a duration.

    Args:
        spike_times: Array of spike times (ms)
        duration: Duration of recording (ms)

    Returns:
        Mean firing rate (Hz)
    """
    if duration <= 0:
        return 0.0
    return (len(spike_times) / duration) * 1000.0  # Convert to Hz


def coefficient_of_variation(isi: np.ndarray) -> float:
    """
    Calculate coefficient of variation of inter-spike intervals.

    Args:
        isi: Array of inter-spike intervals (ms)

    Returns:
        Coefficient of variation (std/mean)
        Returns 0.0 if fewer than 2 intervals or mean is zero
    """
    if len(isi) < 2:
        return 0.0
    mean_isi = np.mean(isi)
    if mean_isi == 0:
        return 0.0
    return np.std(isi) / mean_isi


def spike_count_histogram(
    spike_times_list: List[np.ndarray],
    bin_edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a histogram of spike counts across multiple trials/neurons.

    Args:
        spike_times_list: List of arrays containing spike times for each trial/neuron
        bin_edges: Edges of time bins (ms)

    Returns:
        Tuple of (counts, bin_centers)
        counts: Array of spike counts per bin
        bin_centers: Center of each bin (ms)
    """
    all_spikes = np.concatenate(spike_times_list) if spike_times_list else np.array([])
    counts, _ = np.histogram(all_spikes, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts, bin_centers


def raster_plot_data(
    spike_times_list: List[np.ndarray]
) -> Tuple[List[float], List[int]]:
    """
    Prepare data for raster plot.

    Args:
        spike_times_list: List of arrays containing spike times for each neuron/trial

    Returns:
        Tuple of (spike_times_flat, neuron_indices)
        spike_times_flat: Flattened list of all spike times
        neuron_indices: List indicating which neuron/trial each spike belongs to
    """
    spike_times_flat = []
    neuron_indices = []

    for i, spike_times in enumerate(spike_times_list):
        spike_times_flat.extend(spike_times)
        neuron_indices.extend([i] * len(spike_times))

    return spike_times_flat, neuron_indices


def detect_spikes_from_voltage(
    voltage: np.ndarray,
    time: np.ndarray,
    threshold: float = -40.0,
    min_interval: float = 2.0
) -> np.ndarray:
    """
    Detect spikes from voltage trace using threshold crossing.

    Args:
        voltage: Membrane potential trace (mV)
        time: Time points corresponding to voltage (ms)
        threshold: Spike detection threshold (mV)
        min_interval: Minimum interval between spikes (ms) for refractory period

    Returns:
        Array of detected spike times (ms)
    """
    # Find threshold crossings (positive going)
    above_threshold = voltage >= threshold
    # Find rising edges
    rising_edges = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

    if len(rising_edges) == 0:
        return np.array([])

    spike_times = time[rising_edges]

    # Apply refractory period constraint
    if len(spike_times) > 1 and min_interval > 0:
        filtered_spikes = [spike_times[0]]
        for spike in spike_times[1:]:
            if spike - filtered_spikes[-1] >= min_interval:
                filtered_spikes.append(spike)
        spike_times = np.array(filtered_spikes)

    return spike_times


def analyze_spike_train(
    spike_times: np.ndarray,
    duration: Optional[float] = None
) -> dict:
    """
    Perform comprehensive analysis of a spike train.

    Args:
        spike_times: Array of spike times (ms)
        duration: Recording duration (ms). If None, estimated from spike times

    Returns:
        Dictionary containing spike train analysis results
    """
    if len(spike_times) == 0:
        if duration is None:
            duration = 0.0
        return {
            'n_spikes': 0,
            'mean_firing_rate_Hz': 0.0,
            'cv_isi': 0.0,
            'mean_isi_ms': 0.0,
            'std_isi_ms': 0.0,
            'duration_ms': duration,
            'isi_values': np.array([]),
            'spike_times': spike_times
        }

    if duration is None:
        duration = spike_times[-1] - spike_times[0] if len(spike_times) > 1 else 1000.0

    isi = spike_times_to_intervals(spike_times)

    analysis = {
        'n_spikes': len(spike_times),
        'mean_firing_rate_Hz': mean_firing_rate(spike_times, duration),
        'cv_isi': coefficient_of_variation(isi) if len(isi) > 0 else 0.0,
        'mean_isi_ms': np.mean(isi) if len(isi) > 0 else 0.0,
        'std_isi_ms': np.std(isi) if len(isi) > 0 else 0.0,
        'duration_ms': duration,
        'isi_values': isi,
        'spike_times': spike_times,
        'fano_factor': np.var(isi) / np.mean(isi) if len(isi) > 0 and np.mean(isi) > 0 else 0.0
    }

    return analysis


def population_synchrony(
    spike_times_list: List[np.ndarray],
    bin_width: float = 10.0
) -> float:
    """
    Calculate population synchrony as the variance of binned spike counts.

    Args:
        spike_times_list: List of arrays containing spike times for each neuron
        bin_width: Width of time bins for synchrony calculation (ms)

    Returns:
        Synchrony measure (variance of binned spike counts normalized by mean)
    """
    if len(spike_times_list) == 0:
        return 0.0

    # Find time range
    all_spikes = np.concatenate(spike_times_list) if spike_times_list else np.array([0])
    if len(all_spikes) == 0:
        return 0.0

    t_min, t_max = np.min(all_spikes), np.max(all_spikes)
    if t_max <= t_min:
        t_max = t_min + 1000.0  # Default 1 second if no spikes or single spike

    # Create bins
    bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Count spikes in each bin for each neuron
    binned_counts = []
    for spike_times in spike_times_list:
        counts, _ = np.histogram(spike_times, bins=bin_edges)
        binned_counts.append(counts)

    # Sum across neurons to get population activity per bin
    if len(binned_counts) > 0:
        population_activity = np.sum(binned_counts, axis=0)
        mean_activity = np.mean(population_activity)
        if mean_activity > 0:
            synchrony = np.var(population_activity) / mean_activity
        else:
            synchrony = 0.0
    else:
        synchrony = 0.0

    return synchrony


def cross_correlation(
    spike_times_1: np.ndarray,
    spike_times_2: np.ndarray,
    max_lag: float = 100.0,
    bin_width: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cross-correlation between two spike trains.

    Args:
        spike_times_1: Spike times of first neuron (ms)
        spike_times_2: Spike times of second neuron (ms)
        max_lag: Maximum lag to consider (ms)
        bin_width: Width of time bins (ms)

    Returns:
        Tuple of (lags, cross_corr)
        lags: Array of lag values (ms)
        cross_corr: Cross-correlation values at each lag
    """
    if len(spike_times_1) == 0 or len(spike_times_2) == 0:
        lags = np.arange(-max_lag, max_lag + bin_width, bin_width)
        return lags, np.zeros_like(lags)

    # Create binned spike trains
    t_min = min(np.min(spike_times_1), np.min(spike_times_2))
    t_max = max(np.max(spike_times_1), np.max(spike_times_2))
    t_max = max(t_max, t_min + max_lag)  # Ensure we cover the lag range

    bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Bin the spike trains
    train_1_binned, _ = np.histogram(spike_times_1, bins=bin_edges)
    train_2_binned, _ = np.histogram(spike_times_2, bins=bin_edges)

    # Calculate cross-correlation
    cross_corr = np.correlate(train_1_binned, train_2_binned, mode='full')

    # Extract lags corresponding to desired range
    mid_point = len(cross_corr) // 2
    max_index = int(max_lag / bin_width)
    start_index = mid_point - max_index
    end_index = mid_point + max_index + 1

    # Ensure indices are within bounds
    start_index = max(0, start_index)
    end_index = min(len(cross_corr), end_index)

    cross_corr_segment = cross_corr[start_index:end_index]
    lags = np.arange(-max_lag, max_lag + bin_width, bin_width)[:len(cross_corr_segment)]

    # Normalize by sqrt of product of autocorrelations at zero lag
    auto_1 = np.correlate(train_1_binned, train_1_binned, mode='full')[mid_point]
    auto_2 = np.correlate(train_2_binned, train_2_binned, mode='full')[mid_point]
    norm_factor = np.sqrt(auto_1 * auto_2) if auto_1 > 0 and auto_2 > 0 else 1.0

    if norm_factor > 0:
        cross_corr_segment = cross_corr_segment / norm_factor

    return lags, cross_corr_segment


# Example usage and testing functions
def example_spike_analysis():
    """Example function demonstrating spike analysis."""
    # Generate some example spike trains
    np.random.seed(42)

    # Regular spiking neuron
    regular_spikes = np.arange(100, 1000, 100) + np.random.normal(0, 5, 9)

    # Bursting neuron
    burst1 = np.arange(200, 250, 10)
    burst2 = np.arange(500, 550, 10)
    burst_spikes = np.concatenate([burst1, burst2])

    # Analyze regular spiking
    regular_analysis = analyze_spike_train(regular_spikes, duration=1000.0)
    print("Regular spiking analysis:")
    for key, value in regular_analysis.items():
        if isinstance(value, np.ndarray) and len(value) < 5:
            print(f"  {key}: {value}")
        elif not isinstance(value, np.ndarray):
            print(f"  {key}: {value}")

    print("\nBursting analysis:")
    burst_analysis = analyze_spike_train(burst_spikes, duration=1000.0)
    for key, value in burst_analysis.items():
        if isinstance(value, np.ndarray) and len(value) < 5:
            print(f"  {key}: {value}")
        elif not isinstance(value, np.ndarray):
            print(f"  {key}: {value}")

    # Calculate synchrony
    synchrony = population_synchrony([regular_spikes, burst_spikes], bin_width=50.0)
    print(f"\nPopulation synchrony: {synchrony:.3f}")

    # Cross-correlation
    lags, xcorr = cross_correlation(regular_spikes, burst_spikes, max_lag=200.0, bin_width=5.0)
    print(f"Cross-correlation peak at lag: {lags[np.argmax(xcorr)]:.1f} ms")


if __name__ == "__main__":
    example_spike_analysis()