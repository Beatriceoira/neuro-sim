"""
JIT-compiled hot loops for the biological neuron simulator (Phase 12).

Uses Numba to accelerate the performance-critical inner loops that dominate
simulation time: spike detection, refractory updates, and synaptic current
accumulation. Falls back to plain NumPy when Numba is unavailable.
"""

from typing import List, Tuple
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _detect_spikes_jit(
        membrane_potentials: np.ndarray,
        refractory_remaining: np.ndarray,
        spike_threshold: float,
        refractory_period: float,
        current_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect threshold-crossing spikes in one vectorized pass.

        Returns (spiked_mask, updated_refractory).
        spiked_mask is a boolean array of length N.
        """
        n = len(membrane_potentials)
        spiked = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if refractory_remaining[i] <= 0.0 and membrane_potentials[i] >= spike_threshold:
                spiked[i] = True
                refractory_remaining[i] = refractory_period
            else:
                refractory_remaining[i] = max(0.0, refractory_remaining[i] - refractory_period)
        return spiked, refractory_remaining

    @njit(cache=True, fastmath=True)
    def _accumulate_synaptic_currents_jit(
        target_indices: np.ndarray,
        conductances: np.ndarray,
        weights: np.ndarray,
        reversal_potentials: np.ndarray,
        is_inhibitory: np.ndarray,
        membrane_potentials: np.ndarray,
    ) -> np.ndarray:
        """
        Accumulate synaptic currents for each target neuron.

        Args:
            target_indices: int array of target neuron indices per connection
            conductances: float array of current synaptic conductance per connection
            weights: float array of synaptic weights per connection
            reversal_potentials: float array of E_syn per connection
            is_inhibitory: bool array, True for inhibitory synapses
            membrane_potentials: float array of target neuron V_m

        Returns:
            Array of accumulated synaptic current per neuron
        """
        n_neurons = len(membrane_potentials)
        synaptic_currents = np.zeros(n_neurons, dtype=np.float64)
        n_conns = len(target_indices)
        for c in range(n_conns):
            tgt = target_indices[c]
            g = conductances[c]
            w = weights[c]
            e_syn = reversal_potentials[c]
            v = membrane_potentials[tgt]
            if is_inhibitory[c]:
                # Inhibitory: I = g * w * (V - E_syn)  (negative when V > E_syn)
                synaptic_currents[tgt] += g * w * (v - e_syn)
            else:
                # Excitatory: I = g * w * (E_syn - V)  (positive when V < E_syn)
                synaptic_currents[tgt] += g * w * (e_syn - v)
        return synaptic_currents


def detect_spikes(
    membrane_potentials: np.ndarray,
    refractory_remaining: np.ndarray,
    spike_threshold: float,
    refractory_period: float,
    current_time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect spikes and update refractory periods.

    Vectorized NumPy fallback when Numba is unavailable.
    """
    if HAS_NUMBA:
        return _detect_spikes_jit(
            np.ascontiguousarray(membrane_potentials, dtype=np.float64),
            np.ascontiguousarray(refractory_remaining, dtype=np.float64),
            spike_threshold,
            refractory_period,
            current_time,
        )

    spiked = (
        (refractory_remaining <= 0.0)
        & (membrane_potentials >= spike_threshold)
    )
    refractory_remaining = np.where(
        spiked, refractory_period,
        np.maximum(0.0, refractory_remaining - refractory_period),
    )
    return spiked, refractory_remaining


def accumulate_synaptic_currents(
    target_indices: np.ndarray,
    conductances: np.ndarray,
    weights: np.ndarray,
    reversal_potentials: np.ndarray,
    is_inhibitory: np.ndarray,
    membrane_potentials: np.ndarray,
) -> np.ndarray:
    """
    Accumulate per-neuron synaptic currents from connection arrays.

    Vectorized NumPy fallback when Numba is unavailable.
    """
    if HAS_NUMBA:
        return _accumulate_synaptic_currents_jit(
            np.ascontiguousarray(target_indices, dtype=np.int64),
            np.ascontiguousarray(conductances, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(reversal_potentials, dtype=np.float64),
            np.ascontiguousarray(is_inhibitory, dtype=np.bool_),
            np.ascontiguousarray(membrane_potentials, dtype=np.float64),
        )

    synaptic_currents = np.zeros(len(membrane_potentials), dtype=np.float64)
    for c in range(len(target_indices)):
        tgt = target_indices[c]
        g = conductances[c]
        w = weights[c]
        e_syn = reversal_potentials[c]
        v = membrane_potentials[tgt]
        if is_inhibitory[c]:
            synaptic_currents[tgt] += g * w * (v - e_syn)
        else:
            synaptic_currents[tgt] += g * w * (e_syn - v)
    return synaptic_currents