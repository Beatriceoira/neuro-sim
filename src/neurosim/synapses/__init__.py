"""Synapse models for the biological neuron simulator."""

from .base import BaseSynapse
from .excitatory import ExcitatorySynapse
from .inhibitory import InhibitorySynapse
from .stp import ShortTermPlasticitySynapse
from .stdp import STDSynapse

__all__ = [
    "BaseSynapse",
    "ExcitatorySynapse",
    "InhibitorySynapse",
    "ShortTermPlasticitySynapse",
    "STDSynapse",
]
