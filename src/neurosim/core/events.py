"""Event types used by the simulation engine.

Events are the only mechanism by which information flows between neurons
across a network.  A presynaptic neuron emits a ``SpikeEvent``; synapses
translate that into ``SynapticEvent`` objects delivered to postsynaptic
neurons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpikeEvent:
    """A single action potential emitted by a neuron."""

    time: float
    source_id: int
    amplitude: float = 1.0

    def __lt__(self, other: "SpikeEvent") -> bool:
        return self.time < other.time


@dataclass
class SynapticEvent:
    """A synaptic activation delivered to a postsynaptic neuron."""

    time: float
    target_id: int
    weight: float = 1.0
    delay: float = 0.0
    synapse_type: str = "excitatory"

    def __lt__(self, other: "SynapticEvent") -> bool:
        return self.time < other.time


@dataclass
class EventQueue:
    """A simple time-ordered queue of events."""

    events: list = field(default_factory=list)

    def push(self, event) -> None:
        self.events.append(event)

    def pop(self, time: float) -> list:
        """Pop all events whose time <= ``time``."""
        ready = [e for e in self.events if e.time <= time]
        self.events = [e for e in self.events if e.time > time]
        return ready

    def __len__(self) -> int:
        return len(self.events)

    def clear(self) -> None:
        self.events.clear()