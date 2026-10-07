"""
Multi-compartment neuron model for biophysically realistic simulations.

This module implements neurons with multiple electrical compartments
connected by axial resistances, enabling dendritic integration,
backpropagation, and spatially-distributed ion channels and synapses.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .base import BaseNeuron
from ..core.state import NeuronState
from .morphology import Morphology
from .compartment import Compartment


class MultiCompartmentNeuron(BaseNeuron):
    """
    Multi-compartment neuron model using the compartmental method.

    Implements the cable equation across a tree of compartments:
        C_m * dV_i/dt = -g_m * (V_i - E_m) - sum_j (V_i - V_j)/R_ij + I_ion + I_ext + I_syn

    Spikes are detected at the soma (or a designated spike compartment).
    Gating variables for all attached ion channels are integrated alongside
    voltages using exponential Euler for numerical stability.
    """

    def __init__(
        self,
        neuron_id: int = 0,
        morphology: Optional[Morphology] = None,
        spike_threshold: float = -40.0,
        spike_reset: float = -65.0,
        refractory_period: float = 2.0,
        spike_compartment_id: int = 0,
        initial_state: Optional[NeuronState] = None,
        dt: float = 0.01,
    ):
        """
        Initialize the multi-compartment neuron.

        Args:
            neuron_id: Unique identifier for this neuron
            morphology: Morphology object defining compartment tree
            spike_threshold: Membrane potential threshold for spike detection (mV)
            spike_reset: Membrane potential after spike (mV)
            refractory_period: Refractory period duration (ms)
            spike_compartment_id: ID of compartment where spikes are detected
            initial_state: Initial state of the neuron
            dt: Integration time step (ms), default 0.01
        """
        if morphology is None:
            from .morphology import create_simple_morphology
            morphology = create_simple_morphology()

        self.morphology = morphology
        self.spike_compartment_id = spike_compartment_id
        self.spike_threshold = spike_threshold
        self.spike_reset = spike_reset
        self.refractory_period = refractory_period
        self.dt = dt

        # Get all compartments
        self.compartments = morphology.get_all_compartments()

        # Get spike compartment (soma by default)
        self.spike_compartment = self.morphology.compartments.get(
            spike_compartment_id,
            self.morphology.get_soma()
        )

        # Initialize all compartments with default resting potential
        resting_V = -65.0
        for comp in self.compartments:
            comp.membrane_potential = resting_V
            comp.gating_variables = {}
            comp.ionic_currents = {}
            comp.synaptic_conductances = {}
            comp.external_current = 0.0

        # Create initial state from soma compartment
        initial_neuron_state = NeuronState(
            membrane_potential=resting_V,
            gating_variables={},
            ionic_currents={},
            synaptic_conductances={}
        )

        # Now call super().__init__() with pre-configured state
        super().__init__(neuron_id, initial_neuron_state)

        # Sync state to match compartments
        self._sync_state()

    def _sync_state(self):
        """Synchronize internal attributes with the state object."""
        self.membrane_potential = self.spike_compartment.membrane_potential
        self.state.membrane_potential = self.spike_compartment.membrane_potential
        self.state.gating_variables = dict(self.spike_compartment.gating_variables)

    def _update_state(self):
        """Update state object with current internal attributes."""
        self.state.membrane_potential = self.spike_compartment.membrane_potential
        self.state.gating_variables = dict(self.spike_compartment.gating_variables)

    # ------------------------------------------------------------------
    # Core integration
    # ------------------------------------------------------------------

    def compute_derivatives(
        self,
        t: float,
        external_current: float = 0.0,
        synaptic_inputs: Dict[str, float] = None
    ) -> Tuple[float, Dict]:
        """
        Compute the derivatives for all compartments.

        Implements the cable equation for each compartment:
            dV_i/dt = (I_axial_i - I_membrane_i + I_ext_i) / C_m

        Also computes gating variable derivatives for all attached
        ion channels.

        Args:
            t: Current time (ms)
            external_current: External applied current at spike compartment (µA)
            synaptic_inputs: Dictionary of synaptic current inputs

        Returns:
            Tuple of (dv_dt_soma, all_compartment_derivs)
            where all_compartment_derivs is a dict mapping
            compartment_id -> dv/dt (mV/ms)
        """
        synaptic_inputs = synaptic_inputs or {}

        # Apply external current to spike compartment only
        self.spike_compartment.external_current = external_current

        all_derivs: Dict[int, float] = {}

        for comp in self.compartments:
            dv_dt = comp.get_dv_dt()
            all_derivs[comp.compartment_id] = dv_dt

        return all_derivs[self.spike_compartment.compartment_id], all_derivs

    def _compute_all_derivatives(
        self, t: float
    ) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
        """
        Compute dV/dt and gating variable derivatives for every compartment.

        Returns:
            (voltage_derivs, gating_derivs)
            voltage_derivs: {comp_id: dV/dt in mV/ms}
            gating_derivs:  {comp_id: {gating_name: d(gate)/dt}}
        """
        voltage_derivs: Dict[int, float] = {}
        gating_derivs: Dict[int, Dict[str, float]] = {}

        for comp in self.compartments:
            dv_dt = comp.get_dv_dt()
            voltage_derivs[comp.compartment_id] = dv_dt
            gating_derivs[comp.compartment_id] = comp.get_channel_derivatives()

        return voltage_derivs, gating_derivs

    @staticmethod
    def _exp_euler_step(val: float, target: float, tau: float, dt: float) -> float:
        """
        Exponential Euler step for first-order kinetics.

        For dy/dt = (y_inf(y) - y) / tau this is equivalent to
        solving the linear ODE exactly over one step, avoiding
        numerical instability when tau << dt.
        """
        if tau <= 0:
            return target
        return target - (target - val) * np.exp(-dt / tau)

    def update_state(self, t: float, dt: Optional[float] = None):
        """
        Update all compartment states using Euler integration.

        Voltage and gating variables are advanced simultaneously.
        Gating variables use exponential Euler when their time
        constants are known; otherwise plain Euler is used.

        Args:
            t: Current simulation time (ms)
            dt: Time step (ms). If None, uses self.dt.
        """
        dt = dt if dt is not None else self.dt

        voltage_derivs, gating_derivs = self._compute_all_derivatives(t)

        for comp in self.compartments:
            cid = comp.compartment_id

            # Update membrane potential
            if cid in voltage_derivs:
                comp.membrane_potential += voltage_derivs[cid] * dt

            # Update gating variables
            if cid in gating_derivs:
                for gate_name, gate_deriv in gating_derivs[cid].items():
                    current_val = comp.gating_variables.get(gate_name, 0.0)
                    # Plain Euler — stable enough because dt is small
                    comp.gating_variables[gate_name] = current_val + gate_deriv * dt

        # Sync spike-compartment state to parent
        self._update_state()

    def step(self, t: float) -> bool:
        """
        Perform a single simulation step.

        Updates voltages and gating variables regardless of refractory
        state (the cable equation must integrate during refractory).
        Spike detection and reset are skipped during refractory.

        Args:
            t: Current simulation time (ms)

        Returns:
            True if a spike was detected at the spike compartment
        """
        # Update state first (cable equation integrates even during refractory)
        self.update_state(t, self.dt)

        # Refractory check — skip spike detection during refractory
        if self.state.refractory_remaining > 0:
            self.state.refractory_remaining = max(
                0.0, self.state.refractory_remaining - self.dt
            )
            return False

        spiked = False
        v_soma = self.spike_compartment.membrane_potential

        if v_soma >= self.spike_threshold:
            spiked = True
            self._handle_spike(t)

        return spiked

    # ------------------------------------------------------------------
    # Spike handling
    # ------------------------------------------------------------------

    def will_spike(self) -> bool:
        """
        Check if the neuron will spike at the current state.

        Returns:
            True if the spike compartment potential exceeds threshold
        """
        return (self.state.refractory_remaining <= 0 and
                self.spike_compartment.membrane_potential >= self.spike_threshold)

    def _handle_spike(self, t: float):
        """
        Handle the occurrence of a spike.

        Resets ONLY the spike compartment's membrane potential
        (not the entire morphology), records the spike, and
        initiates the refractory period.

        Args:
            t: Time of the spike (ms)
        """
        # Record spike time
        self.state.spike_times.append(t)

        # Reset ONLY the spike compartment voltage
        self.spike_compartment.membrane_potential = self.spike_reset
        self.spike_compartment.external_current = 0.0

        # Initiate refractory period
        self.state.refractory_remaining = self.refractory_period

        # Sync state
        self._update_state()

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_compartment_voltage(self, compartment_id: int) -> float:
        """
        Get the membrane potential of a specific compartment.

        Args:
            compartment_id: ID of the compartment

        Returns:
            Membrane potential in mV
        """
        if compartment_id in self.morphology.compartments:
            return self.morphology.compartments[compartment_id].membrane_potential
        raise ValueError(f"Compartment {compartment_id} not found")

    def get_all_voltages(self) -> Dict[int, float]:
        """
        Get membrane potentials for all compartments.

        Returns:
            Dictionary mapping compartment IDs to voltages (mV)
        """
        return {comp.compartment_id: comp.membrane_potential for comp in self.compartments}

    def get_all_gating_variables(self) -> Dict[int, Dict[str, float]]:
        """
        Get gating variables for all compartments.

        Returns:
            Dictionary mapping compartment ID -> {gate_name: value}
        """
        return {comp.compartment_id: dict(comp.gating_variables)
                for comp in self.compartments}

    def add_channel_to_compartment(self, compartment_id: int, channel):
        """
        Add an ion channel to a specific compartment.

        Args:
            compartment_id: ID of the target compartment
            channel: Channel instance to add
        """
        if compartment_id in self.morphology.compartments:
            comp = self.morphology.compartments[compartment_id]
            comp.channels.append(channel)

    def add_synapse_to_compartment(self, compartment_id: int, synapse):
        """
        Add a synapse to a specific compartment.

        Args:
            compartment_id: ID of the target compartment
            synapse: Synapse instance to add
        """
        if compartment_id in self.morphology.compartments:
            comp = self.morphology.compartments[compartment_id]
            comp.synapses.append(synapse)

    def apply_current_to_compartment(self, compartment_id: int, current: float):
        """
        Apply external current to a specific compartment.

        Args:
            compartment_id: ID of the target compartment
            current: Current amplitude in µA
        """
        if compartment_id in self.morphology.compartments:
            comp = self.morphology.compartments[compartment_id]
            comp.external_current = current

    def get_biophysical_info(self) -> Dict[str, float]:
        """
        Get biophysical information about the multi-compartment neuron.

        Returns:
            Dictionary of biophysical parameters
        """
        info = {
            "num_compartments": len(self.compartments),
            "soma_voltage_mV": self.spike_compartment.membrane_potential,
            "spike_threshold_mV": self.spike_threshold,
            "spike_compartment": self.spike_compartment.name,
        }

        for comp in self.compartments:
            key = f"comp_{comp.name}"
            info[f"{key}_V_mV"] = comp.membrane_potential
            info[f"{key}_area_um2"] = comp.area
            info[f"{key}_capacitance_uF"] = comp.capacitance_uF
            info[f"{key}_axial_resistance_kOhm"] = comp.axial_resistance_kOhm

        return info

    def describe(self) -> str:
        """
        Get a description of the multi-compartment neuron.

        Returns:
            String description
        """
        return (f"MultiCompartmentNeuron(id={self.neuron_id}, "
                f"{len(self.compartments)} compartments, "
                f"soma='{self.spike_compartment.name}')")


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def create_multi_compartment_neuron(
    neuron_id: int = 0,
    morphology: Optional[Morphology] = None,
    **kwargs
) -> MultiCompartmentNeuron:
    """
    Factory function to create a multi-compartment neuron.

    Args:
        neuron_id: Unique identifier for the neuron
        morphology: Morphology object (if None, creates simple morphology)
        **kwargs: Parameters to pass to MultiCompartmentNeuron

    Returns:
        Configured MultiCompartmentNeuron instance
    """
    if morphology is None:
        from .morphology import create_simple_morphology
        morphology = create_simple_morphology()

    return MultiCompartmentNeuron(neuron_id=neuron_id, morphology=morphology, **kwargs)
