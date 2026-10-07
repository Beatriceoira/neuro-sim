"""
Compartment class for multi-compartment neuron models.

This module implements the basic electrical compartment unit used in
cable theory-based multi-compartment neuron models.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class Compartment:
    """
    Single electrical compartment for multi-compartment neuron models.

    Each compartment represents a small piece of neuronal membrane with:
    - Membrane capacitance and resistance
    - Ion channels (via ChannelManager)
    - Synapses
    - Axial coupling to parent and child compartments

    The cable equation for this compartment is:
        C_m * dV/dt = -g_m * (V - E_m) - sum(axial currents) + I_ion + I_ext + I_syn

    All units are chosen to be consistent:
    - Voltage: mV
    - Current: µA
    - Conductance: µS (µA/mV)
    - Capacitance: µF
    - Length: µm
    """

    def __init__(
        self,
        compartment_id: int,
        name: str = "",
        length: float = 10.0,       # µm
        diameter: float = 10.0,     # µm
        specific_capacitance: float = 1.0,  # µF/cm²
        specific_axial_resistivity: float = 100.0,  # Ohm*cm
        resting_potential: float = -65.0,     # mV
        membrane_conductance: float = 0.3,    # mS/cm²
        reversal_potential: float = -65.0     # mV
    ):
        """
        Initialize a compartment.

        Args:
            compartment_id: Unique identifier within the neuron
            name: Human-readable name (e.g., 'soma', 'dendrite_1')
            length: Length of the cylindrical compartment (µm)
            diameter: Diameter of the cylindrical compartment (µm)
            specific_capacitance: Specific membrane capacitance (µF/cm²)
            specific_axial_resistivity: Cytoplasmic resistivity (Ohm*cm)
            resting_potential: Resting membrane potential (mV)
            membrane_conductance: Specific membrane conductance (mS/cm²)
            reversal_potential: Reversal potential for leak current (mV)
        """
        self.compartment_id = compartment_id
        self.name = name or f"compartment_{compartment_id}"
        self.parent = None
        self.children: List['Compartment'] = []

        # Morphological parameters
        self.length = length       # µm
        self.diameter = diameter   # µm

        # Biophysical parameters
        self.specific_capacitance = specific_capacitance  # µF/cm²
        self.specific_axial_resistivity = specific_axial_resistivity  # Ohm*cm
        self.resting_potential = resting_potential  # mV
        self.membrane_conductance = membrane_conductance  # mS/cm²
        self.reversal_potential = reversal_potential  # mV

        # Compute derived quantities
        self._compute_geometry()

        # State variables
        self.membrane_potential = resting_potential  # mV
        self.gating_variables: Dict[str, float] = {}
        self.ionic_currents: Dict[str, float] = {}
        self.synaptic_conductances: Dict[str, float] = {}

        # Channel and synapse containers
        self.channels: List = []
        self.synapses: List = []
        self.external_current = 0.0  # µA

    def _compute_geometry(self):
        """Compute surface area, capacitance, and axial resistance with correct unit conversions."""
        # Surface area in µm²: π * d * l
        self.area = np.pi * self.diameter * self.length  # µm²

        # Convert to cm²: 1 µm² = 1e-8 cm²
        self.area_cm2 = self.area * 1e-8  # cm²

        # Membrane capacitance in µF
        self.capacitance_uF = self.specific_capacitance * self.area_cm2  # µF

        # Membrane conductance in mS (millisiemens)
        # g_m = specific_conductance (mS/cm²) * area (cm²)
        # I = g * (V - E) gives µA when g is in mS and V in mV
        self.membrane_conductance_total = self.membrane_conductance * self.area_cm2  # mS

        # Axial resistance in kOhm
        # R_axial = (Ra * L) / (pi * r²)
        # I = V / R gives µA when V is in mV and R in kOhm
        radius_cm = (self.diameter / 2.0) * 1e-4  # µm to cm
        length_cm = self.length * 1e-4  # µm to cm
        R_axial_Ohm = (
            self.specific_axial_resistivity * length_cm /
            (np.pi * radius_cm**2)
        )  # Ohm
        self.axial_resistance_kOhm = R_axial_Ohm * 1e-3  # kOhm

        # Axial conductance in µS (for reference)
        if self.axial_resistance_kOhm > 0:
            self.axial_conductance_uS = 1.0 / self.axial_resistance_kOhm  # µS
        else:
            self.axial_conductance_uS = 0.0

    @property
    def is_soma(self) -> bool:
        """Check if this compartment is the soma (has no parent)."""
        return self.parent is None

    def get_axial_current(self) -> float:
        """
        Compute the total axial current flowing into this compartment
        from neighboring compartments.

        In cable theory, the axial resistance between two connected
        compartments is the half-sum of their individual axial resistances.

        Returns:
            Total axial current in µA (positive = net inward)
        """
        total_current_uA = 0.0

        # Current from parent
        if self.parent is not None:
            parent = self.parent
            R_axial = parent.axial_resistance_kOhm / 2.0 + self.axial_resistance_kOhm / 2.0
            if R_axial > 0:
                total_current_uA += (parent.membrane_potential - self.membrane_potential) / R_axial

        # Currents from children
        for child in self.children:
            R_axial = self.axial_resistance_kOhm / 2.0 + child.axial_resistance_kOhm / 2.0
            if R_axial > 0:
                total_current_uA += (child.membrane_potential - self.membrane_potential) / R_axial

        return total_current_uA

    def compute_membrane_current(self) -> float:
        """
        Compute the total membrane current (leak + ion channels + synapses).

        Returns:
            Total membrane current in µA
        """
        # Leak current in µA: I = g_leak * (V - E_leak)
        I_leak = self.membrane_conductance_total * (self.membrane_potential - self.reversal_potential)

        # Ion channel currents (from attached channels)
        # Channel.current() returns current density in µA/cm²;
        # multiply by area to get total current in µA
        I_ion = 0.0
        for channel in self.channels:
            if hasattr(channel, 'current'):
                I_density = channel.current(self.membrane_potential, self.gating_variables)
                I_ion += I_density * self.area_cm2

        # Synaptic currents
        I_syn = 0.0
        for synapse in self.synapses:
            if hasattr(synapse, 'current'):
                V = self.membrane_potential
                state = synapse.state
                g_syn = state.get('g', 0.0)
                I_syn += g_syn * synapse.weight * (V - synapse.reversal_potential)

        return I_leak + I_ion + I_syn

    def get_dv_dt(self) -> float:
        """
        Compute dV/dt for this compartment.

        Returns:
            dV/dt in mV/ms
        """
        I_axial = self.get_axial_current()
        I_membrane = self.compute_membrane_current()
        I_total = I_axial - I_membrane + self.external_current

        # dV/dt = I_total / C_m
        # I in µA, C in µF => dV/dt in mV/ms
        if self.capacitance_uF > 0:
            return I_total / self.capacitance_uF
        else:
            return 0.0

    def get_channel_derivatives(self) -> Dict[str, float]:
        """
        Compute derivatives of channel gating variables.

        Returns:
            Dictionary mapping state variable names to their derivatives
        """
        derivatives = {}
        for channel in self.channels:
            if hasattr(channel, 'derivatives'):
                ch_derivs = channel.derivatives(self.membrane_potential, self.gating_variables)
                for var_name, deriv in ch_derivs.items():
                    # Prefix with channel id to avoid conflicts across channels
                    key = f"{channel.channel_id}_{var_name}"
                    derivatives[key] = deriv
        return derivatives

    def get_state_vector(self) -> Dict[str, float]:
        """
        Get the state vector for this compartment.

        Returns:
            Dictionary mapping state variable names to values
        """
        state = {'V': self.membrane_potential}
        state.update(self.gating_variables)
        return state

    def describe(self) -> str:
        """
        Get a description of the compartment.

        Returns:
            String description
        """
        return (f"Compartment(id={self.compartment_id}, name='{self.name}', "
                f"L={self.length}µm, d={self.diameter}µm, "
                f"V={self.membrane_potential:.1f}mV)")
