"""
Tests for multi-compartment neuron models (Phase 10).

Covers:
- Compartment geometry and cable equation
- Morphology construction (simple, ball-and-stick, branching)
- Multi-compartment neuron creation and state management
- Channel addition and current computation
- Spike detection and refractory period
- Dendritic integration and signal propagation
- SWC file loading
"""

import pytest
import numpy as np
from typing import List, Tuple

import sys
sys.path.insert(0, 'src')

from neurosim.neurons.compartment import Compartment
from neurosim.neurons.morphology import (
    Morphology,
    create_simple_morphology,
    create_ball_and_stick,
    create_branching_dendrite,
)
from neurosim.neurons.multi_compartment import (
    MultiCompartmentNeuron,
    create_multi_compartment_neuron,
)
from neurosim.channels.channel_models import create_hh_channels
from neurosim.channels.sodium import SodiumChannel
from neurosim.channels.potassium import PotassiumChannel
from neurosim.channels.leak import LeakChannel
from neurosim.stimuli.step import StepCurrent
from neurosim.neurons.morphology import load_swc


# ======================================================================
# Compartment tests
# ======================================================================

class TestCompartment:
    """Tests for the Compartment class."""

    def test_basic_geometry(self):
        """Test compartment geometry calculations."""
        comp = Compartment(
            compartment_id=0,
            name="test",
            length=10.0,
            diameter=10.0,
        )
        # Surface area: π * d * l
        expected_area = np.pi * 10.0 * 10.0
        assert abs(comp.area - expected_area) < 1e-10

        # Area in cm²
        expected_area_cm2 = expected_area * 1e-8
        assert abs(comp.area_cm2 - expected_area_cm2) < 1e-16

        # Capacitance
        expected_cap = 1.0 * expected_area_cm2  # specific_capacitance = 1 µF/cm²
        assert abs(comp.capacitance_uF - expected_cap) < 1e-12

    def test_axial_resistance(self):
        """Test axial resistance calculation."""
        comp = Compartment(
            compartment_id=0,
            name="test",
            length=100.0,
            diameter=2.0,
            specific_axial_resistivity=100.0,
        )
        # R = ρ * L / (π * r²)
        radius_cm = 1.0e-4  # 1 µm in cm
        length_cm = 100.0e-4  # 100 µm in cm
        R_ohm = 100.0 * length_cm / (np.pi * radius_cm**2)
        R_mohm = R_ohm * 1e-6

        R_kohm = R_ohm * 1e-3  # Convert to kOhm
        assert abs(comp.axial_resistance_kOhm - R_kohm) < 1e-6

    def test_is_soma(self):
        """Test soma detection."""
        comp = Compartment(0, "soma")
        assert comp.is_soma
        assert not comp.children

    def test_axial_current(self):
        """Test axial current computation with parent-child."""
        parent = Compartment(0, "parent", length=10, diameter=10)
        child = Compartment(1, "child", length=10, diameter=5)
        parent.children.append(child)
        child.parent = parent

        parent.membrane_potential = -65.0
        child.membrane_potential = -60.0

        # Axial current direction: flows from higher to lower potential
        # Here child (-60) is more positive than parent (-65), so current
        # flows OUT of child toward parent → negative net current into child
        I = child.get_axial_current()
        assert I < 0  # Child loses current to parent

    def test_dv_dt_resting(self):
        """Test dV/dt at resting potential with no current."""
        comp = Compartment(
            0,
            "test",
            length=10,
            diameter=10,
            membrane_conductance=0.3,
            reversal_potential=-65.0,
        )
        comp.membrane_potential = -65.0
        comp.external_current = 0.0

        dv_dt = comp.get_dv_dt()
        assert abs(dv_dt) < 1e-10  # Should be near zero at rest

    def test_dv_dt_with_current(self):
        """Test dV/dt with external current injection."""
        comp = Compartment(
            0,
            "test",
            length=10,
            diameter=10,
            specific_capacitance=1.0,
        )
        comp.membrane_potential = -65.0
        comp.external_current = 1.0  # 1 µA

        dv_dt = comp.get_dv_dt()
        # dV/dt = I / C = 1 µA / (1 µF/cm² * area_cm²)
        expected_C = 1.0 * comp.area_cm2
        expected_dv_dt = 1.0 / expected_C
        assert abs(dv_dt - expected_dv_dt) < 1e-6


# ======================================================================
# Morphology tests
# ======================================================================

class TestMorphology:
    """Tests for the Morphology class."""

    def test_create_simple(self):
        """Test simple morphology creation."""
        morph = create_simple_morphology(num_dendrites=2)
        assert len(morph.compartments) == 3  # soma + 2 dendrites
        soma = morph.get_soma()
        assert soma.name == "soma"
        assert soma.is_soma
        assert len(soma.children) == 2

    def test_add_compartment(self):
        """Test adding compartments manually."""
        morph = Morphology()
        soma = morph.add_compartment(name="soma", length=20, diameter=20)
        dend1 = morph.add_compartment(name="dend1", length=100, diameter=2, parent_id=soma.compartment_id)

        assert len(morph.compartments) == 2
        assert dend1.parent is soma
        assert soma.children == [dend1]

    def test_connect_compartments(self):
        """Test connecting existing compartments."""
        morph = Morphology()
        c1 = morph.add_compartment(name="c1")
        c2 = morph.add_compartment(name="c2")
        c3 = morph.add_compartment(name="c3")

        morph.connect_compartments(c3.compartment_id, c1.compartment_id)
        assert c3.parent is c1
        assert c1.children == [c3]

    def test_get_compartments_at_distance(self):
        """Test distance-based compartment queries."""
        morph = create_simple_morphology(num_dendrites=2, dendrite_length=100)
        soma = morph.get_soma()

        close = morph.get_compartments_at_distance(soma.compartment_id, max_distance=50)
        assert len(close) == 0  # No dendrites within 50 µm

        far = morph.get_compartments_at_distance(soma.compartment_id, max_distance=150)
        assert len(far) == 2  # Both dendrites

    def test_get_path_length(self):
        """Test path length calculation."""
        morph = create_simple_morphology(num_dendrites=1, dendrite_length=100)
        soma = morph.get_soma()
        dendrite = soma.children[0]

        assert morph.get_path_length(dendrite.compartment_id) == 120.0  # soma(20) + dendrite(100)
        assert morph.get_path_length(soma.compartment_id) == 0.0


# ======================================================================
# Multi-compartment neuron tests
# ======================================================================

class TestMultiCompartmentNeuron:
    """Tests for the MultiCompartmentNeuron class."""

    def test_creation(self):
        """Test basic neuron creation."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        assert neuron.neuron_id == 0
        assert len(neuron.compartments) >= 1
        assert neuron.spike_compartment is not None

    def test_factory(self):
        """Test factory function."""
        neuron = create_multi_compartment_neuron(neuron_id=5)
        assert neuron.neuron_id == 5

    def test_get_all_voltages(self):
        """Test retrieving voltages from all compartments."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        voltages = neuron.get_all_voltages()
        assert len(voltages) == len(neuron.compartments)
        for v in voltages.values():
            assert v == -65.0  # All at rest

    def test_get_compartment_voltage(self):
        """Test getting voltage from a specific compartment."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()
        v = neuron.get_compartment_voltage(soma.compartment_id)
        assert v == -65.0

        with pytest.raises(ValueError):
            neuron.get_compartment_voltage(999)

    def test_add_channel(self):
        """Test adding ion channels to compartments."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()

        na = SodiumChannel()
        neuron.add_channel_to_compartment(soma.compartment_id, na)
        assert na in soma.channels

    def test_add_synapse(self):
        """Test adding synapses to compartments."""
        from neurosim.synapses.excitatory import ExcitatorySynapse

        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()

        syn = ExcitatorySynapse()
        neuron.add_synapse_to_compartment(soma.compartment_id, syn)
        assert syn in soma.synapses

    def test_apply_current(self):
        """Test applying current to a compartment."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()

        neuron.apply_current_to_compartment(soma.compartment_id, 5.0)
        assert soma.external_current == 5.0

    def test_biophysical_info(self):
        """Test biophysical information retrieval."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        info = neuron.get_biophysical_info()

        assert "num_compartments" in info
        assert info["num_compartments"] >= 1
        assert "soma_voltage_mV" in info


# ======================================================================
# Simulation tests
# ======================================================================

class TestSimulation:
    """Tests for multi-compartment neuron simulation dynamics."""

    def test_leak_only_resting(self):
        """Test that a leak-only neuron stays at rest."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()
        neuron.add_channel_to_compartment(soma.compartment_id, LeakChannel(conductance=0.3, reversal_potential=-65.0))

        for _ in range(100):
            neuron.step(0.01)  # 0.01 ms time step

        # Should remain near resting potential (leak channel keeps it stable)
        v_soma = neuron.get_compartment_voltage(soma.compartment_id)
        assert -66.0 < v_soma < -64.0

    def test_step_current_depoloarizes(self):
        """Test that step current depolarizes the soma."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()

        # Apply 0.001 µA (1 nA) to soma for 100 ms
        # This should cause a small but measurable depolarization
        for t in range(1000):
            neuron.step(float(t) * 0.1)
            if 100 <= t < 200:
                soma.external_current = 0.001
            else:
                soma.external_current = 0.0

        v_soma = neuron.get_compartment_voltage(soma.compartment_id)
        assert v_soma > -65.0  # Should be depolarized

    def test_spike_detection(self):
        """Test spike detection at threshold."""
        neuron = MultiCompartmentNeuron(
            neuron_id=0,
            spike_threshold=-40.0,
            spike_reset=-65.0,
        )
        soma = neuron.morphology.get_soma()
        soma.external_current = 100.0  # Strong current to elicit spike

        spikes = []
        for t in range(500):
            spiked = neuron.step(float(t) * 0.1)
            if spiked:
                spikes.append(float(t) * 0.1)

        # Should detect at least one spike
        assert len(spikes) > 0

    def test_refractory_period(self):
        """Test that refractory period prevents immediate respike."""
        neuron = MultiCompartmentNeuron(
            neuron_id=0,
            spike_threshold=-40.0,
            spike_reset=-65.0,
            refractory_period=2.0,
        )
        soma = neuron.morphology.get_soma()
        soma.external_current = 100.0

        spikes = []
        for t in range(1000):
            spiked = neuron.step(float(t) * 0.1)
            if spiked:
                spikes.append(float(t) * 0.1)

        # Check minimum interspike interval
        if len(spikes) >= 2:
            isi = [spikes[i+1] - spikes[i] for i in range(len(spikes)-1)]
            assert min(isi) >= 2.0  # At least refractory period

    def test_dendritic_integration(self):
        """Test that dendritic input affects soma potential."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()
        dendrite = soma.children[0]

        # Apply small current (1 nA) only to dendrite
        dendrite.external_current = 0.001

        for _ in range(1000):
            neuron.step(0.1)

        # Soma should be slightly depolarized due to dendritic coupling
        v_soma = neuron.get_compartment_voltage(soma.compartment_id)
        assert v_soma > -65.0

    def test_voltages_propagate(self):
        """Test that voltage changes propagate from soma to dendrites."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()
        dendrite = soma.children[0]

        # Apply small current to soma (1 nA)
        soma.external_current = 0.001

        for _ in range(500):
            neuron.step(0.1)

        # Both soma and dendrite should depolarize
        v_soma = neuron.get_compartment_voltage(soma.compartment_id)
        v_dend = neuron.get_compartment_voltage(dendrite.compartment_id)

        assert v_soma > -65.0
        assert v_dend > -65.0
        # At steady state they equalize; check intermediate that soma leads


# ======================================================================
# Channel integration tests
# ======================================================================

class TestChannelIntegration:
    """Tests for ion channel integration in multi-compartment neurons."""

    def test_hh_channels_soma(self):
        """Test adding HH channels to soma."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()

        channels = create_hh_channels()
        for ch in channels.values():
            neuron.add_channel_to_compartment(soma.compartment_id, ch)

        assert len(soma.channels) == 3  # Na, K, leak

    def test_channel_currents_computed(self):
        """Test that channel currents are computed correctly."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()

        # Add only leak channel
        leak = LeakChannel(conductance=0.3, reversal_potential=-65.0)
        neuron.add_channel_to_compartment(soma.compartment_id, leak)

        # At rest, current should be zero
        I = soma.compute_membrane_current()
        assert abs(I) < 1e-6

    def test_gating_variables_tracked(self):
        """Test that gating variables are tracked per compartment."""
        neuron = MultiCompartmentNeuron(neuron_id=0)
        soma = neuron.morphology.get_soma()

        na = SodiumChannel()
        neuron.add_channel_to_compartment(soma.compartment_id, na)

        # Gate values start empty; populate after one simulation step
        neuron.step(0.0)
        gating = neuron.get_all_gating_variables()
        assert soma.compartment_id in gating
        assert "na_m" in gating[soma.compartment_id]
        assert "na_h" in gating[soma.compartment_id]


# ======================================================================
# Morphology variety tests
# ======================================================================

class TestMorphologyVarieties:
    """Tests for different morphology types."""

    def test_ball_and_stick(self):
        """Test ball-and-stick morphology."""
        morph = create_ball_and_stick(soma_diameter=20, dendrite_length=200)
        assert len(morph.compartments) == 2
        soma = morph.get_soma()
        assert soma.name == "soma"
        assert len(soma.children) == 1
        assert soma.children[0].length == 200.0

    def test_branching_dendrite(self):
        """Test branching dendrite morphology."""
        morph = create_branching_dendrite(branch_order=2)
        # Should have soma + branching dendrites
        assert len(morph.compartments) > 1
        soma = morph.get_soma()
        assert soma.is_soma


# ======================================================================
# Edge case tests
# ======================================================================

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_empty_morphology(self):
        """Test neuron with minimal morphology."""
        morph = Morphology()
        morph.add_compartment(name="soma", length=10, diameter=10)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
        assert len(neuron.compartments) == 1

    def test_single_compartment(self):
        """Test neuron with single compartment behaves like LIF."""
        morph = Morphology()
        soma = morph.add_compartment(name="soma", length=20, diameter=20)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)

        # Add strong current
        soma.external_current = 100.0

        spiked = False
        for _ in range(1000):
            if neuron.step(0.1):
                spiked = True
                break

        assert spiked

    def test_large_morphology(self):
        """Test with larger morphology."""
        morph = create_branching_dendrite(branch_order=3)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
        assert len(neuron.compartments) > 5

        # Should simulate without errors
        for _ in range(100):
            neuron.step(0.1)


# ======================================================================
# SWC file loading tests
# ======================================================================

class TestSWCLoading:
    """Tests for SWC morphology file loading."""

    def test_load_swc_simple(self):
        """Test loading a simple SWC file."""
        import tempfile
        import os

        swc_content = """# Test SWC
1 1 0 0 0 10 -1
2 3 0 0 10 2 1
3 3 0 0 20 2 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.swc', delete=False) as f:
            f.write(swc_content)
            tmpfile = f.name

        try:
            morph = load_swc(tmpfile)
            assert len(morph.compartments) == 3
            soma = morph.get_soma()
            assert soma.name == "soma"
            assert soma.is_soma
        finally:
            os.unlink(tmpfile)

    def test_load_swc_types(self):
        """Test SWC type name mapping."""
        import tempfile
        import os

        # Test various SWC types
        swc_content = """# Test SWC with different types
1 1 0 0 0 10 -1
2 2 0 0 0 5 1
3 3 0 0 5 3 2
4 5 0 10 0 2 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.swc', delete=False) as f:
            f.write(swc_content)
            tmpfile = f.name

        try:
            morph = load_swc(tmpfile)
            names = {c.name for c in morph.compartments.values()}
            assert "soma" in names
            assert "axon" in names
            assert "dendrite" in names
            assert "apical_dendrite" in names
        finally:
            os.unlink(tmpfile)


# ======================================================================
# HH spiking in multi-compartment neurons
# ======================================================================

class TestHHSpiking:
    """Tests for Hodgkin-Huxley spiking in multi-compartment neurons."""

    def test_hh_soma_spiking(self):
        """Test that HH channels in soma produce action potentials."""
        from neurosim.channels.channel_models import create_hh_channels

        morph = create_simple_morphology(num_dendrites=1, dendrite_length=100, dendrite_diameter=2, soma_length=20, soma_diameter=20)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
        soma = morph.get_soma()

        # Add HH channels to soma
        channels = create_hh_channels()
        for ch in channels.values():
            neuron.add_channel_to_compartment(soma.compartment_id, ch)

        # Apply strong current to elicit spikes
        soma.external_current = 5.0

        # Run simulation
        spikes = []
        for t_ms in range(0, 200):
            t = float(t_ms) * 0.1
            if neuron.step(t):
                spikes.append(t)

        # Should produce at least one spike
        assert len(spikes) >= 1

    def test_hh_spikes_with_refractory(self):
        """Test that HH spikes respect refractory period."""
        from neurosim.channels.channel_models import create_hh_channels

        morph = create_simple_morphology(num_dendrites=1, dendrite_length=100, dendrite_diameter=2, soma_length=20, soma_diameter=20)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph, refractory_period=2.0)
        soma = morph.get_soma()

        channels = create_hh_channels()
        for ch in channels.values():
            neuron.add_channel_to_compartment(soma.compartment_id, ch)

        soma.external_current = 5.0

        spikes = []
        for t_ms in range(0, 300):
            t = float(t_ms) * 0.1
            if neuron.step(t):
                spikes.append(t)

        # Check interspike intervals respect refractory period
        if len(spikes) >= 2:
            isis = [spikes[i+1] - spikes[i] for i in range(len(spikes)-1)]
            assert min(isis) >= 2.0  # At least refractory_period


# ======================================================================
# Dendritic integration tests
# ======================================================================

class TestDendriticIntegration:
    """Tests for dendritic signal propagation and attenuation."""

    def test_voltage_attenuation(self):
        """Test that voltage attenuates with distance from soma."""
        morph = create_simple_morphology(num_dendrites=1, dendrite_length=200, dendrite_diameter=2, soma_length=20, soma_diameter=20)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
        soma = morph.get_soma()
        dendrite = soma.children[0]

        # Add leak channels to both compartments for stability
        neuron.add_channel_to_compartment(soma.compartment_id, LeakChannel())
        neuron.add_channel_to_compartment(dendrite.compartment_id, LeakChannel())

        # Apply current only to soma
        soma.external_current = 0.01  # Small current for subthreshold response

        for _ in range(1000):
            neuron.step(0.1)

        v_soma = neuron.get_compartment_voltage(soma.compartment_id)
        v_dend = neuron.get_compartment_voltage(dendrite.compartment_id)

        # Both should depolarize, but soma more than dendrite
        assert v_soma > -65.0
        assert v_dend > -65.0
        assert v_soma >= v_dend  # Soma should depolarize at least as much

    def test_dendritic_current_injection(self):
        """Test that dendritic current injection affects soma."""
        morph = create_simple_morphology(num_dendrites=1, dendrite_length=100, dendrite_diameter=2, soma_length=20, soma_diameter=20)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
        soma = morph.get_soma()
        dendrite = soma.children[0]

        # Apply current only to dendrite
        dendrite.external_current = 0.005

        for _ in range(1000):
            neuron.step(0.1)

        # Soma should depolarize due to dendritic coupling
        v_soma = neuron.get_compartment_voltage(soma.compartment_id)
        assert v_soma > -65.0


# ======================================================================
# Per-compartment properties tests
# ======================================================================

class TestPerCompartmentProperties:
    """Tests for per-compartment channel and synapse properties."""

    def test_different_channels_per_compartment(self):
        """Test adding different channel types to different compartments."""
        from neurosim.channels.channel_models import create_hh_channels
        from neurosim.channels.leak import LeakChannel

        morph = create_simple_morphology(num_dendrites=1, dendrite_length=100, dendrite_diameter=2, soma_length=20, soma_diameter=20)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
        soma = morph.get_soma()
        dendrite = soma.children[0]

        # Add HH channels to soma
        hh_channels = create_hh_channels()
        for ch in hh_channels.values():
            neuron.add_channel_to_compartment(soma.compartment_id, ch)

        # Add only leak to dendrite
        neuron.add_channel_to_compartment(dendrite.compartment_id, LeakChannel())

        # Verify channels are in correct compartments
        assert len(soma.channels) == 3  # Na, K, leak
        assert len(dendrite.channels) == 1  # leak only

    def test_different_properties_per_compartment(self):
        """Test that compartments can have different biophysical properties."""
        morph = create_simple_morphology(num_dendrites=1, dendrite_length=100, dendrite_diameter=2, soma_length=20, soma_diameter=20)
        soma = morph.get_soma()
        dendrite = soma.children[0]

        # Soma should have larger diameter
        assert soma.diameter > dendrite.diameter
        assert soma.length == 20.0
        assert dendrite.length == 100.0


# ======================================================================
# Simulation integration tests
# ======================================================================

class TestSimulationIntegration:
    """Integration tests for complete simulation workflows."""

    def test_full_simulation_recording(self):
        """Test recording voltages during full simulation."""
        morph = create_simple_morphology(num_dendrites=2, dendrite_length=100, dendrite_diameter=2, soma_length=20, soma_diameter=20)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
        soma = morph.get_soma()

        # Record voltages
        recorded_voltages = []
        for t_ms in range(0, 100):
            t = float(t_ms) * 0.1
            neuron.step(t)
            recorded_voltages.append(neuron.get_all_voltages())

        # Check that we recorded voltages for all compartments
        assert len(recorded_voltages) == 100
        assert all(len(v) == len(morph.compartments) for v in recorded_voltages)

    def test_gating_variable_dynamics(self):
        """Test that gating variables evolve correctly."""
        from neurosim.channels.sodium import SodiumChannel

        morph = create_simple_morphology(num_dendrites=1, dendrite_length=100, dendrite_diameter=2, soma_length=20, soma_diameter=20)
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
        soma = morph.get_soma()

        na = SodiumChannel()
        neuron.add_channel_to_compartment(soma.compartment_id, na)

        # Gates are populated after first step
        neuron.step(0.0)
        initial_gates = neuron.get_all_gating_variables()[soma.compartment_id]
        assert 'na_m' in initial_gates
        assert 'na_h' in initial_gates

        # After depolarization, gates should change
        soma.external_current = 0.01
        for _ in range(100):
            neuron.step(0.1)

        final_gates = neuron.get_all_gating_variables()[soma.compartment_id]
        # m should increase with depolarization
        assert final_gates['na_m'] > initial_gates['na_m']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
