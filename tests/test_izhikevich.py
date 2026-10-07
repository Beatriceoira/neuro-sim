"""
Unit tests for the Izhikevich neuron model.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neurosim.neurons.izhikevich import (
    IzhikevichNeuron,
    create_regular_spiking_neuron,
    create_intrinsically_bursting_neuron,
    create_chattering_neuron,
    create_fast_spiking_neuron,
    create_low_threshold_spiking_neuron
)
from neurosim.stimuli.step import StepCurrent


class TestIzhikevichNeuron(unittest.TestCase):
    """Test cases for the Izhikevich neuron model."""

    def test_regular_spiking_creation(self):
        """Test creating a regular spiking Izhikevich neuron."""
        neuron = create_regular_spiking_neuron()
        self.assertEqual(neuron.a, 0.02)
        self.assertEqual(neuron.b, 0.2)
        self.assertEqual(neuron.c, -65.0)
        self.assertEqual(neuron.d, 8.0)

    def test_intrinsically_bursting_creation(self):
        """Test creating an intrinsically bursting Izhikevich neuron."""
        neuron = create_intrinsically_bursting_neuron()
        self.assertEqual(neuron.a, 0.02)
        self.assertEqual(neuron.b, 0.2)
        self.assertEqual(neuron.c, -55.0)
        self.assertEqual(neuron.d, 4.0)

    def test_chattering_creation(self):
        """Test creating a chattering Izhikevich neuron."""
        neuron = create_chattering_neuron()
        self.assertEqual(neuron.a, 0.02)
        self.assertEqual(neuron.b, 0.2)
        self.assertEqual(neuron.c, -50.0)
        self.assertEqual(neuron.d, 2.0)

    def test_fast_spiking_creation(self):
        """Test creating a fast spiking Izhikevich neuron."""
        neuron = create_fast_spiking_neuron()
        self.assertEqual(neuron.a, 0.1)
        self.assertEqual(neuron.b, 0.2)
        self.assertEqual(neuron.c, -65.0)
        self.assertEqual(neuron.d, 2.0)

    def test_low_threshold_spiking_creation(self):
        """Test creating a low-threshold spiking Izhikevich neuron."""
        neuron = create_low_threshold_spiking_neuron()
        self.assertEqual(neuron.a, 0.02)
        self.assertEqual(neuron.b, 0.25)
        self.assertEqual(neuron.c, -65.0)
        self.assertEqual(neuron.d, 2.0)

    def test_izhikevich_dynamics(self):
        """Test basic Izhikevich neuron dynamics."""
        neuron = IzhikevichNeuron(
            a=0.02,
            b=0.2,
            c=-65.0,
            d=8.0
        )

        # Test initial state
        self.assertEqual(neuron.state.membrane_potential, -65.0)
        self.assertEqual(neuron.state.recovery_variable, 0.0)

        # Test derivatives at rest
        dv_dt, du_dt = neuron.compute_derivatives(t=0.0, external_current=0.0)
        # At V=-65, u=0: dv/dt = 0.04*(-65)^2 + 5*(-65) + 140 - 0 + 0 = -16
        self.assertAlmostEqual(dv_dt, -16.0, places=1)
        # du/dt = a*(b*V - u) = 0.02*(0.2*(-65) - 0) = 0.02*(-13) = -0.26
        self.assertAlmostEqual(du_dt['recovery_variable'], -0.26, places=2)

        # Test with zero input over multiple steps - should integrate
        v_initial = neuron.state.membrane_potential
        u_initial = neuron.state.recovery_variable
        # Take a few steps with zero current
        for _ in range(5):
            neuron.update(t=0.0, dt=1.0, external_current=0.0)
        # Should have changed (integrated) but not necessarily returned to initial
        # This test mainly checks that the code runs without error

    def test_izhikevich_spiking(self):
        """Test that Izhikevich neuron can spike with sufficient input."""
        neuron = create_regular_spiking_neuron(neuron_id=0)

        # Apply strong current to elicit spiking
        # Based on Izhikevich dynamics, we need sufficient current to overcome the negative dV/dt at rest
        spike_occurred = False
        for i in range(200):  # 200 ms simulation
            t = float(i)
            # Use higher current that should elicit spiking in regular spiking model
            spike_occurred = neuron.update(t=t, dt=1.0, external_current=20.0)
            if spike_occurred:
                # Check that we reset appropriately
                self.assertEqual(neuron.state.membrane_potential, neuron.c)
                break

        # Should have spiked
        self.assertTrue(spike_occurred, "Izhikevich neuron did not spike with sufficient input")
        # Should have recorded spike
        self.assertGreater(len(neuron.state.spike_times), 0)


if __name__ == '__main__':
    unittest.main()