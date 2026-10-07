"""
Unit tests for the Hodgkin-Huxley neuron model.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neurosim.neurons.hodgkin_huxley import HodgkinHuxleyNeuron
from neurosim.stimuli.step import StepCurrent


class TestHodgkinHuxleyNeuron(unittest.TestCase):
    """Test cases for the Hodgkin-Huxley neuron model."""

    def test_hh_creation(self):
        """Test creating a Hodgkin-Huxley neuron with default parameters."""
        neuron = HodgkinHuxleyNeuron()
        self.assertEqual(neuron.neuron_id, 0)
        self.assertEqual(neuron.membrane_capacitance, 1.0)
        self.assertEqual(neuron.g_na, 120.0)
        self.assertEqual(neuron.e_na, 50.0)
        self.assertEqual(neuron.g_k, 36.0)
        self.assertEqual(neuron.e_k, -77.0)
        self.assertEqual(neuron.g_l, 0.3)
        self.assertEqual(neuron.e_l, -54.387)

    def test_hh_initial_state(self):
        """Test initial state of Hodgkin-Huxley neuron."""
        neuron = HodgkinHuxleyNeuron()
        # Check initial membrane potential
        self.assertAlmostEqual(neuron.state.membrane_potential, -65.0, places=1)
        # Check initial gating variables
        self.assertAlmostEqual(neuron.state.gating_variables['m'], 0.05, places=2)
        self.assertAlmostEqual(neuron.state.gating_variables['h'], 0.6, places=1)
        self.assertAlmostEqual(neuron.state.gating_variables['n'], 0.32, places=2)

    def test_hh_derivatives_at_rest(self):
        """Test computing derivatives at initial state with zero input."""
        neuron = HodgkinHuxleyNeuron()
        # With initial conditions, compute expected derivatives
        dv_dt, other_derivs = neuron.compute_derivatives(t=0.0, external_current=0.0)
        # Expected values based on manual calculation:
        # I_Na = -1.035, I_K = 4.530, I_L = -3.184 (µA/cm²)
        # dV/dt = (0 - (-1.035) - 4.530 - (-3.184)) / 1.0 = -0.311 mV/ms
        self.assertAlmostEqual(dv_dt, -0.311, places=2)
        # Check individual currents
        self.assertAlmostEqual(other_derivs.get('I_Na', 0.0), -1.035, places=2)
        self.assertAlmostEqual(other_derivs.get('I_K', 0.0), 4.530, places=2)
        self.assertAlmostEqual(other_derivs.get('I_L', 0.0), -3.184, places=2)

    def test_hh_spiking(self):
        """Test that HH neuron can spike with sufficient input current."""
        neuron = HodgkinHuxleyNeuron()

        # Apply suprathreshold current to elicit spiking
        # Based on debug output, HH needs about 20 µA/cm² to spike reliably
        spike_occurred = False
        # Try for up to 50ms with strong current
        for i in range(50):
            t = float(i)
            # Use current that should elicit spiking in HH model
            spike_occurred = neuron.update(t=t, dt=1.0, external_current=20.0)
            if spike_occurred:
                break

        # Should have spiked
        self.assertTrue(spike_occurred, "HH neuron did not spike with sufficient input")
        # Should have recorded spike
        self.assertGreater(len(neuron.state.spike_times), 0)
        # After spike, membrane potential should be reset (approximately)
        # Note: exact reset depends on spike dynamics, but should be near reset potential
        self.assertLess(neuron.state.membrane_potential, -50.0)  # Should be reasonably hyperpolarized after spike

    def test_hh_ionic_currents(self):
        """Test that ionic currents are computed correctly."""
        neuron = HodgkinHuxleyNeuron()
        # Set to known state
        neuron.state.membrane_potential = -65.0
        neuron.state.gating_variables = {'m': 0.05, 'h': 0.6, 'n': 0.32}
        neuron._sync_state()

        # Compute derivatives (which includes current calculations)
        dv_dt, other_derivs = neuron.compute_derivatives(t=0.0, external_current=0.0)

        # Check that we got current values
        self.assertIn('I_Na', other_derivs)
        self.assertIn('I_K', other_derivs)
        self.assertIn('I_L', other_derivs)

        # Verify they have the expected values at rest
        self.assertAlmostEqual(other_derivs['I_Na'], -1.035, places=2)
        self.assertAlmostEqual(other_derivs['I_K'], 4.530, places=2)
        self.assertAlmostEqual(other_derivs['I_L'], -3.184, places=2)


if __name__ == '__main__':
    unittest.main()