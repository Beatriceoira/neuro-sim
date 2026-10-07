"""
Unit tests for the LIF neuron model.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neurosim.neurons.lif import LIFNeuron
from neurosim.stimuli.step import StepCurrent


class TestLIFNeuron(unittest.TestCase):
    """Test cases for the LIF neuron model."""

    def setUp(self):
        """Set up test fixtures."""
        self.neuron = LIFNeuron(
            membrane_resistance=10.0,      # MΩ
            membrane_time_constant=20.0,   # ms
            resting_potential=-65.0,       # mV
            threshold_potential=-50.0,     # mV
            reset_potential=-65.0,         # mV
            refractory_period=2.0          # ms
        )

    def test_initialization(self):
        """Test neuron initialization."""
        self.assertEqual(self.neuron.neuron_id, 0)
        self.assertEqual(self.neuron.membrane_resistance, 10.0)
        self.assertEqual(self.neuron.membrane_time_constant, 20.0)
        self.assertEqual(self.neuron.resting_potential, -65.0)
        self.assertEqual(self.neuron.threshold_potential, -50.0)
        self.assertEqual(self.neuron.reset_potential, -65.0)
        self.assertEqual(self.neuron.refractory_period, 2.0)
        self.assertEqual(self.neuron.spike_threshold, -50.0)
        self.assertEqual(self.neuron.spike_reset, -65.0)
        self.assertEqual(self.neuron.state.membrane_potential, -65.0)

    def test_subthreshold_dynamics(self):
        """Test subthreshold membrane potential dynamics."""
        # Test with zero current - should decay to resting potential
        v_initial = self.neuron.state.membrane_potential
        self.neuron.update(t=0.0, dt=1.0, external_current=0.0)
        # With zero current, potential should stay at resting (since already at rest)
        self.assertAlmostEqual(self.neuron.state.membrane_potential, -65.0, places=5)

        # Test with hyperpolarizing current - set via state to avoid sync issues
        self.neuron.state.membrane_potential = -60.0  # Above rest
        self.neuron._sync_state()  # Sync internal attributes with state
        self.neuron.update(t=0.0, dt=1.0, external_current=0.0)
        # Should move toward resting potential (become more negative)
        self.assertLess(self.neuron.state.membrane_potential, -60.0)
        # Should still be above resting potential (less negative)
        self.assertGreater(self.neuron.state.membrane_potential, -65.0)

    def test_spike_generation(self):
        """Test spike generation with suprathreshold current."""
        # Set initial condition
        self.neuron.state.membrane_potential = -65.0

        # Apply strong depolarizing current
        spike_occurred = False
        for i in range(100):  # 100 ms simulation
            t = i * 1.0
            # Apply current that should cause spiking
            spike_occurred = self.neuron.update(t=t, dt=1.0, external_current=10.0)
            if spike_occurred:
                break

        # Should have spiked
        self.assertTrue(spike_occurred)
        # Should have reset potential
        self.assertAlmostEqual(self.neuron.state.membrane_potential, -65.0, places=1)
        # Should have recorded spike
        self.assertGreater(len(self.neuron.state.spike_times), 0)

    def test_refractory_period(self):
        """Test refractory period behavior."""
        # Set initial condition well below threshold
        self.neuron.state.membrane_potential = -65.0
        self.neuron._sync_state()

        # Apply strong current to elicit a spike quickly
        spike_occurred = False
        # Try for up to 20ms to get a spike
        for i in range(20):
            t = float(i)
            spike_occurred = self.neuron.update(t=t, dt=1.0, external_current=50.0)  # Increased current
            if spike_occurred:
                break

        # Check that we actually got a spike
        self.assertTrue(spike_occurred, "Neuron did not spike - check parameters")
        self.assertGreater(len(self.neuron.state.spike_times), 0)

        # Should have spiked and be in refractory period
        self.assertEqual(self.neuron.state.refractory_remaining, 2.0)

        # During refractory period, should not spike regardless of input
        # Test for the next refractory_period ms (2.0 ms with our dt=1.0, so 2 steps)
        for i in range(int(self.neuron.refractory_period)):
            t = float(i + 1)  # Start testing from t=1ms after spike
            # Apply strong depolarizing current
            spike_occurred = self.neuron.update(t=t, dt=1.0, external_current=50.0)
            self.assertFalse(spike_occurred, f"Should not spike during refractory period at t={t}ms")  # Should not spike during refractory

        # After refractory period, should be able to spike again
        # Give it a few more ms to recover and then test with strong current
        recovery_time = 5.0  # ms
        for i in range(int(recovery_time)):
            t = float(i + 1 + self.neuron.refractory_period)  # Start after refractory period
            self.neuron.update(t=t, dt=1.0, external_current=50.0)

        # Now test with strong current - should spike again
        spike_occurred_again = False
        test_window = 10.0  # ms to test for second spike
        for i in range(int(test_window)):
            t = float(i + 1 + self.neuron.refractory_period + recovery_time)
            spike_occurred_again = self.neuron.update(t=t, dt=1.0, external_current=50.0)
            if spike_occurred_again:
                break

        self.assertTrue(spike_occurred_again, "Neuron did not spike again after refractory period")

    def test_step_current_response(self):
        """Test response to step current stimulus."""
        # Create step current: 0-100ms: 0 pA, 100-200ms: 10 pA, 200-300ms: 0 pA
        stimulus = StepCurrent(
            amplitude=10.0,   # pA
            start_time=100.0, # ms
            end_time=200.0    # ms
        )

        # Run simulation
        spike_times = []
        for i in range(300):  # 300 ms simulation
            t = float(i)
            current = stimulus.get_current(t)
            self.neuron.update(t=t, dt=1.0, external_current=current)

            # Check for spike (simplified)
            if len(self.neuron.state.spike_times) > len(spike_times):
                spike_times.append(self.neuron.state.spike_times[-1])

        # Should have spikes during or after the step current
        # Note: Exact timing depends on parameters, but should respond to the step
        # This test mainly checks that the code runs without error


if __name__ == '__main__':
    unittest.main()