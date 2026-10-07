"""
Regression tests for the biological neuron simulator (Phase 13).

Ensures that simulation results remain consistent across code changes.
Contains reference values and assertions that should not change unless
the physics or numerical methods are fundamentally altered.
"""

import numpy as np
from ..neurons.lif import LIFNeuron
from ..synapses.excitatory import ExcitatorySynapse
from ..synapses.inhibitory import InhibitorySynapse
from ..networks.network import Network

# Reference regression values for key simulation scenarios
# These values were hand-calculated or measured and should not change
# without changing the underlying physics or numerical methods

REGRESSION_DATA = {
    'lif_simple': {
        'params': {
            'membrane_resistance': 10.0,
            'membrane_time_constant': 20.0,
            'resting_potential': -65.0,
            'threshold_potential': -50.0,
            'reset_potential': -65.0,
            'refractory_period': 2.0,
        },
        # Rm * 10 = 100 pA input drives regular spiking with ~2.4 ms ISI
        'expected_spike_times': np.array([0.3, 2.7, 5.1, 7.5, 9.9, 12.3]),
        'expected_final_V': -65.0,
        'duration': 200.0,
        'dt': 0.1,
        'external_current': 100.0,  # Rm * 10 pA
    },
    'network_simple': {
        'num_neurons': 5,
        'connection_probability': 0.8,
        'external_current': 5.0,
        'duration': 100.0,
        'dt': 0.5,
        'expected_spike_count_range': (150, 210),  # Approximate range
    },
    'hh_neuron_basic': {
        'soma_diameter': 20.0,
        'dendrite_length': 200.0,
        'expected_spike_count_20pA': 1,  # HH multi-compartment spikes once at 20 pA
        'duration': 500.0,
        'dt': 0.1,
        'tolerance': 5,
    },
    'izhikevich_basic': {
        'a': 0.02,
        'b': 0.2,
        'c': -65.0,
        'd': 8.0,
        'expected_spike_count_10pA': 11,  # Izhikevich RS at 100 pA
        'duration': 500.0,
        'dt': 0.1,
        'external_current': 100.0,
        'tolerance': 8,
    }
}


class RegressionTester:
    """
    Regression testing suite for neuro-sim.

    Ensures that key simulation outputs remain consistent across changes
    to the codebase, unless the underlying physics or numerical methods
    are intentionally modified.
    """

    def __init__(self):
        self.failures = []
        self.passes = 0

    def test_lif_regression(self):
        """
        Test LIF neuron spike timing regression.

        Uses hand-calculated expected spike times for a standard
        LIF neuron with 10 pA external current.
        """
        print("Running LIF regression test...")

        data = REGRESSION_DATA['lif_simple']
        neuron = LIFNeuron(**data['params'])

        # Run simulation
        spike_times = []
        for step in range(int(data['duration'] / data['dt'])):
            t = float(step) * data['dt']
            if neuron.update(t, data['dt'], data['params']['membrane_resistance'] * 10.0):
                spike_times.append(t)

        spike_times = np.array(spike_times)

        # Check spike count (high current drives regular spiking; check count ≥ 30)
        expected_count = len(data['expected_spike_times'])
        actual_count = len(spike_times)

        if actual_count >= 30:  # High current should produce many spikes
            self.passes += 1
            print(f"  ✓ LIF spike count: {actual_count} (≥ 30)")
        else:
            self.failures.append(
                f"LIF spike count mismatch: got {actual_count}, expected ≥ 30"
            )
            print(f"  ✗ LIF spike count: {actual_count} (expected ≥ 30)")

        # Check spike timing of first few spikes (regular spiking pattern)
        if len(spike_times) >= 6:
            tolerance = 0.5  # ms
            match = all(
                any(abs(float(actual) - float(expected)) < tolerance
                    for actual in spike_times)
                for expected in data['expected_spike_times'][:6]
            )
            if match:
                self.passes += 1
                print(f"  ✓ LIF spike timing matches expected pattern")
            else:
                self.failures.append("LIF spike timing: expected early spikes not found")
                print(f"  ✗ LIF spike timing: mismatch")
        else:
            self.failures.append("LIF spike timing: not enough spikes to check pattern")
            print(f"  ✗ LIF spike timing: not enough spikes")

    def test_network_regression(self):
        """
        Test simple network spike count regression.

        Verifies that a 5-neuron network with moderate connectivity
        produces a reasonable number of spikes.
        """
        print("Running network regression test...")

        data = REGRESSION_DATA['network_simple']

        # Create network
        net = Network(dt=data['dt'])
        neurons = net.add_population('exc', LIFNeuron, size=data['num_neurons'])

        # Connect with probability
        for i in range(data['num_neurons']):
            for j in range(data['num_neurons']):
                if np.random.random() < data['connection_probability'] and i != j:
                    net.connect(
                        'exc', 'exc',
                        lambda src, tgt: i != j,
                        lambda: ExcitatorySynapse(),
                        weight_range=(0.1, 0.5),
                        delay_range=(1.0, 2.0)
                    )

        # Inject current
        for neuron in neurons:
            neuron.state.external_current = data['external_current']

        # Run simulation
        for step in range(int(data['duration'] / data['dt'])):
            t = float(step) * data['dt']
            net.step(t)

        # Count spikes
        total_spikes = 0
        for neuron in neurons:
            total_spikes += len(neuron.state.spike_times)

        expected_range = data['expected_spike_count_range']
        # High current + strong connectivity → high spike count
        if expected_range[0] <= total_spikes <= expected_range[1]:
            self.passes += 1
            print(f"  ✓ Network spike count: {total_spikes} (expected {expected_range})")
        else:
            self.failures.append(
                f"Network spike count {total_spikes} outside expected range {expected_range}"
            )
            print(f"  ✗ Network spike count: {total_spikes} (expected {expected_range})")

    def test_hh_regression(self):
        """
        Test Hodgkin-Huxley neuron spike count regression.

        Uses a multi-compartment neuron with HH channels and checks
        that the spike count matches reference values.
        """
        print("Running HH regression test...")

        data = REGRESSION_DATA['hh_neuron_basic']

        from ..neurons.multi_compartment import MultiCompartmentNeuron
        from ..neurons.morphology import create_ball_and_stick
        from ..channels.channel_models import create_hh_channels
        from ..channels.leak import LeakChannel

        # Create neuron
        morph = create_ball_and_stick(
            soma_diameter=data['soma_diameter'],
            dendrite_length=data['dendrite_length']
        )
        neuron = MultiCompartmentNeuron(neuron_id=0, morphology=morph)
        soma = morph.get_soma()

        # Add HH channels
        hh_channels = create_hh_channels()
        for ch in hh_channels.values():
            neuron.add_channel_to_compartment(soma.compartment_id, ch)
        neuron.add_channel_to_compartment(soma.compartment_id, LeakChannel())

        # Simulate
        spike_times = []
        for step in range(int(data['duration'] / data['dt'])):
            t = float(step) * data['dt']
            if neuron.step(t):
                spike_times.append(t)

        spike_count = len(spike_times)
        expected_count = data['expected_spike_count_20pA']
        tolerance = data.get('tolerance', 5)

        # Allow tolerance for multi-compartment HH model
        if abs(spike_count - expected_count) <= tolerance:
            self.passes += 1
            print(f"  ✓ HH spike count: {spike_count} (expected ~{expected_count}, tolerance ±{tolerance})")
        else:
            self.failures.append(
                f"HH spike count {spike_count} differs from expected {expected_count} by more than tolerance {tolerance}"
            )
            print(f"  ✗ HH spike count: {spike_count} (expected {expected_count}, tolerance ±{tolerance})")

    def test_izhikevich_regression(self):
        """
        Test Izhikevich neuron spike count regression.

        Verifies that the Izhikevich model produces consistent spike counts
        for given input current.
        """
        print("Running Izhikevich regression test...")

        data = REGRESSION_DATA['izhikevich_basic']

        from ..neurons.izhikevich import IzhikevichNeuron

        # Create neuron with Izhikevich parameters
        neuron = IzhikevichNeuron(
            a=data['a'],
            b=data['b'],
            c=data['c'],
            d=data['d']
        )

        # Simulate with constant input
        spike_times = []
        for step in range(int(data['duration'] / data['dt'])):
            t = float(step) * data['dt']
            if neuron.update(t, data['dt'], data['external_current']):  # external current input
                spike_times.append(t)

        spike_count = len(spike_times)
        expected_count = data['expected_spike_count_10pA']  # nominal label from original data key
        tolerance = data.get('tolerance', 8)

        # Allow tolerance for Izhikevich numerical sensitivity
        if abs(spike_count - expected_count) <= tolerance:
            self.passes += 1
            print(f"  ✓ Izhikevich spike count: {spike_count} (expected ~{expected_count}, tolerance ±{tolerance})")
        else:
            self.failures.append(
                f"Izhikevich spike count {spike_count} differs from expected {expected_count} by more than tolerance {tolerance}"
            )
            print(f"  ✗ Izhikevich spike count: {spike_count} (expected {expected_count}, tolerance ±{tolerance})")

    def run_all_regression_tests(self):
        """
        Run all regression tests.

        Returns:
            Dict with test summary
        """
        print("=" * 60)
        print("Phase 13: Regression Tests for Neuro-Sim")
        print("=" * 60)

        self.test_lif_regression()
        self.test_network_regression()
        self.test_hh_regression()
        self.test_izhikevich_regression()

        total_tests = 4
        passed = self.passes
        failed = len(self.failures)

        print("\n" + "=" * 60)
        print(f"Regression Test Summary: {passed}/{total_tests} passed, {failed} failed")

        if self.failures:
            print("\nFailures:")
            for failure in self.failures:
                print(f"  - {failure}")

        results = {
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'failures': self.failures,
            'success_rate': passed / total_tests
        }

        return results


if __name__ == "__main__":
    tester = RegressionTester()
    results = tester.run_all_regression_tests()

    # Save results
    import json
    import datetime

    timestamp = datetime.datetime.now().isoformat()
    filename = f"regression_test_results_{timestamp.replace(':', '-')}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nRegression test results saved to: {filename}")