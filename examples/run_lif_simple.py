"""
Simple example script to run a LIF neuron with step current stimulus.
No plotting to avoid GUI issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from neurosim.neurons.lif import LIFNeuron
from neurosim.stimuli.step import StepCurrent
from neurosim.core.simulation import Simulation


def simulate_lif_with_step():
    """Simulate a LIF neuron with step current input."""

    # Create LIF neuron with typical parameters
    neuron = LIFNeuron(
        membrane_resistance=10.0,      # MΩ
        membrane_time_constant=20.0,   # ms
        resting_potential=-65.0,       # mV
        threshold_potential=-50.0,     # mV
        reset_potential=-65.0,         # mV
        refractory_period=2.0          # ms
    )

    # Create step current: 0-100ms: 0 pA, 100-200ms: 15 pA, 200-300ms: 0 pA
    stimulus = StepCurrent(
        amplitude=15.0,    # pA
        start_time=100.0,  # ms
        end_time=200.0     # ms
    )

    # Set up simulation - the deriv_func now updates the state directly
    def deriv_func(t, net_state, sim_state, spiked_neurons=None):
        # Get external current from stimulus
        I_ext = stimulus.get_current(t)
        # Compute neuron derivatives for the first (and only) neuron
        dv_dt, other_derivs = neuron.compute_derivatives(t, external_current=I_ext)
        # Apply Euler integration to update the membrane potential
        neuron.state.membrane_potential += dv_dt * net_state.dt

    # Create simulation
    sim = Simulation(
        deriv_function=deriv_func,
        dt=0.01,
        spike_threshold=-50.0,
        spike_reset=-65.0,
        refractory_period=2.0
    )

    # Override network state with our neuron's state
    sim.state.network_state.neuron_states = [neuron.state]

    # Run simulation
    print("Running LIF simulation with step current...")
    print(f"Simulation parameters: dt={sim.dt}ms, duration=300.0ms")
    final_state = sim.run(duration=300.0)  # Remove record_variables for simplicity

    # Extract results directly from neuron
    voltage_trace = final_state.get_recorded_variable('voltage') if 'voltage' in final_state.recorded_variables else np.array([])
    spike_times = np.array(neuron.state.spike_times)
    time_points = np.arange(0, 300.0, 0.01) if len(voltage_trace) > 0 else np.array([])

    print(f"Simulation completed!")
    print(f"Number of spikes: {len(spike_times)}")
    if len(spike_times) > 0:
        print(f"First spike at: {spike_times[0]:.2f} ms")
        print(f"Last spike at: {spike_times[-1]:.2f} ms")
        print(f"Spike times: {spike_times}")
    else:
        print("No spikes detected")

    # Simple text-based output
    if len(spike_times) > 0 and len(voltage_trace) > 0:
        # Show voltage around spike times
        print("\nVoltage samples around first spike:")
        first_spike_idx = int(spike_times[0] / 0.01)
        start_idx = max(0, first_spike_idx - 5)
        end_idx = min(len(time_points), first_spike_idx + 5)
        for i in range(start_idx, end_idx):
            t = time_points[i]
            v = voltage_trace[i]
            marker = " ***" if abs(t - spike_times[0]) < 0.02 else ""
            print(f"  t={t:6.2f}ms, V={v:6.2f}mV{marker}")

    return spike_times, voltage_trace, time_points


if __name__ == "__main__":
    simulate_lif_with_step()