#!/usr/bin/env python3
"""
Configuration file usage demonstration.

Prints an example YAML configuration and writes a small config file
to disk so users can see the expected format.
"""

import sys
sys.path.insert(0, 'src')

import yaml


def demo_configuration_examples():
    """Demonstrate configuration file usage."""
    print("=" * 60)
    print("Demo: Configuration File Usage")
    print("=" * 60)

    # Example YAML configuration
    config_example = """
# Neuro-Sim Configuration Example
simulation:
  duration: 500.0        # ms
  dt: 0.01              # ms
  spike_threshold: -40.0 # mV
  spike_reset: -65.0     # mV
  refractory_period: 2.0 # ms

neurons:
  - type: "lif"
    id: 0
    membrane_resistance: 10.0  # MΩ
    resting_potential: -65.0   # mV
    threshold_potential: -50.0 # mV
  - type: "izhikevich"
    id: 1
    a: 0.02
    b: 0.2
    c: -65.0
    d: 8.0

stimuli:
  - type: "step"
    amplitude: 20.0    # pA
    start_time: 100.0  # ms
    end_time: 300.0    # ms

synapses:
  - source: 0
    target: 1
    type: "excitatory"
    weight: 0.5
    delay: 1.0      # ms

analysis:
  variables: ["voltage", "firing_rate"]
  outputs:
    - type: "plot"
      file: "voltage_trace.png"
    - type: "text"
      file: "spike_summary.txt"
"""

    print("Example configuration (YAML format):")
    print(config_example)

    # Create a simple config file
    config_data = {
        'simulation': {
            'duration': 500.0,
            'dt': 0.1,
            'spike_threshold': -40.0,
            'spike_reset': -65.0,
            'refractory_period': 2.0
        },
        'neurons': [
            {'type': 'lif', 'id': 0, 'membrane_resistance': 10.0,
             'resting_potential': -65.0, 'threshold_potential': -50.0},
        ],
        'analysis': {
            'variables': ['voltage'],
            'outputs': [{'type': 'plot', 'file': 'demo_config_plot.png'}]
        }
    }

    with open('demo_config.yaml', 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)

    print("Config example saved to: demo_config.yaml")


if __name__ == "__main__":
    demo_configuration_examples()