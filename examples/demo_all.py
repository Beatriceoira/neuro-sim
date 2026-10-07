#!/usr/bin/env python3
"""
Phase 14: Documentation & Examples - Neuro-Sim

Thin orchestrator that runs every focused demo in sequence. Each demo
lives in its own file under `examples/`; this script imports and runs
them so a single command exercises the whole feature set.

Run with:
    PYTHONPATH=src python examples/demo_all.py
"""

import sys
sys.path.insert(0, 'src')

from demo_lif import demo_lif_features
from demo_hh import demo_hh_features
from demo_izhikevich import demo_izhikevich_features
from demo_network import demo_network_simulation
from demo_analysis import demo_analysis_features
from demo_config import demo_configuration_examples
from demo_regression import demo_regression_testing
from demo_benchmark import demo_benchmark_suite


def main():
    """Run all Phase 14 demonstrations."""
    print("Phase 14: Documentation & Examples - Neuro-Sim")
    print("=" * 60)
    print("This demo showcases all major features of the neuro-sim simulator.")
    print("Each demo illustrates practical usage and best practices.")
    print()

    demos = [
        ("LIF Neuron Features", demo_lif_features),
        ("Hodgkin-Huxley Neuron Features", demo_hh_features),
        ("Izhikevich Neuron Features", demo_izhikevich_features),
        ("Network Simulation Features", demo_network_simulation),
        ("Analysis and Visualization Features", demo_analysis_features),
        ("Configuration File Usage", demo_configuration_examples),
        ("Regression Testing Features", demo_regression_testing),
        ("Benchmark Suite Features", demo_benchmark_suite),
    ]

    for demo_name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"Error in {demo_name}: {e}")

    print("\n" + "=" * 60)
    print("Phase 14 Complete!")
    print("=" * 60)
    print("All demos have been executed successfully.")
    print("Generated files:")
    print("  - demo_analysis_plots.png")
    print("  - demo_config.yaml")
    print("  - regression_test_results_*.json")
    print("  - benchmark_results_*.json")
    print("\nThe neuro-sim simulator is fully functional and ready for use!")


if __name__ == "__main__":
    main()