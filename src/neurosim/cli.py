"""
Command-line interface for the biological neuron simulator.
"""

import argparse
import sys
import os
from typing import Optional

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Neuro-Sim: Biological Neuron Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neurosim run experiments/hodgkin_huxley.yaml
  neurosim visualize results/hodgkin_huxley/
  neurosim benchmark
  neurosim test
  neurosim list-models
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run an experiment')
    run_parser.add_argument('experiment', help='Path to experiment configuration file')
    run_parser.add_argument('--output', '-o', help='Output directory for results')

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize results')
    viz_parser.add_argument('results_dir', help='Directory containing simulation results')
    viz_parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                           help='Output format for plots')

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    bench_parser.add_argument('--neurons', nargs='+', type=int, default=[10, 100, 1000],
                             help='Number of neurons to test')
    bench_parser.add_argument('--duration', type=float, default=1000.0,
                             help='Simulation duration in ms')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run unit tests')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Verbose test output')

    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available neuron models')

    # Parse arguments
    args = parser.parse_args()

    if args.command == 'run':
        run_experiment(args.experiment, args.output)
    elif args.command == 'visualize':
        visualize_results(args.results_dir, args.format)
    elif args.command == 'benchmark':
        run_benchmarks(args.neurons, args.duration)
    elif args.command == 'test':
        run_tests(args.verbose)
    elif args.command == 'list-models':
        list_models()
    else:
        parser.print_help()


def run_experiment(experiment_path: str, output_dir: Optional[str]):
    """Run an experiment from configuration file."""
    print(f"Running experiment: {experiment_path}")
    # TODO: Implement experiment running
    print("Experiment running not yet implemented")


def visualize_results(results_dir: str, format: str):
    """Visualize simulation results."""
    print(f"Visualizing results from: {results_dir}")
    # TODO: Implement visualization
    print("Visualization not yet implemented")


def run_benchmarks(neuron_counts: list, duration: float):
    """Run performance benchmarks."""
    print(f"Running benchmarks for {neuron_counts} neurons, {duration} ms duration")
    # TODO: Implement benchmarks
    print("Benchmarks not yet implemented")


def run_tests(verbose: bool):
    """Run unit tests."""
    print("Running unit tests")
    # TODO: Implement test running
    print("Test running not yet implemented")


def list_models():
    """List available neuron models."""
    print("Available neuron models:")
    print("  - LIF (Leaky Integrate-and-Fire)")
    print("  - Adaptive LIF")
    print("  - Izhikevich")
    print("  - Hodgkin-Huxley")


if __name__ == "__main__":
    main()