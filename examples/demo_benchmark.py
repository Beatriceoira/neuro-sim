#!/usr/bin/env python3
"""
Benchmark suite demonstration.

Runs the Phase 13 benchmark suite: numerical stability checks across
parameter ranges, scaling benchmarks, F-I curve validation, and
steady-state convergence tests.
"""

import sys
sys.path.insert(0, 'src')

from neurosim.benchmark import run_comprehensive_benchmark


def demo_benchmark_suite():
    """Demonstrate benchmark suite features."""
    print("=" * 60)
    print("Demo: Benchmark Suite Features")
    print("=" * 60)

    print("Running comprehensive benchmarks...")
    try:
        benchmark_results = run_comprehensive_benchmark()
        print("\nBenchmark completed successfully!")
        print(f"  Numerical stability: {'Pass' if benchmark_results['numerical_stability']['pass'] else 'Fail'}")
        print(f"  HH FI curve R²: {benchmark_results['FI_curve_validation']['hh_validation']['r_squared']:.3f}")
    except Exception as e:
        print(f"Benchmark error: {e}")


if __name__ == "__main__":
    demo_benchmark_suite()