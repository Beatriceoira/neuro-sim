#!/usr/bin/env python3
"""
Regression testing demonstration.

Runs the Phase 13 regression suite, which checks that key simulation
outputs (spike counts, timing patterns) remain consistent across
code changes.
"""

import sys
sys.path.insert(0, 'src')

from neurosim.benchmark.regression_tests import RegressionTester


def demo_regression_testing():
    """Demonstrate regression testing features."""
    print("=" * 60)
    print("Demo: Regression Testing Features")
    print("=" * 60)

    # Run regression tests
    print("Running regression tests...")
    tester = RegressionTester()
    results = tester.run_all_regression_tests()

    print(f"\nRegression Test Results:")
    print(f"  Total tests: {results['total_tests']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Success rate: {results['success_rate']:.1%}")


if __name__ == "__main__":
    demo_regression_testing()