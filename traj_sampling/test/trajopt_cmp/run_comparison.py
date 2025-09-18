#!/usr/bin/env python3
"""
Simple test script to run trajectory optimization comparison.

This script demonstrates how to use the comparison framework to evaluate
different trajectory optimization methods.
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from trajopt_cmp import TrajectoryOptimizationComparison, ExperimentConfig

def run_quick_test():
    """Run a quick test with minimal parameters."""
    print("Running quick trajectory optimization comparison test...")
    
    # Configure for quick test
    # Note: horizon_nodes represents sections, actual nodes will be horizon_nodes+1
    config = ExperimentConfig(
        env_name="navigation2d",
        horizon_nodes=7,  # Actual nodes: 6
        horizon_samples=31,  # Actual samples: 32
        num_samples=40,
        num_iterations=10,
        num_trials=1,
        noise_schedule_types=["exponential_decay"],
        results_dir="quick_test_results",
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        visualize_optimization=True,  # Enable optimization process visualization
        viz_save_interval=2,  # Save every 2 iterations
        viz_max_trajectories=2  # Show max 2 trials
    )
    
    # Run comparison
    comparison = TrajectoryOptimizationComparison(config)
    results = comparison.run_experiment()
    
    # Generate plots
    comparison.generate_plots(results)
    
    print("Quick test completed!")
    print(f"Results saved to: {config.results_dir}")

def run_full_navigation_test():
    """Run full test for navigation environment."""
    print("Running full navigation trajectory optimization comparison...")
    
    config = ExperimentConfig(
        env_name="navigation2d",
        horizon_nodes=15,  # Actual nodes: 8
        horizon_samples=63,  # Actual samples: 64
        num_samples=10,
        num_iterations=10,
        num_trials=1,
        # noise_schedule_types=["constant", "exponential_decay", "hierarchical"],
        noise_schedule_types=["exponential_decay"],
        results_dir="navigation_results",
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        # Noise sampling settings
        noise_sampler_type='lhs',  # None, 'mc', 'lhs', 'halton'
        noise_distribution='normal',  # 'normal', 'uniform'
        noise_sampler_seed=None,
    )
    
    comparison = TrajectoryOptimizationComparison(config)
    results = comparison.run_experiment()
    comparison.generate_plots(results)
    
    print("Navigation test completed!")

def run_full_pendulum_test():
    """Run full test for inverted pendulum environment."""
    print("Running full pendulum trajectory optimization comparison...")
    
    config = ExperimentConfig(
        env_name="inverted_pendulum",
        horizon_nodes=15,  # Actual nodes: 8
        horizon_samples=255,  # Actual samples: 64
        num_samples=20,
        num_iterations=20,
        num_trials=1,
        noise_schedule_types=["exponential_decay"],
        results_dir="pendulum_results",
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        # Noise sampling settings
        noise_sampler_type = 'lhs',  # None, 'mc', 'lhs', 'halton'
        noise_distribution = 'normal',  # 'normal', 'uniform'
        noise_sampler_seed = None,
    )
    
    comparison = TrajectoryOptimizationComparison(config)
    results = comparison.run_experiment()
    comparison.generate_plots(results)
    
    print("Pendulum test completed!")


def run_visualization_test():
    """Run a test specifically to demonstrate optimization process visualization."""
    print("Running optimization process visualization test...")
    
    config = ExperimentConfig(
        env_name="navigation2d",
        horizon_nodes=7,  # Small trajectory for faster demo
        horizon_samples=31,  # Small horizon
        num_samples=20,   # Few samples for quick demo
        num_iterations=20,  # Enough iterations to see progress
        num_trials=1,     # Multiple trials to see variance
        noise_schedule_types=["exponential_decay"],
        results_dir="visualization_test_results",
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        visualize_optimization=True,  # Enable optimization process visualization
        viz_save_interval=3,  # Save every 3 iterations
        viz_max_trajectories=10  # Show all 3 trials
    )
    
    # Run comparison
    comparison = TrajectoryOptimizationComparison(config)
    results = comparison.run_experiment()
    
    # Generate plots
    comparison.generate_plots(results)
    
    print("Optimization process visualization test completed!")
    print(f"Check {config.results_dir}/plots/optimization_process/ for visualization results")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trajectory optimization comparison tests")
    parser.add_argument('--test', type=str, default='quick',
                       choices=['quick', 'navigation', 'pendulum', 'visualization', 'all'],
                       help='Test type to run')
    
    args = parser.parse_args()
    
    if args.test == 'quick':
        run_quick_test()
    elif args.test == 'navigation':
        run_full_navigation_test()
    elif args.test == 'pendulum':
        run_full_pendulum_test()
    elif args.test == 'visualization':
        run_visualization_test()
    elif args.test == 'all':
        run_full_navigation_test()
        run_full_pendulum_test()
        run_visualization_test()