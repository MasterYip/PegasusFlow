#!/usr/bin/env python3

"""
Test script for ObsAvoid2 Environment Runner with enhanced trajectory optimization,
data collection, and transformer training capabilities.

This script demonstrates the complete workflow:
1. Data collection using sampling-based trajectory optimization
2. Training a transformer policy on collected data
3. Performance comparison between sampling and transformer policies
"""

import torch
import numpy as np
import sys
import os
from traj_sampling.env_runner.obsavoid2.obsavoid2_envrunner import ObsAvoid2EnvRunner


def test_enhanced_environment():
    """Test the enhanced obsavoid2 environment with high complexity."""
    print("\n" + "="*80)
    print("TESTING ENHANCED OBSAVOID2 ENVIRONMENT")
    print("="*80)
    
    # Create enhanced environment runner
    runner = ObsAvoid2EnvRunner(
        num_main_envs=4,
        num_rollout_per_main=16,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=150,
        horizon_samples=20,
        horizon_nodes=5,
        optimize_interval=2,
        env_type="complex",
        complexity_level=3,
        enable_trajectory_optimization=True,
        enable_vis=True,
        vis_time_window=2.0
    )
    
    print(f"Created enhanced environment runner:")
    print(f"  Complexity level: {runner.complexity_level}")
    print(f"  Environment type: {runner.env_type}")
    print(f"  Main environments: {runner.env.num_main_envs}")
    print(f"  Rollout environments per main: {runner.env.num_rollout_per_main}")
    print(f"  Observation dimension: {runner.env.obs_dim}")
    print(f"  Fractal boundaries: {runner.env.use_fractal_bounds}")
    print(f"  Adaptive difficulty: {runner.env.adaptive_difficulty}")
    
    # Run trajectory optimization
    results = runner.run_with_trajectory_optimization(seed=42)
    
    print(f"\nEnhanced Environment Results:")
    print(f"  Test mean score: {results['test_mean_score']:.4f}")
    print(f"  Episode time: {results['episode_time']:.2f} seconds")
    print(f"  Average step time: {results['step_time']*1000:.2f} ms")
    print(f"  Collision count: {results['collision_count']}")
    print(f"  Safety violations: {results['safety_violations']}")
    
    # Show environment performance metrics
    if 'environment_metrics' in results:
        env_metrics = results['environment_metrics']
        if env_metrics:
            print(f"  Environment efficiency:")
            if 'environment_efficiency' in env_metrics:
                eff = env_metrics['environment_efficiency']
                print(f"    Steps per second: {eff['steps_per_second']:.1f}")
                print(f"    Observations per second: {eff['obs_per_second']:.1f}")
    
    # Keep visualization open
    input("Press Enter to continue to data collection test...")
    runner.env.end()
    
    return results


def test_data_collection():
    """Test data collection for transformer training."""
    print("\n" + "="*80)
    print("TESTING DATA COLLECTION FOR TRANSFORMER TRAINING")
    print("="*80)
    
    # Create environment runner with data collection enabled
    runner = ObsAvoid2EnvRunner(
        num_main_envs=4,
        num_rollout_per_main=16,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=100,  # Shorter episodes for data collection
        horizon_samples=16,
        horizon_nodes=4,
        optimize_interval=1,
        env_type="complex",
        complexity_level=2,  # Medium complexity for faster data collection
        enable_trajectory_optimization=True,
        enable_data_collection=True,
        data_collection_mode="delta_traj",
        max_data_samples=1000,  # Smaller dataset for testing
        enable_vis=False  # Disable visualization for faster collection
    )
    
    print(f"Data collection settings:")
    print(f"  Collection mode: {runner.data_collection_mode.value}")
    print(f"  Target samples: {runner.max_data_samples}")
    print(f"  Complexity level: {runner.complexity_level}")
    
    # Collect training data
    data_summary = runner.collect_training_data(
        num_episodes=5,  # Small number for testing
        steps_per_episode=80,
        seed=123
    )
    
    print(f"\nData Collection Results:")
    print(f"  Episodes completed: {data_summary['episodes_completed']}")
    print(f"  Total data samples: {data_summary['total_data_samples']}")
    print(f"  Average episode reward: {data_summary['average_episode_reward']:.3f}")
    print(f"  Collection mode: {data_summary['data_collection_mode']}")
    
    # Save collected dataset
    if hasattr(runner.traj_sampler, 'save_collected_dataset'):
        dataset_path = "./obsavoid2_training_data.pt"
        runner.traj_sampler.save_collected_dataset(dataset_path)
        print(f"  Dataset saved to: {dataset_path}")
    
    return data_summary, runner


def test_transformer_training(runner, data_summary):
    """Test transformer policy training on collected data."""
    print("\n" + "="*80)
    print("TESTING TRANSFORMER POLICY TRAINING")
    print("="*80)
    
    if data_summary['total_data_samples'] < 100:
        print("Insufficient data samples for training, skipping transformer test")
        return None
    
    print(f"Training transformer on {data_summary['total_data_samples']} samples...")
    
    # Train transformer policy
    training_results = runner.train_transformer_policy(
        num_epochs=50,  # Small number for testing
        batch_size=32,
        learning_rate=1e-4,
        validation_split=0.2,
        save_checkpoint_dir="./obsavoid2_checkpoints"
    )
    
    print(f"\nTransformer Training Results:")
    print(f"  Training completed successfully")
    print(f"  Final training loss: {training_results['training_losses'][-1]:.6f}")
    if training_results['validation_losses']:
        print(f"  Final validation loss: {training_results['validation_losses'][-1]:.6f}")
    
    return training_results


def test_transformer_comparison(runner):
    """Test performance comparison between sampling and transformer policies."""
    print("\n" + "="*80)
    print("TESTING SAMPLING VS TRANSFORMER COMPARISON")
    print("="*80)
    
    try:
        # Test transformer vs sampling performance
        comparison_results = runner.deploy_and_test_transformer(seed=200)
        
        print(f"\nPerformance Comparison Results:")
        summary = comparison_results['comparison_summary']
        print(f"  Sampling policy score: {summary['sampling_score']:.4f}")
        print(f"  Transformer policy score: {summary['transformer_score']:.4f}")
        print(f"  Performance improvement: {summary['reward_improvement']:.4f}")
        print(f"  Speed improvement: {summary['speed_improvement']:.1f}%")
        
        # Determine winner
        if summary['reward_improvement'] > 0:
            print(f"  🏆 Transformer policy outperformed sampling by {summary['reward_improvement']:.4f}")
        elif summary['reward_improvement'] < -0.01:
            print(f"  📉 Sampling policy outperformed transformer by {-summary['reward_improvement']:.4f}")
        else:
            print(f"  🤝 Performance is similar between policies")
        
        return comparison_results
        
    except Exception as e:
        print(f"Transformer comparison failed: {e}")
        return None


def test_multi_complexity_comparison():
    """Test different complexity levels to show scalability."""
    print("\n" + "="*80)
    print("TESTING MULTI-COMPLEXITY COMPARISON")
    print("="*80)
    
    complexity_results = {}
    
    for complexity_level in [1, 2, 3]:
        print(f"\nTesting complexity level {complexity_level}...")
        
        runner = ObsAvoid2EnvRunner(
            num_main_envs=2,
            num_rollout_per_main=8,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            max_steps=80,
            horizon_samples=12,
            horizon_nodes=3,
            optimize_interval=2,
            env_type="complex",
            complexity_level=complexity_level,
            enable_trajectory_optimization=True,
            enable_vis=False
        )
        
        results = runner.run_with_trajectory_optimization(seed=300 + complexity_level)
        
        complexity_results[complexity_level] = {
            'score': results['test_mean_score'],
            'step_time': results['step_time'],
            'optimization_time': results['optimization_time'],
            'collision_count': results['collision_count']
        }
        
        print(f"  Complexity {complexity_level}: Score={results['test_mean_score']:.3f}, "
              f"StepTime={results['step_time']*1000:.1f}ms, "
              f"Collisions={results['collision_count']}")
        
        runner.env.end()
    
    print(f"\nComplexity Comparison Summary:")
    for level, result in complexity_results.items():
        print(f"  Level {level}: Score={result['score']:.3f}, "
              f"Time={result['step_time']*1000:.1f}ms, "
              f"Collisions={result['collision_count']}")
    
    return complexity_results


def test_performance_monitoring():
    """Test performance monitoring and metrics collection."""
    print("\n" + "="*80)
    print("TESTING PERFORMANCE MONITORING")
    print("="*80)
    
    runner = ObsAvoid2EnvRunner(
        num_main_envs=3,
        num_rollout_per_main=12,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=60,
        horizon_samples=16,
        horizon_nodes=4,
        complexity_level=3,
        enable_trajectory_optimization=True,
        enable_vis=False
    )
    
    # Run with performance monitoring
    results = runner.run_with_trajectory_optimization(seed=400)
    
    # Get comprehensive performance summary
    perf_summary = runner.get_performance_summary()
    
    print(f"Performance Monitoring Results:")
    print(f"  Environment Configuration:")
    env_config = perf_summary['environment_config']
    for key, value in env_config.items():
        print(f"    {key}: {value}")
    
    print(f"  Performance Metrics:")
    perf_metrics = perf_summary['performance_metrics']
    if perf_metrics['step_times']:
        print(f"    Average step time: {np.mean(perf_metrics['step_times'])*1000:.2f} ms")
        print(f"    Step time std: {np.std(perf_metrics['step_times'])*1000:.2f} ms")
    if perf_metrics['reward_history']:
        print(f"    Average reward: {np.mean(perf_metrics['reward_history']):.3f}")
        print(f"    Reward std: {np.std(perf_metrics['reward_history']):.3f}")
    
    # Environment-specific metrics
    env_metrics = results.get('environment_metrics', {})
    if env_metrics and 'computation_times' in env_metrics:
        comp_times = env_metrics['computation_times']
        print(f"  Environment Computation Times:")
        print(f"    Average step time: {comp_times['avg_step_time']*1000:.2f} ms")
        print(f"    Average obs time: {comp_times['avg_obs_time']*1000:.2f} ms")
        print(f"    Average reward time: {comp_times['avg_reward_time']*1000:.2f} ms")
    
    runner.env.end()
    return perf_summary


def main():
    """Main test function."""
    print("ObsAvoid2 Enhanced Environment Test Suite")
    print("========================================")
    print("High-performance trajectory optimization with data collection and transformer training")
    
    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("./obsavoid2_checkpoints", exist_ok=True)
    os.makedirs("./obsavoid2_results", exist_ok=True)
    
    all_results = {}
    
    try:
        # Test 1: Enhanced environment with high complexity
        print("\n🚀 Running enhanced environment test...")
        enhanced_results = test_enhanced_environment()
        all_results['enhanced_environment'] = enhanced_results
        
        # Test 2: Data collection for transformer training
        print("\n📊 Running data collection test...")
        data_summary, runner = test_data_collection()
        all_results['data_collection'] = data_summary
        
        # Test 3: Transformer training (if enough data collected)
        if data_summary['total_data_samples'] >= 100:
            print("\n🤖 Running transformer training test...")
            training_results = test_transformer_training(runner, data_summary)
            all_results['transformer_training'] = training_results
            
            # Test 4: Performance comparison
            if training_results is not None:
                print("\n⚔️ Running sampling vs transformer comparison...")
                comparison_results = test_transformer_comparison(runner)
                all_results['performance_comparison'] = comparison_results
        else:
            print("\n⚠️ Skipping transformer training - insufficient data samples")
        
        runner.env.end()
        
        # Test 5: Multi-complexity comparison
        print("\n📈 Running multi-complexity comparison...")
        complexity_results = test_multi_complexity_comparison()
        all_results['complexity_comparison'] = complexity_results
        
        # Test 6: Performance monitoring
        print("\n📊 Running performance monitoring test...")
        perf_summary = test_performance_monitoring()
        all_results['performance_monitoring'] = perf_summary
        
        # Final summary
        print("\n" + "="*80)
        print("OBSAVOID2 TEST SUITE SUMMARY")
        print("="*80)
        
        if 'enhanced_environment' in all_results:
            result = all_results['enhanced_environment']
            print(f"✅ Enhanced Environment: Score={result['test_mean_score']:.3f}, "
                  f"Time={result['episode_time']:.1f}s")
        
        if 'data_collection' in all_results:
            result = all_results['data_collection']
            print(f"✅ Data Collection: {result['total_data_samples']} samples collected")
        
        if 'transformer_training' in all_results:
            print(f"✅ Transformer Training: Completed successfully")
        
        if 'performance_comparison' in all_results:
            result = all_results['performance_comparison']
            if result:
                summary = result['comparison_summary']
                print(f"✅ Performance Comparison: Improvement={summary['reward_improvement']:.3f}")
        
        if 'complexity_comparison' in all_results:
            result = all_results['complexity_comparison']
            print(f"✅ Complexity Testing: {len(result)} levels tested")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Results saved to: ./obsavoid2_results/")
        print(f"💾 Checkpoints saved to: ./obsavoid2_checkpoints/")
        
        # Save comprehensive results
        results_path = "./obsavoid2_results/test_results.pt"
        torch.save(all_results, results_path)
        print(f"📊 Comprehensive results saved to: {results_path}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)