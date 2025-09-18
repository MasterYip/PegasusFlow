#!/usr/bin/env python3

"""
Test script demonstrating the usage of profiling decorators with TrajGradSampling.

This script shows how to:
1. Enable different profiling modes using environment variables
2. Use the profiling decorators with TrajGradSampling
3. Analyze and save profiling results
"""

import os
import sys
import torch
import numpy as np

from traj_sampling.traj_grad_sampling import TrajGradSampling, TrajGradSamplingCfg
from traj_sampling.utils.benchmark import (
    enable_profiling, disable_profiling, set_profiling_mode,
    print_time_profile_summary, print_gpu_profile_summary, 
    print_profile, save_profile_summary, clear_profile_data
)


def create_mock_rollout_callback():
    """Create a mock rollout callback for testing."""
    def rollout_callback(trajectories, rollout_callback, n_diffuse):
        # Mock rollout that just returns random rewards
        batch_size = trajectories.shape[0]
        rewards = torch.randn(batch_size, device=trajectories.device)
        return rewards
    return rollout_callback


def test_profiling_basic():
    """Test basic profiling functionality with TrajGradSampling."""
    print("Testing basic profiling with TrajGradSampling...")
    
    # Create configuration
    cfg = TrajGradSamplingCfg()
    cfg.trajectory_opt.horizon_samples = 50
    cfg.trajectory_opt.horizon_nodes = 10
    cfg.trajectory_opt.num_samples = 32
    cfg.trajectory_opt.num_diffuse_steps = 3
    cfg.trajectory_opt.num_diffuse_steps_init = 5
    
    # Create TrajGradSampling instance
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_envs = 4
    num_actions = 12
    dt = 0.02
    main_env_indices = torch.arange(num_envs, device=device)
    
    traj_grad = TrajGradSampling(
        cfg=cfg,
        device=device,
        num_envs=num_envs,
        num_actions=num_actions,
        dt=dt,
        main_env_indices=main_env_indices
    )
    
    # Create mock rollout callback
    rollout_callback = create_mock_rollout_callback()
    
    # Test profiled methods
    print("Running optimize_all_trajectories with profiling...")
    for i in range(3):
        traj_grad.optimize_all_trajectories(rollout_callback, n_diffuse=2, initial=(i == 0))
    
    print("Running node2u_batch conversions...")
    for i in range(5):
        _ = traj_grad.node2u_batch(traj_grad.node_trajectories)
    
    print("Profiling test completed!")


def test_profiling_modes():
    """Test different profiling modes."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT PROFILING MODES")
    print("="*60)
    
    # Test 1: Time profiling only
    print("\n1. Testing TIME PROFILING mode:")
    clear_profile_data()
    set_profiling_mode(time_prof=True)
    test_profiling_basic()
    print_time_profile_summary()
    
    # Test 2: GPU profiling only (if CUDA available)
    if torch.cuda.is_available():
        print("\n2. Testing GPU PROFILING mode:")
        clear_profile_data()
        set_profiling_mode(gpu_prof=True)
        test_profiling_basic()
        print_gpu_profile_summary()
    
    # Test 3: cProfile profiling
    print("\n3. Testing cPROFILE mode:")
    clear_profile_data()
    set_profiling_mode(cprofile=True)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    test_profiling_basic()
    
    # Print cProfile results
    profile_file = "results/optimize_all_trajectories.prof"
    if os.path.exists(profile_file):
        print_profile(profile_file)
    else:
        print(f"Profile file {profile_file} not found")
    
    # Test 4: All profiling modes
    print("\n4. Testing ALL PROFILING modes:")
    clear_profile_data()
    enable_profiling()
    test_profiling_basic()
    
    # Print all summaries
    print_time_profile_summary()
    if torch.cuda.is_available():
        print_gpu_profile_summary()
    
    # Save comprehensive summary
    save_profile_summary("results/profiling_summary.txt")
    
    # Disable profiling
    disable_profiling()


def test_environment_variables():
    """Test profiling using environment variables directly."""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT VARIABLE CONTROL")
    print("="*60)
    
    # Test setting environment variables manually
    print("\n1. Setting environment variables manually:")
    os.environ["TIME_PROFILING"] = "1"
    os.environ["GPU_PROFILING"] = "1"
    
    print("Environment variables set:")
    for var in ["PROFILING", "TIME_PROFILING", "GPU_PROFILING", "BENCHMARKING"]:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")
    
    clear_profile_data()
    test_profiling_basic()
    
    print("\nResults with manual environment variables:")
    print_time_profile_summary()
    if torch.cuda.is_available():
        print_gpu_profile_summary()
    
    # Clean up
    for var in ["TIME_PROFILING", "GPU_PROFILING"]:
        if var in os.environ:
            del os.environ[var]


def test_selective_profiling():
    """Test selective profiling of specific methods."""
    print("\n" + "="*60)
    print("TESTING SELECTIVE PROFILING")
    print("="*60)
    
    # Enable only time profiling
    set_profiling_mode(time_prof=True)
    clear_profile_data()
    
    # Create TrajGradSampling instance
    cfg = TrajGradSamplingCfg()
    cfg.trajectory_opt.horizon_samples = 100
    cfg.trajectory_opt.horizon_nodes = 20
    cfg.trajectory_opt.num_samples = 64
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_envs = 8
    num_actions = 12
    dt = 0.02
    main_env_indices = torch.arange(num_envs, device=device)
    
    traj_grad = TrajGradSampling(
        cfg=cfg,
        device=device,
        num_envs=num_envs,
        num_actions=num_actions,
        dt=dt,
        main_env_indices=main_env_indices
    )
    
    rollout_callback = create_mock_rollout_callback()
    
    print("Testing node2u_batch performance with different sizes...")
    for batch_size in [1, 4, 8, 16]:
        test_nodes = torch.randn(batch_size, cfg.trajectory_opt.horizon_nodes + 1, num_actions, device=device)
        # This will be profiled due to the decorator
        _ = traj_grad.spline_interpolator.node2dense(test_nodes)
    
    print("Testing optimization with different diffusion steps...")
    for n_diffuse in [1, 3, 5]:
        traj_grad.optimize_all_trajectories(rollout_callback, n_diffuse=n_diffuse)
    
    print("\nSelective profiling results:")
    print_time_profile_summary()
    
    # Save results
    save_profile_summary("results/selective_profiling_summary.txt")
    
    disable_profiling()


def main():
    """Main test function."""
    print("TrajGradSampling Profiling Test Suite")
    print("====================================")
    
    # Check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    try:
        # Test different profiling modes
        test_profiling_modes()
        
        # Test environment variable control
        test_environment_variables()
        
        # Test selective profiling
        test_selective_profiling()
        
        print("\n" + "="*60)
        print("ALL PROFILING TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        result_files = [
            "results/optimize_all_trajectories.prof",
            "results/profiling_summary.txt", 
            "results/selective_profiling_summary.txt"
        ]
        for file_path in result_files:
            if os.path.exists(file_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not found)")
        
        print("\nTo enable profiling in your code:")
        print("  1. Import: from traj_sampling.utils.benchmark import enable_profiling")
        print("  2. Call: enable_profiling() before running your code")
        print("  3. Or set environment variables manually:")
        print("     export PROFILING=1")
        print("     export TIME_PROFILING=1") 
        print("     export GPU_PROFILING=1")
        print("     export BENCHMARKING=1")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up environment
        disable_profiling()
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)