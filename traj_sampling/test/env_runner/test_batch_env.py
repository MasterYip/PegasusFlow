#!/usr/bin/env python3

"""
Simple test script for the ObsAvoid batch environment to verify the tensor dimension fix.
"""

import torch
import sys
import os

def test_basic_batch_env():
    """Test basic functionality of the batch environment."""
    print("Testing ObsAvoid Batch Environment...")

    # Import here to avoid path issues
    from traj_sampling.env_runner.obsavoid.obsavoid_batch_env import create_randpath_bound_batch_env

    # Create a small batch environment for testing
    env = create_randpath_bound_batch_env(
        num_main_envs=2,
        num_rollout_per_main=4,
        device="cpu"  # Use CPU to avoid CUDA issues
    )

    print(f"Created batch environment:")
    print(f"  Main environments: {env.num_main_envs}")
    print(f"  Rollout environments per main: {env.num_rollout_per_main}")
    print(f"  Total environments: {env.total_num_envs}")
    print(f"  Observation dimension: {env.obs_dim}")
    print(f"  Action dimension: {env.action_dim}")

    # Test reset
    print("\nTesting reset...")
    obs = env.reset()
    print(f"  Reset successful! Observation shape: {obs.shape}")

    # Test step
    print("\nTesting step...")
    actions = torch.randn((env.num_main_envs, env.action_dim)) * 0.1
    obs, rewards, dones, info = env.step(actions)
    print(f"  Step successful!")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Dones shape: {dones.shape}")

    # Test rollout functionality
    print("\nTesting rollout functionality...")
    env.cache_main_env_states()
    env.sync_main_to_rollout()

    # Create rollout actions for all rollout environments
    num_rollout_envs = env.num_rollout_per_main * env.num_main_envs
    rollout_actions = torch.randn((num_rollout_envs, env.action_dim)) * 0.1

    rollout_obs, rollout_rewards, rollout_dones, rollout_info = env.step_rollout(rollout_actions)
    print(f"  Rollout step successful!")
    print(f"  Rollout observations shape: {rollout_obs.shape}")
    print(f"  Rollout rewards shape: {rollout_rewards.shape}")

    env.restore_main_env_states()
    print(f"  State restoration successful!")

    print("\n✅ All tests passed! The batch environment is working correctly.")
    return True


if __name__ == "__main__":
    try:
        success = test_basic_batch_env()
        if success:
            print("\n🎉 Batch environment test completed successfully!")
        else:
            print("\n❌ Some tests failed.")
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
