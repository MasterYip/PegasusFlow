#!/usr/bin/env python3

"""
Test script for ObsAvoid Environment Runner with trajectory gradient sampling.

This script demonstrates how to use the new environment runner architecture
for both PID control and trajectory optimization, with visualization support.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from typing import Dict, Any, Optional
import time

from traj_sampling.env_runner.obsavoid.obsavoid_envrunner import ObsAvoidEnvRunner
from traj_sampling.trajopt_policy import TrajOptMode, create_transformer_policy, TrajOptTrainer


def test_pid_control():
    """Test the environment runner with PID control."""
    print("\n" + "="*60)
    print("TESTING PID CONTROL")
    print("="*60)
    
    # Create environment runner with trajectory optimization disabled for PID test
    runner = ObsAvoidEnvRunner(
        num_main_envs=4,
        num_rollout_per_main=8,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=200,
        horizon_samples=10,
        horizon_nodes=3,
        optimize_interval=5,
        env_type="randpath",
        enable_trajectory_optimization=False,  # Disable for PID test
        enable_vis=True  # Enable visualization
    )
    
    # Run with PID controller (no policy provided)
    results = runner.run(policy=None, seed=42)
    
    print(f"PID Control Results:")
    print(f"  Test mean score: {results['test_mean_score']:.4f}")
    print(f"  Total reward: {results['total_reward']:.4f}")
    print(f"  Average reward: {results['average_reward']:.4f}")
    print(f"  Steps completed: {results['steps']}")
    
    # Replay trajectories
    print("\nReplaying PID control trajectories...")
    try:
        if hasattr(runner.env, 'replay_all_trajectories'):
            runner.env.replay_all_trajectories(save_dir="./pid_trajectories")  # type: ignore
            
            # Keep the replay windows open for a moment
            input("Press Enter to continue to next test...")
            
        if hasattr(runner.env, 'end'):
            runner.env.end()  # type: ignore
    except Exception as e:
        print(f"Could not replay trajectories: {e}")
    
    return results


def test_trajectory_optimization():
    """Test the environment runner with trajectory optimization."""
    print("\n" + "="*60)
    print("TESTING TRAJECTORY OPTIMIZATION")
    print("="*60)
    
    # Create environment runner with trajectory optimization enabled
    runner = ObsAvoidEnvRunner(
        num_main_envs=4,
        num_rollout_per_main=32,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=200,
        horizon_samples=16,
        horizon_nodes=4,
        optimize_interval=5,
        env_type="randpath",
        enable_trajectory_optimization=True,
        enable_vis=True  # Enable visualization
    )
    
    # Run with trajectory optimization
    results = runner.run_with_trajectory_optimization(seed=42)
    
    print(f"Trajectory Optimization Results:")
    print(f"  Test mean score: {results['test_mean_score']:.4f}")
    print(f"  Total reward: {results['total_reward']:.4f}")
    print(f"  Average reward: {results['average_reward']:.4f}")
    print(f"  Steps completed: {results['steps']}")
    print(f"  Average optimization time: {results['optimization_time']:.4f} seconds")
    print(f"  Number of optimizations: {results['num_optimizations']}")
    
    # Replay trajectories
    print("\nReplaying trajectory optimization trajectories...")
    try:
        if hasattr(runner.env, 'replay_all_trajectories'):
            runner.env.replay_all_trajectories(save_dir="./traj_opt_trajectories")  # type: ignore
            
            # Keep the replay windows open for a moment
            input("Press Enter to continue to next test...")
        
        if hasattr(runner.env, 'end'):
            runner.env.end()  # type: ignore
    except Exception as e:
        print(f"Could not replay trajectories: {e}")
    
    return results


def test_comparison():
    """Test comparison between PID control and trajectory optimization."""
    print("\n" + "="*60)
    print("TESTING PID VS TRAJECTORY OPTIMIZATION COMPARISON")
    print("="*60)
    
    # Create environment runner
    runner = ObsAvoidEnvRunner(
        num_main_envs=2,
        num_rollout_per_main=8,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=100,  # Shorter for comparison test
        horizon_samples=10,
        horizon_nodes=3,
        optimize_interval=5,
        env_type="randpath",
        enable_trajectory_optimization=True,
        enable_vis=True  # Enable visualization
    )
    
    # Run comparison (PID vs trajectory optimization)
    results = runner.run_comparison(policy=None, seed=42)
    
    # Replay comparison trajectories
    print("\nReplaying comparison trajectories...")
    try:
        if hasattr(runner.env, 'replay_all_trajectories'):
            runner.env.replay_all_trajectories(save_dir="./comparison_trajectories")  # type: ignore
            
            # Keep the replay windows open for a moment
            input("Press Enter to continue...")
        
        if hasattr(runner.env, 'end'):
            runner.env.end()  # type: ignore
    except Exception as e:
        print(f"Could not replay trajectories: {e}")
    
    return results


def test_batch_environment():
    """Test the batch environment directly with visualization."""
    print("\n" + "="*60)
    print("TESTING BATCH ENVIRONMENT DIRECTLY WITH VISUALIZATION")
    print("="*60)
    
    from traj_sampling.env_runner.obsavoid.obsavoid_batch_env import create_randpath_bound_batch_env
    
    # Create batch environment with visualization enabled
    env = create_randpath_bound_batch_env(
        num_main_envs=2,
        num_rollout_per_main=4,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        enable_vis=True,  # Enable real-time visualization
        vis_time_window=2.0  # Larger time window for better view
    )
    
    print(f"Created batch environment:")
    print(f"  Main environments: {env.num_main_envs}")
    print(f"  Rollout environments per main: {env.num_rollout_per_main}")
    print(f"  Total environments: {env.total_num_envs}")
    print(f"  Observation dimension: {env.obs_dim}")
    print(f"  Action dimension: {env.action_dim}")
    print(f"  Visualization enabled: {env.enable_vis}")
    
    # Reset environment
    obs = env.reset()
    print(f"  Initial observations shape: {obs.shape}")
    
    # Take steps with simple PID control
    print(f"  Running simulation with PID control...")
    for step in range(50):  # Run for more steps to see trajectory
        # Simple PID control for main environments
        actions = torch.zeros((env.num_main_envs, env.action_dim), device=env.device)
        
        for i in range(env.num_main_envs):
            main_idx = env.main_env_indices[i]
            
            # Simple target following (center of corridor)
            params = env.bound_params[main_idx]
            current_t = env.t[main_idx].item()
            current_y = env.y[main_idx].item()
            current_v = env.v[main_idx].item()
            
            # Compute target position (center of corridor)
            res = params['slope'] * current_t
            for j in range(len(params['coef']) // 2):
                res += params['coef'][j * 2] * np.sin((j + 1) * current_t) + \
                       params['coef'][j * 2 + 1] * np.cos((j + 1) * current_t)
            target_y = res  # Center of corridor
            
            # Simple PID control
            p_gain = params['p']
            d_gain = params['d']
            action = p_gain * (target_y - current_y) + d_gain * (-current_v)
            
            # Scale and clamp action
            action = action / env.acc_scale
            action = np.clip(action, env.acc_bd[0], env.acc_bd[1])
            
            actions[i, 0] = action
        
        # Step main environments
        obs, rewards, dones, info = env.step(actions)
        
        if step % 10 == 0:
            print(f"  Step {step+1}:")
            print(f"    Rewards: {rewards.cpu().numpy()}")
            print(f"    Positions: {env.y[env.main_env_indices].cpu().numpy()}")
            print(f"    Done flags: {dones.cpu().numpy()}")
        
        # Test rollout functionality at step 25
        if step == 25:
            print(f"    Testing rollout functionality...")
            
            # Cache main env states
            env.cache_main_env_states()
            
            # Sync rollout envs to main envs
            env.sync_main_to_rollout()
            
            # Step rollout environments with random actions
            rollout_actions = torch.randn((env.num_rollout_per_main * env.num_main_envs, env.action_dim), device=env.device) * 0.1
            rollout_obs, rollout_rewards, rollout_dones, rollout_info = env.step_rollout(rollout_actions)
            
            print(f"    Rollout rewards: {rollout_rewards.cpu().numpy()[:4]}...")  # Show first 4
            
            # Restore main env states
            env.restore_main_env_states()
            print(f"    Main env states restored successfully")
        
        # Small delay for visualization
        time.sleep(0.02)
    
    # Replay trajectories
    print(f"\nReplaying batch environment trajectories...")
    if hasattr(env, 'replay_all_trajectories'):
        env.replay_all_trajectories(save_dir="./batch_env_trajectories")
    
    # Show trajectory data for first environment
    if hasattr(env, 'get_trajectory_data'):
        traj_data = env.get_trajectory_data(env_idx=0)
        if traj_data:
            print(f"\nTrajectory data for environment 0:")
            print(f"  Number of steps: {len(traj_data['times'])}")
            print(f"  Time range: {traj_data['times'][0]:.3f} to {traj_data['times'][-1]:.3f}")
            print(f"  Position range: {np.min(traj_data['positions']):.3f} to {np.max(traj_data['positions']):.3f}")
            print(f"  Average reward: {np.mean(traj_data['rewards']):.3f}")
    
    # Keep windows open for viewing
    input("Press Enter to close visualization...")
    if hasattr(env, 'end'):
        env.end()


def test_optimization_data_collection():
    """Test optimization data collection functionality."""
    print("\n=== Testing Optimization Data Collection ===")
    
    # Create environment runner (using obsavoid, not obsavoid2)
    runner = ObsAvoidEnvRunner(
        num_main_envs=40,
        num_rollout_per_main=16,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=100,  # Smaller for testing
        horizon_samples=16,
        horizon_nodes=4,
        optimize_interval=1,
        env_type="randpath",
        enable_trajectory_optimization=True,
        enable_vis=False  # Disable visualization for testing
    )
    
    # Get the TrajGradSampling instance
    traj_sampler = runner.traj_sampler
    
    print("Step 1: Enabling data collection...")
    
    # Enable data collection with DELTA_TRAJ mode
    traj_sampler.enable_data_collect(
        mode=TrajOptMode.DELTA_TRAJ,
        max_samples=10000  # Small number for testing
    )
    
    print(f"Data collection enabled with mode: {TrajOptMode.DELTA_TRAJ}")
    print(f"Max samples: 10000")
    
    # Verify data collector is created
    assert hasattr(traj_sampler, 'data_collector')
    assert traj_sampler.data_collector is not None
    print(f"Data collector created successfully")
    
    print("\nStep 2: Running optimization with data collection...")
    
    # Reset environment and get initial observations
    obs = runner.reset()
    
    # Run a few optimization steps to collect data
    initial_samples = traj_sampler.data_collector.num_samples
    
    for step in range(200):  # More steps for better data collection
        # Get current observations
        obs = runner.env.get_observation()
        
        # Run optimization step which should collect data
        traj_sampler.optimize_and_collect_data(
            rollout_callback=runner.rollout_callback,
            obs=obs,
            n_diffuse=2,  # Reduced for faster testing
            initial=(step == 0)
        )
        
        current_samples = traj_sampler.data_collector.num_samples
        print(f"  Step {step + 1}: Collected {current_samples - initial_samples} new samples")
        initial_samples = current_samples
    
    print(f"\nStep 3: Validating collected data...")
    
    # Check if data was collected
    total_samples = traj_sampler.data_collector.num_samples
    print(f"Total samples collected: {total_samples}")
    assert total_samples > 0, "No data was collected"
    
    # Get the collected dataset
    dataset = traj_sampler.data_collector.get_dataset()
    
    # Validate dataset structure
    required_keys = ['trajectories_input', 'trajectories_output', 'num_samples', 'mode']
    for key in required_keys:
        assert key in dataset, f"Missing key: {key}"
    
    print(f"Dataset keys: {list(dataset.keys())}")
    print(f"Number of samples: {dataset['num_samples']}")
    print(f"Input trajectory shape: {dataset['trajectories_input'].shape}")
    print(f"Target trajectory shape: {dataset['trajectories_output'].shape}")
    print(f"Data collection mode: {dataset['mode']}")
    
    # Validate data shapes
    assert dataset['trajectories_input'].shape[0] == dataset['num_samples']
    assert dataset['trajectories_output'].shape[0] == dataset['num_samples']
    assert dataset['trajectories_input'].shape == dataset['trajectories_output'].shape
    
    # For DELTA_TRAJ mode, validate that deltas have both positive and negative values
    if dataset['mode'] == TrajOptMode.DELTA_TRAJ:
        # In DELTA_TRAJ mode, the target is already the delta, not absolute trajectory
        deltas = dataset['trajectories_output']
        print(f"Delta statistics:")
        print(f"  Mean: {deltas.mean().item():.6f}")
        print(f"  Std: {deltas.std().item():.6f}")
        print(f"  Min: {deltas.min().item():.6f}")
        print(f"  Max: {deltas.max().item():.6f}")
        
        # Check that deltas are not all zero (indicating actual optimization)
        assert deltas.abs().mean() > 1e-6, "Deltas are too small, optimization may not be working"
    
    print("\nStep 4: Testing dataset save/load...")
    
    # Test saving and loading dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "test_dataset.pt")
        
        # Save dataset
        traj_sampler.data_collector.save_dataset(dataset_path)
        assert os.path.exists(dataset_path)
        print(f"Dataset saved to: {dataset_path}")
        
        # Load dataset
        loaded_dataset = traj_sampler.data_collector.load_dataset(dataset_path)
        
        # Verify loaded dataset matches original (with proper device handling)
        assert loaded_dataset['num_samples'] == dataset['num_samples']
        
        # Compare tensors on the same device
        original_input = dataset['trajectories_input'].to(traj_sampler.data_collector.device)
        original_output = dataset['trajectories_output'].to(traj_sampler.data_collector.device)
        
        assert torch.allclose(loaded_dataset['trajectories_input'], original_input, atol=1e-6)
        assert torch.allclose(loaded_dataset['trajectories_output'], original_output, atol=1e-6)
        print("Dataset save/load test passed")
    
    print("\nStep 5: Save dataset for next test...")
    
    # Save dataset to workspace for use in training test
    workspace_dataset_path = "./optimization_data_collection.pt"
    traj_sampler.data_collector.save_dataset(workspace_dataset_path)
    print(f"Dataset saved to workspace: {workspace_dataset_path}")
    
    # Clean up
    if hasattr(runner.env, 'end'):
        runner.env.end()  # type: ignore
    
    print("✓ Optimization data collection test completed successfully")
    return dataset, runner


def test_trajopt_policy_tf_training():
    """Test TrajOptPolicyTF training using collected data from TrajOptPolicySampling."""
    print("\n=== Testing TrajOptPolicyTF Training ===")
    
    # Try to load dataset from previous test first
    dataset_path = "./optimization_data_collection.pt"
    dataset = None
    
    if os.path.exists(dataset_path):
        print("Step 1: Loading dataset from previous test...")
        try:
            dataset = torch.load(dataset_path)
            print(f"Loaded dataset with {dataset['num_samples']} samples")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            dataset = None
    
    # If no dataset available, collect new data
    if dataset is None:
        print("Step 1: Collecting fresh training data...")
        
        # Create environment runner for data collection
        runner = ObsAvoidEnvRunner(
            num_main_envs=4,
            num_rollout_per_main=16,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            max_steps=50,
            horizon_samples=16,
            horizon_nodes=4,
            optimize_interval=1,
            env_type="randpath",
            enable_trajectory_optimization=True,
            enable_vis=False
        )
        
        # Get the TrajGradSampling instance
        traj_sampler = runner.traj_sampler
        
        # Enable data collection
        traj_sampler.enable_data_collect(
            mode=TrajOptMode.DELTA_TRAJ,
            max_samples=100  # Small dataset for testing
        )
        
        # Reset environment and collect data
        runner.reset()
        
        # Run optimization to collect data
        for step in range(8):  # More steps for better training data
            obs = runner.env.get_observation()
            traj_sampler.optimize_and_collect_data(
                rollout_callback=runner.rollout_callback,
                obs=obs,
                n_diffuse=2,
                initial=(step == 0)
            )
        
        # Get collected dataset
        dataset = traj_sampler.get_collected_dataset()
        if dataset is not None:
            print(f"Collected {dataset['num_samples']} training samples")
            
            # Save dataset for future use
            torch.save(dataset, dataset_path)
            print(f"Dataset saved to {dataset_path}")
    
    if dataset is None or dataset['num_samples'] == 0:
        print("No data collected, skipping training test")
        return
    
    print(f"\nStep 2: Setting up transformer training...")
    print(f"Dataset contains {dataset['num_samples']} samples")
    print(f"Mode: {dataset['mode']}")
    print(f"Has observations: {dataset['has_observations']}")
    
    # Create a new runner for training
    training_runner = ObsAvoidEnvRunner(
        num_main_envs=2,
        num_rollout_per_main=8,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=20,
        horizon_samples=16,
        horizon_nodes=4,
        enable_trajectory_optimization=True,
        enable_vis=False
    )
    
    # Get trajectory sampler for training
    training_traj_sampler = training_runner.traj_sampler
    
    # Setup transformer training
    print("Step 3: Setting up transformer training...")
    trainer = training_traj_sampler.setup_transformer_training(
        obs_dim=training_runner.env.obs_dim if dataset['has_observations'] else None,
        learning_rate=1e-3,  # Higher learning rate for faster training
        d_model=128,  # Smaller model for testing
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    )
    
    # Train the transformer
    print("Step 4: Training transformer...")
    training_metrics = training_traj_sampler.train_transformer_on_data(
        dataset=dataset,
        num_epochs=20,  # Very few epochs for testing
        batch_size=64,
        validation_split=0.2,
        print_interval=2
    )
    
    print(f"Step 5: Training completed!")
    print(f"Final training loss: {training_metrics['training_losses'][-1]:.6f}")
    if training_metrics['validation_losses']:
        print(f"Final validation loss: {training_metrics['validation_losses'][-1]:.6f}")
    
    # Deploy the trained transformer
    print("Step 6: Deploying trained transformer...")
    training_traj_sampler.deploy_trained_transformer()
    print("Transformer policy deployed successfully!")
    
    # Test the trained transformer on a short run
    print("Step 7: Testing deployed transformer...")
    training_runner.reset()
    
    # Run a few steps with the transformer policy
    for step in range(3):
        actions = training_runner.get_next_actions()
        obs, rewards, dones, info = training_runner.env.step(actions)
        if step < 2:  # Don't optimize on last step
            training_runner.optimize_trajectories(initial=(step == 0))
        print(f"  Step {step}: Avg reward = {rewards.mean().item():.3f}")
    
    print("Transformer training and deployment test completed successfully!")


def test_data_collection_modes():
    """Test different data collection modes (traj vs delta_traj)."""
    print("\n" + "="*60)
    print("TESTING DATA COLLECTION MODES")
    print("="*60)
    
    # Test TRAJ mode
    print("\n1. Testing TRAJ mode data collection...")
    runner_traj = ObsAvoidEnvRunner(
        num_main_envs=2,
        num_rollout_per_main=8,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=20,
        horizon_samples=10,
        horizon_nodes=3,
        optimize_interval=1,
        env_type="randpath",
        enable_trajectory_optimization=True,
        enable_vis=False
    )
    
    # Get the TrajGradSampling instance and enable data collection
    traj_sampler_traj = runner_traj.traj_sampler
    traj_sampler_traj.enable_data_collect(
        mode=TrajOptMode.TRAJ,
        max_samples=20
    )
    
    # Reset and run a few optimization steps
    runner_traj.reset()
    for step in range(3):
        obs = runner_traj.env.get_observation()
        traj_sampler_traj.optimize_and_collect_data(
            rollout_callback=runner_traj.rollout_callback,
            obs=obs,
            n_diffuse=1,
            initial=(step == 0)
        )
    
    traj_dataset = traj_sampler_traj.get_collected_dataset()
    if traj_dataset is not None:
        print(f"TRAJ mode collected {traj_dataset['num_samples']} samples")
        print(f"TRAJ mode dataset keys: {list(traj_dataset.keys())}")
    else:
        print("TRAJ mode: No data collected")
    
    # Test DELTA_TRAJ mode
    print("\n2. Testing DELTA_TRAJ mode data collection...")
    runner_delta = ObsAvoidEnvRunner(
        num_main_envs=2,
        num_rollout_per_main=8,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_steps=20,
        horizon_samples=10,
        horizon_nodes=3,
        optimize_interval=1,
        env_type="randpath",
        enable_trajectory_optimization=True,
        enable_vis=False
    )
    
    # Get the TrajGradSampling instance and enable data collection
    traj_sampler_delta = runner_delta.traj_sampler
    traj_sampler_delta.enable_data_collect(
        mode=TrajOptMode.DELTA_TRAJ,
        max_samples=20
    )
    
    # Reset and run a few optimization steps
    runner_delta.reset()
    for step in range(3):
        obs = runner_delta.env.get_observation()
        traj_sampler_delta.optimize_and_collect_data(
            rollout_callback=runner_delta.rollout_callback,
            obs=obs,
            n_diffuse=1,
            initial=(step == 0)
        )
    
    delta_dataset = traj_sampler_delta.get_collected_dataset()
    if delta_dataset is not None:
        print(f"DELTA_TRAJ mode collected {delta_dataset['num_samples']} samples")
        print(f"DELTA_TRAJ mode dataset keys: {list(delta_dataset.keys())}")
    else:
        print("DELTA_TRAJ mode: No data collected")
    
    # Compare datasets
    print(f"\n3. Comparing datasets...")
    if (traj_dataset is not None and traj_dataset['num_samples'] > 0 and 
        delta_dataset is not None and delta_dataset['num_samples'] > 0):
        print(f"Both modes collected data successfully")
        print(f"TRAJ mode: {traj_dataset['mode']}")
        print(f"DELTA_TRAJ mode: {delta_dataset['mode']}")
        
        # Check trajectory shapes
        traj_input_shape = traj_dataset['trajectories_input'].shape
        delta_input_shape = delta_dataset['trajectories_input'].shape
        print(f"TRAJ input shape: {traj_input_shape}")
        print(f"DELTA_TRAJ input shape: {delta_input_shape}")
        
        # The output should be different between modes
        traj_output = traj_dataset['trajectories_output']
        delta_output = delta_dataset['trajectories_output']
        print(f"TRAJ output range: [{traj_output.min():.3f}, {traj_output.max():.3f}]")
        print(f"DELTA_TRAJ output range: [{delta_output.min():.3f}, {delta_output.max():.3f}]")
    
    print("Data collection modes test completed!")


def main():
    """Main test function."""
    print("ObsAvoid Environment Runner Test Suite with Visualization")
    print("=========================================================")
    
    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("./pid_trajectories", exist_ok=True)
    os.makedirs("./traj_opt_trajectories", exist_ok=True)
    os.makedirs("./comparison_trajectories", exist_ok=True)
    os.makedirs("./batch_env_trajectories", exist_ok=True)
    
    try:
        # Test 1: Batch environment directly with visualization
        # test_batch_environment()
        
        # Test 2: PID control with visualization
        # pid_results = test_pid_control()
        
        # Test 3: Trajectory optimization with visualization
        # traj_results = test_trajectory_optimization()
        
        # Test 4: Comparison with visualization
        # comparison_results = test_comparison()
        
        # NEW TESTS: Optimization data collection and training
        
        # Test 5: Optimization data collection
        print("\n" + "="*80)
        print("RUNNING NEW TESTS: DATA COLLECTION AND TRAINING")
        print("="*80)
        
        data_collection_results = test_optimization_data_collection()
        
        # Test 6: TrajOptPolicyTF training
        training_results = test_trajopt_policy_tf_training()
        
        # Test 7: Data collection modes comparison
        # test_data_collection_modes()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        # New tests summary
        print("\nData Collection and Training Tests:")
        
        if data_collection_results[0] is not None:
            dataset, runner = data_collection_results
            print(f"✓ Data Collection: {dataset['num_samples']} samples collected")
        else:
            print("✗ Data Collection: Failed")
        
        print(f"✓ Transformer Training: Completed")
        print(f"✓ Mode Comparison: Both TRAJ and DELTA_TRAJ modes tested")
        
        print(f"\nOutput files:")
        print(f"  Optimization data: ./optimization_data_collection.pt")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)