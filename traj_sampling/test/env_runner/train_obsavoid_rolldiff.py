#!/usr/bin/env python3

"""
Training script for TrajOptPolicyTF using trajectory optimization data.

This script implements a complete pipeline for:
1. Collecting trajectory optimization data from sampling-based methods
2. Training a transformer policy (TrajOptPolicyTF) on the collected data
3. Testing the trained transformer policy in the obsavoid environment
4. Comparing performance between sampling and transformer policies
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from typing import Dict, Any, Optional
from pathlib import Path

from traj_sampling.env_runner.obsavoid.obsavoid_envrunner import ObsAvoidEnvRunner
from traj_sampling.trajopt_policy import TrajOptMode


def collect_training_data(
    num_episodes: int = 20,
    steps_per_episode: int = 200,
    num_envs: int = 8,
    horizon_samples: int = 32,
    horizon_nodes: int = 8,
    env_type: str = "randpath",
    device: str = "cuda:0",
    data_mode: str = "delta_traj",
    max_samples: int = 5000,
    save_path: str = "./rolldiff_training_data.pt"
) -> Dict[str, Any]:
    """Collect training data from trajectory optimization for transformer training.
    
    Args:
        num_episodes: Number of data collection episodes
        steps_per_episode: Steps per episode
        num_envs: Number of main environments
        horizon_samples: Trajectory horizon samples
        horizon_nodes: Trajectory horizon nodes
        env_type: Environment type ("randpath", "sine")
        device: Device for computations
        data_mode: Data collection mode ("traj" or "delta_traj")
        max_samples: Maximum number of samples to collect
        save_path: Path to save collected data
        
    Returns:
        Dictionary containing collection results and dataset
    """
    print("="*80)
    print("TRAJECTORY OPTIMIZATION DATA COLLECTION")
    print("="*80)
    
    # Convert string mode to enum
    mode_enum = TrajOptMode.TRAJ if data_mode == "traj" else TrajOptMode.DELTA_TRAJ
    
    # Create environment runner for data collection
    runner = ObsAvoidEnvRunner(
        num_main_envs=num_envs,
        num_rollout_per_main=32,  # More rollouts for better optimization
        device=device,
        max_steps=steps_per_episode,
        horizon_samples=horizon_samples,
        horizon_nodes=horizon_nodes,
        optimize_interval=1,  # Optimize every step for maximum data
        env_type=env_type,
        enable_trajectory_optimization=True,
        enable_vis=False  # Disable visualization for faster data collection
    )
    
    # Get the TrajGradSampling instance and enable data collection
    traj_sampler = runner.traj_sampler
    
    print(f"Setting up data collection:")
    print(f"  Mode: {data_mode}")
    print(f"  Max samples: {max_samples}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    
    # Enable data collection
    traj_sampler.enable_data_collect(
        mode=mode_enum,
        max_samples=max_samples
    )
    
    # Data collection loop
    episode_rewards = []
    total_samples_collected = 0
    
    print(f"\nStarting data collection...")
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        episode_seed = 1000 + episode * 42  # Deterministic but varied seeds
        
        print(f"\nEpisode {episode + 1}/{num_episodes} (seed: {episode_seed})")
        
        # Reset environment
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        obs = runner.reset()
        
        episode_total_reward = 0.0
        samples_at_start = traj_sampler.data_collector.num_samples
        
        # Run episode with data collection
        for step in range(steps_per_episode):
            # Get current observations for data collection
            obs = runner.env.get_observation()
            
            # Optimize trajectories and collect data
            traj_sampler.optimize_and_collect_data(
                rollout_callback=runner.rollout_callback,
                obs=obs,
                n_diffuse=3,  # Moderate optimization for speed vs quality
                initial=(step == 0)
            )
            
            # Get next actions and step environment
            actions = runner.get_next_actions()
            obs, rewards, dones, info = runner.env.step(actions)
            
            # Track episode metrics
            episode_total_reward += rewards.mean().item()
            
            # Shift trajectories for next step
            if traj_sampler is not None:
                traj_sampler.shift_trajectory_batch()
            
            # Print progress
            if step % 1 == 0:
                current_samples = traj_sampler.data_collector.num_samples
                print(f"  Step {step}: Samples collected: {current_samples}, Avg reward: {rewards.mean().item():.3f}")
            
            # Check if we've collected enough data
            if traj_sampler.data_collector.num_samples >= max_samples:
                print(f"  Reached maximum samples ({max_samples}), stopping data collection")
                break
        
        # Episode summary
        episode_time = time.time() - episode_start_time
        samples_this_episode = traj_sampler.data_collector.num_samples - samples_at_start
        episode_rewards.append(episode_total_reward / steps_per_episode)
        total_samples_collected = traj_sampler.data_collector.num_samples
        
        print(f"  Episode completed in {episode_time:.2f}s")
        print(f"  Samples this episode: {samples_this_episode}")
        print(f"  Total samples: {total_samples_collected}")
        print(f"  Average reward: {episode_rewards[-1]:.3f}")
        
        # Check if we have enough data
        if total_samples_collected >= max_samples:
            print(f"\nData collection complete! Collected {total_samples_collected} samples")
            break
    
    # Get final dataset
    dataset = traj_sampler.get_collected_dataset()
    
    # Save dataset
    traj_sampler.save_collected_dataset(save_path)
    
    # Collection summary
    print(f"\nData Collection Summary:")
    print(f"  Episodes completed: {len(episode_rewards)}")
    print(f"  Total samples collected: {total_samples_collected}")
    print(f"  Average episode reward: {np.mean(episode_rewards):.3f}")
    print(f"  Data collection mode: {dataset['mode']}")
    print(f"  Dataset saved to: {save_path}")
    
    # Clean up
    if hasattr(runner.env, 'end'):
        runner.env.end()  # type: ignore
    
    return {
        'dataset': dataset,
        'num_samples': total_samples_collected,
        'episode_rewards': episode_rewards,
        'collection_episodes': len(episode_rewards),
        'save_path': save_path
    }


def train_transformer_policy(
    dataset_path: str = "./rolldiff_training_data.pt",
    num_epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    validation_split: float = 0.2,
    device: str = "cuda:0",
    model_config: Optional[Dict[str, Any]] = None,
    checkpoint_dir: str = "./rolldiff_checkpoints"
) -> Dict[str, Any]:
    """Train a transformer policy on collected trajectory optimization data.
    
    Args:
        dataset_path: Path to the collected dataset
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        validation_split: Fraction of data for validation
        device: Device for training
        model_config: Optional model configuration
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Dictionary containing training results
    """
    print("="*80)
    print("TRANSFORMER POLICY TRAINING")
    print("="*80)
    
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = torch.load(dataset_path)
    
    print(f"Dataset loaded:")
    print(f"  Samples: {dataset['num_samples']}")
    print(f"  Mode: {dataset['mode']}")
    print(f"  Has observations: {dataset['has_observations']}")
    print(f"  Input shape: {dataset['trajectories_input'].shape}")
    print(f"  Output shape: {dataset['trajectories_output'].shape}")
    
    # Create a temporary runner for training setup
    # Extract dimensions from dataset
    _, horizon_nodes, action_dim = dataset['trajectories_input'].shape
    obs_dim = dataset['observations_input'].shape[-1] if dataset['has_observations'] else None
    
    print(f"Model dimensions:")
    print(f"  Horizon nodes: {horizon_nodes}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Observation dimension: {obs_dim}")
    
    # Create environment runner for transformer setup
    training_runner = ObsAvoidEnvRunner(
        num_main_envs=4,  # Small for training setup
        num_rollout_per_main=16,
        device=device,
        max_steps=50,
        horizon_samples=32,
        horizon_nodes=horizon_nodes - 1,  # Adjust for the +1 in the data
        enable_trajectory_optimization=True,
        enable_vis=False
    )
    
    # Get trajectory sampler for training
    training_traj_sampler = training_runner.traj_sampler
    
    # Setup default model configuration
    default_model_config = {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1
    }
    
    if model_config is not None:
        default_model_config.update(model_config)
    
    print(f"Transformer configuration:")
    for key, value in default_model_config.items():
        print(f"  {key}: {value}")
    
    # Setup transformer training
    print(f"\nSetting up transformer training...")
    trainer = training_traj_sampler.setup_transformer_training(
        obs_dim=obs_dim,
        learning_rate=learning_rate,
        **default_model_config
    )
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train the transformer
    print(f"\nStarting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Validation split: {validation_split}")
    
    training_start_time = time.time()
    
    training_metrics = training_traj_sampler.train_transformer_on_data(
        dataset=dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        print_interval=1
    )
    
    training_time = time.time() - training_start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds!")
    print(f"Final training loss: {training_metrics['training_losses'][-1]:.6f}")
    if training_metrics['validation_losses']:
        print(f"Final validation loss: {training_metrics['validation_losses'][-1]:.6f}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, f"transformer_policy_final.pt")
    metadata = {
        'dataset_path': dataset_path,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'model_config': default_model_config,
        'training_time': training_time,
        'final_train_loss': training_metrics['training_losses'][-1],
        'final_val_loss': training_metrics['validation_losses'][-1] if training_metrics['validation_losses'] else None
    }
    
    training_traj_sampler.save_training_checkpoint(final_checkpoint_path, num_epochs, metadata)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_metrics['training_losses'], label='Training Loss')
    if training_metrics['validation_losses']:
        plt.plot(training_metrics['validation_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_metrics['training_losses'][-50:], label='Training Loss (last 50)')
    if training_metrics['validation_losses']:
        plt.plot(training_metrics['validation_losses'][-50:], label='Validation Loss (last 50)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress (Final Phase)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    
    return {
        'training_metrics': training_metrics,
        'training_time': training_time,
        'checkpoint_path': final_checkpoint_path,
        'model_config': default_model_config,
        'trainer': trainer,
        'traj_sampler': training_traj_sampler
    }


def test_trained_policy(
    checkpoint_path: str = "./rolldiff_checkpoints/transformer_policy_final.pt",
    num_test_episodes: int = 5,
    steps_per_episode: int = 300,
    device: str = "cuda:0",
    env_type: str = "randpath",
    enable_vis: bool = True,
    comparison_mode: bool = True
) -> Dict[str, Any]:
    """Test the trained transformer policy in the obsavoid environment.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        num_test_episodes: Number of test episodes
        steps_per_episode: Steps per test episode
        device: Device for testing
        env_type: Environment type for testing
        enable_vis: Whether to enable visualization
        comparison_mode: Whether to compare with sampling policy
        
    Returns:
        Dictionary containing test results
    """
    print("="*80)
    print("TRANSFORMER POLICY TESTING")
    print("="*80)
    
    # Load checkpoint to get model configuration
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    metadata = checkpoint.get('metadata', {})
    
    print(f"Loading trained policy from: {checkpoint_path}")
    print(f"Training metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Create environment runner for testing
    test_runner = ObsAvoidEnvRunner(
        num_main_envs=20 if comparison_mode else 2,
        num_rollout_per_main=32,
        device=device,
        max_steps=steps_per_episode,
        horizon_samples=32,
        horizon_nodes=8,
        env_type=env_type,
        enable_trajectory_optimization=True,
        enable_vis=enable_vis
    )
    
    # Setup transformer and load checkpoint
    traj_sampler = test_runner.traj_sampler
    
    # Load the model configuration from metadata
    model_config = metadata.get('model_config', {})
    
    trainer = traj_sampler.setup_transformer_training(
        obs_dim=32,  # Will be inferred if observations are used
        **model_config
    )
    
    # Load the trained model
    checkpoint_metadata = traj_sampler.load_training_checkpoint(checkpoint_path)
    
    # Deploy the trained transformer
    traj_sampler.deploy_trained_transformer()
    
    print(f"Transformer policy deployed successfully!")
    
    # Test results storage
    transformer_results = []
    sampling_results = [] if comparison_mode else None
    
    print(f"\nRunning test episodes...")
    
    for episode in range(num_test_episodes):
        episode_seed = 2000 + episode * 37
        
        print(f"\nTest Episode {episode + 1}/{num_test_episodes} (seed: {episode_seed})")
        
        if comparison_mode:
            # Test 1: Transformer policy
            print("  Testing transformer policy...")
            test_runner.reset()
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            
            # The transformer is already deployed, so run optimization normally
            transformer_episode_results = test_runner.run_with_trajectory_optimization(seed=episode_seed)
            transformer_results.append(transformer_episode_results)
            
            print(f"    Transformer score: {transformer_episode_results['test_mean_score']:.4f}")
            
            # Test 2: Switch back to sampling policy for comparison
            print("  Testing sampling policy...")
            test_runner.traj_sampler.switch_to_sampling_policy()
            
            test_runner.reset()
            np.random.seed(episode_seed)  # Same seed for fair comparison
            torch.manual_seed(episode_seed)
            
            sampling_episode_results = test_runner.run_with_trajectory_optimization(seed=episode_seed)
            sampling_results.append(sampling_episode_results)
            
            print(f"    Sampling score: {sampling_episode_results['test_mean_score']:.4f}")
            
            # Switch back to transformer for next episode
            test_runner.traj_sampler.deploy_trained_transformer()
            
        else:
            # Test only transformer policy
            test_runner.reset()
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            
            transformer_episode_results = test_runner.run_with_trajectory_optimization(seed=episode_seed)
            transformer_results.append(transformer_episode_results)
            
            print(f"    Transformer score: {transformer_episode_results['test_mean_score']:.4f}")
    
    # Compute summary statistics
    transformer_scores = [result['test_mean_score'] for result in transformer_results]
    transformer_mean = np.mean(transformer_scores)
    transformer_std = np.std(transformer_scores)
    
    print(f"\nTransformer Policy Results:")
    print(f"  Mean score: {transformer_mean:.4f} ± {transformer_std:.4f}")
    print(f"  Episodes: {transformer_scores}")
    
    results = {
        'transformer_results': transformer_results,
        'transformer_scores': transformer_scores,
        'transformer_mean': transformer_mean,
        'transformer_std': transformer_std
    }
    
    if comparison_mode and sampling_results:
        sampling_scores = [result['test_mean_score'] for result in sampling_results]
        sampling_mean = np.mean(sampling_scores)
        sampling_std = np.std(sampling_scores)
        
        print(f"\nSampling Policy Results:")
        print(f"  Mean score: {sampling_mean:.4f} ± {sampling_std:.4f}")
        print(f"  Episodes: {sampling_scores}")
        
        # Comparison
        improvement = transformer_mean - sampling_mean
        improvement_pct = (improvement / abs(sampling_mean)) * 100 if sampling_mean != 0 else 0
        
        print(f"\nComparison:")
        print(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
        print(f"  Transformer {'better' if improvement > 0 else 'worse'} than sampling")
        
        results.update({
            'sampling_results': sampling_results,
            'sampling_scores': sampling_scores,
            'sampling_mean': sampling_mean,
            'sampling_std': sampling_std,
            'improvement': improvement,
            'improvement_percentage': improvement_pct
        })
    
    # Visualization and trajectory replay
    if enable_vis:
        print(f"\nGenerating trajectory visualizations...")
        try:
            if hasattr(test_runner.env, 'replay_all_trajectories'):
                save_dir = "./transformer_test_trajectories"
                os.makedirs(save_dir, exist_ok=True)
                test_runner.env.replay_all_trajectories(save_dir=save_dir)  # type: ignore
                print(f"Trajectory replays saved to: {save_dir}")
                
                # Keep visualization open briefly
                input("Press Enter to continue...")
                
            if hasattr(test_runner.env, 'end'):
                test_runner.env.end()  # type: ignore
        except Exception as e:
            print(f"Visualization error: {e}")
    
    return results


def main():
    """Main training and testing pipeline."""
    parser = argparse.ArgumentParser(description="Train and test TrajOptPolicyTF on obsavoid environment")
    
    # Data collection arguments
    parser.add_argument('--collect-data', action='store_true', help='Collect training data')
    parser.add_argument('--data-episodes', type=int, default=100, help='Number of data collection episodes')
    parser.add_argument('--data-steps', type=int, default=30, help='Steps per data collection episode')
    parser.add_argument('--max-samples', type=int, default=400000, help='Maximum samples to collect')
    parser.add_argument('--data-mode', choices=['traj', 'delta_traj'], default='delta_traj', help='Data collection mode')
    
    # Training arguments
    parser.add_argument('--train', action='store_true', help='Train transformer policy')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=2000, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    
    # Testing arguments
    parser.add_argument('--test', action='store_true', help='Test trained policy')
    parser.add_argument('--test-episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--test-steps', type=int, default=300, help='Steps per test episode')
    parser.add_argument('--no-comparison', action='store_true', help='Skip comparison with sampling policy')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    
    # Environment arguments
    parser.add_argument('--env-type', choices=['randpath', 'sine'], default='randpath', help='Environment type')
    parser.add_argument('--device', default='cuda:0', help='Device for computation')
    parser.add_argument('--num-envs', type=int, default=1000, help='Number of environments')
    
    # File paths
    parser.add_argument('--data-path', default='./rolldiff_training_data.pt', help='Dataset file path')
    parser.add_argument('--checkpoint-dir', default='./rolldiff_checkpoints', help='Checkpoint directory')
    
    # Pipeline control
    parser.add_argument('--all', action='store_true', help='Run complete pipeline (collect, train, test)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    results = {}
    
    try:
        # Data collection phase
        if args.collect_data or args.all:
            print("\n" + "="*80)
            print("PHASE 1: DATA COLLECTION")
            print("="*80)
            
            collection_results = collect_training_data(
                num_episodes=args.data_episodes,
                steps_per_episode=args.data_steps,
                num_envs=args.num_envs,
                env_type=args.env_type,
                device=device,
                data_mode=args.data_mode,
                max_samples=args.max_samples,
                save_path=args.data_path
            )
            results['data_collection'] = collection_results
        
        # Training phase
        if args.train or args.all:
            print("\n" + "="*80)
            print("PHASE 2: TRANSFORMER TRAINING")
            print("="*80)
            
            training_results = train_transformer_policy(
                dataset_path=args.data_path,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device,
                checkpoint_dir=args.checkpoint_dir
            )
            results['training'] = training_results
        
        # Testing phase
        if args.test or args.all:
            print("\n" + "="*80)
            print("PHASE 3: POLICY TESTING")
            print("="*80)
            
            checkpoint_path = os.path.join(args.checkpoint_dir, "transformer_policy_final.pt")
            
            testing_results = test_trained_policy(
                checkpoint_path=checkpoint_path,
                num_test_episodes=args.test_episodes,
                steps_per_episode=args.test_steps,
                device=device,
                env_type=args.env_type,
                enable_vis=not args.no_vis,
                comparison_mode=not args.no_comparison
            )
            results['testing'] = testing_results
        
        # Final summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        
        if 'data_collection' in results:
            dc = results['data_collection']
            print(f"Data Collection:")
            print(f"  Samples collected: {dc['num_samples']}")
            print(f"  Episodes: {dc['collection_episodes']}")
            print(f"  Average reward: {np.mean(dc['episode_rewards']):.3f}")
        
        if 'training' in results:
            tr = results['training']
            print(f"Training:")
            print(f"  Training time: {tr['training_time']:.2f} seconds")
            print(f"  Final loss: {tr['training_metrics']['training_losses'][-1]:.6f}")
        
        if 'testing' in results:
            ts = results['testing']
            print(f"Testing:")
            print(f"  Transformer score: {ts['transformer_mean']:.4f} ± {ts['transformer_std']:.4f}")
            if 'improvement' in ts:
                print(f"  Improvement over sampling: {ts['improvement']:.4f} ({ts['improvement_percentage']:.2f}%)")
        
        print(f"\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)