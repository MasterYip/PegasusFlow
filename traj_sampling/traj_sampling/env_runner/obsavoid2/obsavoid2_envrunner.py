"""
High-performance ObsAvoid2 Environment Runner with advanced trajectory optimization.

This module implements an enhanced environment runner for the obsavoid2 environment
with improved efficiency, data collection capabilities, and transformer training support.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from ..env_runner_base import BatchEnvRunnerBase
from .obsavoid2_batch_env import ObsAvoid2BatchEnv, create_complex_bound_batch_env, create_multi_obstacle_batch_env

from ...traj_grad_sampling import TrajGradSampling, TrajGradSamplingCfg
from ...trajopt_policy import TrajOptMode


class ObsAvoid2EnvRunner(BatchEnvRunnerBase):
    """Enhanced environment runner for ObsAvoid2 environments with advanced features.

    This runner provides:
    - High-performance trajectory optimization with GPU acceleration
    - Data collection for imitation learning
    - Transformer policy training capabilities
    - Multi-complexity environment support
    - Advanced visualization and performance monitoring
    """

    def __init__(self,
                 output_dir: str = "./output",
                 num_main_envs: int = 8,
                 num_rollout_per_main: int = 32,
                 device: str = "cuda:0",
                 max_steps: int = 500,
                 horizon_samples: int = 32,
                 horizon_nodes: int = 8,
                 optimize_interval: int = 1,
                 env_type: str = "complex",  # "complex", "multi_obstacle"
                 complexity_level: int = 3,
                 env_step: float = 0.01,
                 n_obs_steps: int = 4,
                 n_action_steps: int = 8,
                 enable_trajectory_optimization: bool = True,
                 enable_vis: bool = False,
                 enable_data_collection: bool = False,
                 data_collection_mode: str = "delta_traj",
                 max_data_samples: int = 20000,
                 vis_time_window: float = 2.0):
        """Initialize the ObsAvoid2 environment runner.

        Args:
            output_dir: Output directory for results
            num_main_envs: Number of main environments
            num_rollout_per_main: Number of rollout environments per main
            device: Device for computations
            max_steps: Maximum number of steps per episode
            horizon_samples: Number of samples in trajectory horizon
            horizon_nodes: Number of control nodes in trajectory
            optimize_interval: Steps between trajectory optimizations
            env_type: Type of environment ("complex", "multi_obstacle")
            complexity_level: Environment complexity level (1-3)
            env_step: Environment timestep
            n_obs_steps: Number of observation steps for policy
            n_action_steps: Number of action steps for policy
            enable_trajectory_optimization: Whether to enable trajectory optimization
            enable_vis: Whether to enable visualization
            enable_data_collection: Whether to enable data collection for training
            data_collection_mode: Mode for data collection ("traj" or "delta_traj")
            max_data_samples: Maximum number of data samples to collect
            vis_time_window: Time window for visualization
        """
        # Create the enhanced batch environment
        if env_type == "multi_obstacle":
            env = create_multi_obstacle_batch_env(
                num_main_envs=num_main_envs,
                num_rollout_per_main=num_rollout_per_main,
                device=device,
                env_step=env_step,
                enable_vis=enable_vis,
                vis_time_window=vis_time_window,
                complexity_level=complexity_level
            )
        else:  # Default to complex
            env = create_complex_bound_batch_env(
                num_main_envs=num_main_envs,
                num_rollout_per_main=num_rollout_per_main,
                device=device,
                env_step=env_step,
                enable_vis=enable_vis,
                vis_time_window=vis_time_window,
                complexity_level=complexity_level
            )

        super().__init__(
            env=env,
            device=device,
            max_steps=max_steps,
            horizon_samples=horizon_samples,
            optimize_interval=optimize_interval
        )

        self.output_dir = output_dir
        self.horizon_nodes = horizon_nodes
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.enable_trajectory_optimization = enable_trajectory_optimization
        self.complexity_level = complexity_level
        self.env_type = env_type

        # Data collection settings
        self.enable_data_collection = enable_data_collection
        self.data_collection_mode = TrajOptMode.DELTA_TRAJ if data_collection_mode == "delta_traj" else TrajOptMode.TRAJ
        self.max_data_samples = max_data_samples

        # Initialize trajectory gradient sampling if enabled
        if self.enable_trajectory_optimization:
            self._init_trajectory_optimization()

        # Enable data collection if requested
        if self.enable_data_collection and hasattr(self, 'traj_sampler'):
            self.traj_sampler.enable_data_collect(
                mode=self.data_collection_mode,
                max_samples=self.max_data_samples
            )
            print(f"Data collection enabled: {self.data_collection_mode.value} mode, max {self.max_data_samples} samples")

        # For policy-based execution
        self.obs_history = None

        # Performance tracking
        self.performance_metrics = {
            'step_times': [],
            'optimization_times': [],
            'reward_history': [],
            'data_samples_collected': 0
        }

    def _init_trajectory_optimization(self):
        """Initialize enhanced trajectory gradient sampling for trajectory optimization."""
        # Create configuration for trajectory optimization
        cfg = TrajGradSamplingCfg()

        # Set enhanced trajectory optimization parameters
        cfg.trajectory_opt.enable_traj_opt = True
        cfg.trajectory_opt.horizon_samples = self.horizon_samples
        cfg.trajectory_opt.horizon_nodes = self.horizon_nodes
        cfg.trajectory_opt.num_samples = self.env.num_rollout_per_main - 1
        cfg.trajectory_opt.num_diffuse_steps = 2  # Increased for better optimization
        cfg.trajectory_opt.num_diffuse_steps_init = 8  # Increased for better initial optimization
        cfg.trajectory_opt.temp_sample = 0.08  # Reduced for more focused sampling
        cfg.trajectory_opt.update_method = "avwbfo"
        cfg.trajectory_opt.interp_method = "spline"
        cfg.trajectory_opt.noise_scaling = 150  # Reduced for better stability
        cfg.trajectory_opt.gamma = 0.99
        cfg.trajectory_opt.policy_type = "sampling"  # Start with sampling
        cfg.trajectory_opt.policy_mode = self.data_collection_mode.value

        # Set environment parameters
        cfg.env.num_actions = self.env.num_actions
        cfg.sim_device = self.device

        # Initialize trajectory gradient sampler
        self.traj_sampler = TrajGradSampling(
            cfg=cfg,
            device=self.device,
            num_envs=self.env.num_main_envs,
            num_actions=self.env.num_actions,
            dt=self.env.dt,
            main_env_indices=self.env.main_env_indices
        )

        # Set the trajectory sampler in the base class
        self.set_traj_sampler(self.traj_sampler)

        print(f"Enhanced trajectory optimization initialized:")
        print(f"  Horizon samples: {self.horizon_samples}, nodes: {self.horizon_nodes}")
        print(f"  Rollout environments: {self.env.num_rollout_per_main}")
        print(f"  Complexity level: {self.complexity_level}")
        print(f"  Environment type: {self.env_type}")

    def run(self, policy, **kwargs):
        raise NotImplementedError("Use run_with_trajectory_optimization for enhanced functionality")
        pass

    def run_with_trajectory_optimization(self, 
                                                seed: int = 20, 
                                                collect_data: bool = None,
                                                **kwargs) -> Dict[str, Any]:
        """Run the environment with enhanced trajectory optimization and optional data collection.

        Args:
            seed: Random seed
            collect_data: Whether to collect data (if None, uses class setting)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing run results and metrics
        """
        if not self.enable_trajectory_optimization:
            raise ValueError("Trajectory optimization is not enabled")

        # Set data collection mode
        if collect_data is None:
            collect_data = self.enable_data_collection

        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Reset environment
        obs = self.reset()

        # Track enhanced metrics
        total_rewards = torch.zeros(self.env.num_main_envs, device=self.device)
        ema_rewards = torch.zeros(self.env.num_main_envs, device=self.device)
        ema_coeff = 0.99
        optimization_times = []
        collision_count = 0
        safety_violations = 0

        print(f"Starting enhanced trajectory optimization:")
        print(f"  Steps: {self.max_steps}, Main envs: {self.env.num_main_envs}")
        print(f"  Complexity level: {self.complexity_level}")
        print(f"  Data collection: {collect_data}")

        import time
        episode_start_time = time.time()

        for step in range(self.max_steps):
            step_start_time = time.time()

            # Optimize trajectories at specified intervals
            if step % self.optimize_interval == 0:
                opt_start_time = time.time()
                print(f"\nStep {step}: Optimizing trajectories...")

                if collect_data and hasattr(self.traj_sampler, 'optimize_and_collect_data'):
                    # Optimize and collect data simultaneously
                    self.traj_sampler.optimize_and_collect_data(
                        rollout_callback=self.rollout_callback,
                        obs=obs if self.traj_sampler.data_collector and self.traj_sampler.data_collector.mode else None,
                        initial=(step == 0)
                    )
                else:
                    # Standard optimization
                    self.optimize_trajectories(initial=(step == 0))

                opt_time = time.time() - opt_start_time
                optimization_times.append(opt_time)
                print(f"Optimization completed in {opt_time:.4f} seconds")

            # Get actions from optimized trajectories
            actions = self.get_next_actions()

            # Step environment
            obs, rewards, dones, info = self.env.step(actions)

            # Update trajectory sampler (shift trajectories)
            if self.traj_sampler is not None:
                self.traj_sampler.shift_trajectory_batch()

            # Update enhanced metrics
            total_rewards += rewards
            ema_rewards = rewards * (1 - ema_coeff) + ema_rewards * ema_coeff

            # Track safety metrics
            collision_count += (rewards < -5.0).sum().item()
            safety_violations += (rewards < 0).sum().item()

            # Store performance data
            step_time = time.time() - step_start_time
            self.performance_metrics['step_times'].append(step_time)
            self.performance_metrics['reward_history'].append(rewards.mean().item())

            # Handle environment termination and reset
            if dones.any():
                terminated_envs = torch.nonzero(dones).flatten()
                print(f"Environments {terminated_envs.tolist()} terminated at step {step}")

                # Reset terminated environments
                for env_idx in terminated_envs:
                    main_env_idx = env_idx.item()
                    if main_env_idx < len(self.env.bound_params):
                        params = self.env.bound_params[main_env_idx]
                        self.env.y[env_idx] = params['initial_y']
                        self.env.v[env_idx] = params['initial_v']
                        self.env.t[env_idx] = 0.0

            # Print progress with enhanced metrics
            if step % 10 == 0:
                avg_reward = rewards.mean().item()
                avg_ema_reward = ema_rewards.mean().item()
                avg_position = self.env.y[self.env.main_env_indices].mean().item()
                avg_step_time = np.mean(self.performance_metrics['step_times'][-10:]) if self.performance_metrics['step_times'] else 0
                
                print(f"Step {step}: Reward={avg_reward:.3f}, EMA={avg_ema_reward:.3f}, "
                      f"Pos={avg_position:.3f}, StepTime={avg_step_time*1000:.1f}ms")

                # Data collection progress
                if collect_data and hasattr(self.traj_sampler, 'data_collector') and self.traj_sampler.data_collector:
                    data_samples = self.traj_sampler.data_collector.num_samples
                    self.performance_metrics['data_samples_collected'] = data_samples
                    print(f"  Data samples collected: {data_samples}/{self.max_data_samples}")

        # Calculate final enhanced metrics
        episode_time = time.time() - episode_start_time
        avg_total_reward = total_rewards.mean().item()
        avg_ema_reward = ema_rewards.mean().item()
        avg_optimization_time = np.mean(optimization_times) if optimization_times else 0.0
        avg_step_time = np.mean(self.performance_metrics['step_times'])

        # Get environment performance metrics
        env_metrics = self.env.get_performance_metrics()

        print(f"\nEnhanced trajectory optimization completed:")
        print(f"  Episode time: {episode_time:.2f} seconds")
        print(f"  Average total reward: {avg_total_reward:.3f}")
        print(f"  Average EMA reward: {avg_ema_reward:.3f}")
        print(f"  Average optimization time: {avg_optimization_time:.4f} seconds")
        print(f"  Average step time: {avg_step_time*1000:.2f} ms")
        print(f"  Collision count: {collision_count}")
        print(f"  Safety violation rate: {safety_violations/(self.max_steps*self.env.num_main_envs)*100:.1f}%")

        if collect_data:
            final_data_samples = self.performance_metrics['data_samples_collected']
            print(f"  Data samples collected: {final_data_samples}")

        return {
            "test_mean_score": avg_ema_reward,
            "total_reward": avg_total_reward,
            "average_reward": avg_total_reward / self.max_steps,
            "steps": self.max_steps,
            "optimization_time": avg_optimization_time,
            "step_time": avg_step_time,
            "episode_time": episode_time,
            "num_optimizations": len(optimization_times),
            "collision_count": collision_count,
            "safety_violations": safety_violations,
            "data_samples_collected": self.performance_metrics['data_samples_collected'],
            "environment_metrics": env_metrics
        }

    def collect_training_data(self, 
                            num_episodes: int = 10,
                            steps_per_episode: int = 200,
                            seed: int = 42) -> Dict[str, Any]:
        """Collect training data using trajectory optimization.

        Args:
            num_episodes: Number of episodes to run for data collection
            steps_per_episode: Steps per episode
            seed: Base random seed

        Returns:
            Data collection summary
        """
        if not self.enable_data_collection:
            raise ValueError("Data collection is not enabled")

        print(f"Starting data collection:")
        print(f"  Episodes: {num_episodes}, Steps per episode: {steps_per_episode}")
        print(f"  Target samples: {self.max_data_samples}")

        total_data_collected = 0
        episode_rewards = []

        original_max_steps = self.max_steps
        self.max_steps = steps_per_episode

        try:
            for episode in range(num_episodes):
                episode_seed = seed + episode * 1000
                print(f"\nData collection episode {episode + 1}/{num_episodes} (seed: {episode_seed})")

                # Run episode with data collection
                results = self.run_with_trajectory_optimization(
                    seed=episode_seed,
                    collect_data=True
                )

                episode_rewards.append(results['test_mean_score'])
                total_data_collected = results['data_samples_collected']

                print(f"  Episode reward: {results['test_mean_score']:.3f}")
                print(f"  Total data samples: {total_data_collected}")

                # Check if we have enough data
                if total_data_collected >= self.max_data_samples:
                    print(f"\nData collection target reached: {total_data_collected} samples")
                    break

        finally:
            self.max_steps = original_max_steps

        # Get final dataset
        dataset = None
        if hasattr(self.traj_sampler, 'get_collected_dataset'):
            dataset = self.traj_sampler.get_collected_dataset()

        summary = {
            'episodes_completed': len(episode_rewards),
            'total_data_samples': total_data_collected,
            'average_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'data_collection_mode': self.data_collection_mode.value,
            'dataset': dataset
        }

        print(f"\nData collection completed:")
        print(f"  Episodes: {summary['episodes_completed']}")
        print(f"  Data samples: {summary['total_data_samples']}")
        print(f"  Average reward: {summary['average_episode_reward']:.3f}")

        return summary

    def train_transformer_policy(self,
                               dataset: Optional[Dict[str, Any]] = None,
                               obs_dim: Optional[int] = None,
                               num_epochs: int = 200,
                               batch_size: int = 64,
                               learning_rate: float = 1e-4,
                               validation_split: float = 0.2,
                               save_checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        """Train a transformer policy on collected data.

        Args:
            dataset: Dataset to train on (if None, uses collected data)
            obs_dim: Observation dimension for transformer
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            validation_split: Validation data fraction
            save_checkpoint_dir: Directory to save checkpoints

        Returns:
            Training results
        """
        if not hasattr(self.traj_sampler, 'setup_transformer_training'):
            raise ValueError("Trajectory sampler does not support transformer training")

        print(f"Setting up transformer policy training:")
        print(f"  Epochs: {num_epochs}, Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")

        # Setup transformer training
        trainer = self.traj_sampler.setup_transformer_training(
            obs_dim=obs_dim or self.env.obs_dim,
            learning_rate=learning_rate,
            d_model=512,  # Larger model for complex environment
            nhead=16,
            num_layers=8,
            dim_feedforward=2048
        )

        # Train the model
        training_metrics = self.traj_sampler.train_transformer_on_data(
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            print_interval=20
        )

        # Save checkpoint if requested
        if save_checkpoint_dir:
            import os
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_checkpoint_dir, f"transformer_policy_epoch_{num_epochs}.pt")
            
            metadata = {
                'complexity_level': self.complexity_level,
                'env_type': self.env_type,
                'horizon_nodes': self.horizon_nodes,
                'action_dim': self.env.action_dim,
                'obs_dim': self.env.obs_dim,
                'training_epochs': num_epochs
            }
            
            self.traj_sampler.save_training_checkpoint(checkpoint_path, num_epochs, metadata)

        print(f"\nTransformer training completed!")
        return training_metrics

    def deploy_and_test_transformer(self, seed: int = 100) -> Dict[str, Any]:
        """Deploy trained transformer and test its performance.

        Args:
            seed: Random seed for testing

        Returns:
            Test results comparing sampling vs transformer performance
        """
        if not hasattr(self.traj_sampler, 'trainer') or self.traj_sampler.trainer is None:
            raise ValueError("No trained transformer available")

        print(f"Testing trained transformer policy vs sampling policy:")

        # Test with sampling policy first
        print("\n1. Testing with sampling policy...")
        sampling_results = self.run_with_trajectory_optimization(seed=seed, collect_data=False)

        # Deploy transformer and test
        print("\n2. Deploying and testing transformer policy...")
        self.traj_sampler.deploy_trained_transformer()
        transformer_results = self.run_with_trajectory_optimization(seed=seed + 1, collect_data=False)

        # Compare results
        improvement = transformer_results['test_mean_score'] - sampling_results['test_mean_score']
        speed_improvement = (sampling_results['optimization_time'] - transformer_results['optimization_time']) / sampling_results['optimization_time'] * 100

        comparison_results = {
            'sampling_policy': sampling_results,
            'transformer_policy': transformer_results,
            'performance_improvement': improvement,
            'speed_improvement_percent': speed_improvement,
            'comparison_summary': {
                'reward_improvement': improvement,
                'speed_improvement': speed_improvement,
                'sampling_score': sampling_results['test_mean_score'],
                'transformer_score': transformer_results['test_mean_score']
            }
        }

        print(f"\nTransformer vs Sampling Comparison:")
        print(f"  Sampling policy score: {sampling_results['test_mean_score']:.4f}")
        print(f"  Transformer policy score: {transformer_results['test_mean_score']:.4f}")
        print(f"  Performance improvement: {improvement:.4f}")
        print(f"  Speed improvement: {speed_improvement:.1f}%")

        return comparison_results

    # ...existing methods for run, run_comparison, etc. from parent class

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        env_metrics = self.env.get_performance_metrics()
        
        summary = {
            'environment_config': {
                'complexity_level': self.complexity_level,
                'env_type': self.env_type,
                'num_main_envs': self.env.num_main_envs,
                'num_rollout_per_main': self.env.num_rollout_per_main,
                'horizon_samples': self.horizon_samples,
                'horizon_nodes': self.horizon_nodes
            },
            'performance_metrics': self.performance_metrics,
            'environment_metrics': env_metrics
        }
        
        return summary