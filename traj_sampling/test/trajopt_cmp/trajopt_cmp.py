"""
Comprehensive comparison of trajectory optimization methods.

This script evaluates and compares different trajectory optimization methods
(MPPI, WBFO, AVWBFO) across multiple environments and noise scheduling strategies.
It generates academic-quality plots and detailed performance metrics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import time
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import torch

# Add parent directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from traj_sampling.optimizer import (
    create_wbfo_optimizer, create_avwbfo_optimizer, create_mppi_optimizer,
    WeightedBasisFunctionOptimizer, ActionValueWBFO, MPPIOptimizer
)
from traj_sampling.noise_scheduler import (
    create_noise_scheduler, NoiseScheduleType,
    create_2d_navigation_scheduler, create_inverted_pendulum_scheduler
)
from traj_sampling.noise_sampler import (
    sample_noise, sample_normal_noise, NoiseSamplerFactory, NoiseDistribution
)
from trajopt_env import (
    create_navigation2d_env, create_inverted_pendulum_env, 
    EnvConfig, Navigation2DEnv, InvertedPendulumEnv
)

# Set style for academic plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class ExperimentConfig:
    """Configuration for trajectory optimization experiments."""
    # Environment settings
    env_name: str = "navigation2d"  # "navigation2d" or "inverted_pendulum"
    horizon_nodes: int = 7  # Actual nodes will be horizon_nodes+1 = 8
    horizon_samples: int = 63  # Actual samples will be horizon_samples+1 = 64
    dt: float = 0.02
    
    # Optimization settings
    num_samples: int = 100
    num_iterations: int = 50
    temp_sample: float = 0.1
    gamma: float = 1.00  # For AVWBFO
    
    # Noise scheduling
    noise_schedule_types: List[str] = None
    
    # Noise sampling settings
    noise_sampler_type: str = 'lhs'  # None, 'mc', 'lhs', 'halton'
    noise_distribution: str = 'normal'  # 'normal', 'uniform'
    noise_sampler_seed: Optional[int] = None
    
    # Experiment settings
    num_trials: int = 5
    save_results: bool = True
    save_plots: bool = True
    results_dir: str = "results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optimization process visualization options
    visualize_optimization: bool = False  # Enable optimization process visualization
    viz_save_interval: int = 5  # Save trajectory every N iterations for visualization
    viz_max_trajectories: int = 10  # Maximum number of trajectories to visualize per trial
    
    def __post_init__(self):
        if self.noise_schedule_types is None:
            self.noise_schedule_types = ["constant", "exponential_decay", "hierarchical"]


class TrajectoryOptimizationComparison:
    """Main class for comparing trajectory optimization methods."""

    def __init__(self, config: ExperimentConfig):
        """Initialize the comparison framework.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Create results directory
        if config.save_results or config.save_plots:
            os.makedirs(config.results_dir, exist_ok=True)
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Initialize optimizers
        self.optimizers = self._create_optimizers()
        
        # Initialize noise schedulers
        self.noise_schedulers = self._create_noise_schedulers()
        
        # Initialize noise sampler
        self.noise_sampler = None
        if config.noise_sampler_type:
            self.noise_sampler = NoiseSamplerFactory.create_sampler(
                sampler_type=config.noise_sampler_type,
                distribution=NoiseDistribution(config.noise_distribution),
                device=self.device,
                seed=config.noise_sampler_seed
            )
        
        # Results storage
        self.results = defaultdict(lambda: defaultdict(list))
        
        print(f"Initialized comparison framework on device: {self.device}")
        print(f"Environment: {config.env_name}")
        print(f"Optimizers: {list(self.optimizers.keys())}")
        print(f"Noise schedulers: {list(self.noise_schedulers.keys())}")
        print(f"Noise sampler: {config.noise_sampler_type} ({config.noise_distribution})")

    def _create_environment(self):
        """Create the test environment."""
        env_config = EnvConfig(
            horizon_samples=self.config.horizon_samples,
            dt=self.config.dt,
            device=self.config.device
        )
        
        if self.config.env_name == "navigation2d":
            return create_navigation2d_env(env_config)
        elif self.config.env_name == "inverted_pendulum":
            return create_inverted_pendulum_env(env_config)
        else:
            raise ValueError(f"Unknown environment: {self.config.env_name}")

    def _create_optimizers(self):
        """Create trajectory optimization methods."""
        optimizers = {}
        
        # Common configuration class
        class OptimizerCfg:
            class env:
                num_actions = self.env.get_action_dim()
            
            class trajectory_opt:
                num_samples = self.config.num_samples
                temp_sample = self.config.temp_sample
                horizon_samples = self.config.horizon_samples
                horizon_nodes = self.config.horizon_nodes
                gamma = self.config.gamma
                dt = self.config.dt
            
            sim_device = self.config.device
        
        cfg = OptimizerCfg()
        
        # Create optimizers
        optimizers["WBFO"] = create_wbfo_optimizer(cfg)
        optimizers["AVWBFO"] = create_avwbfo_optimizer(cfg)
        optimizers["MPPI"] = create_mppi_optimizer(cfg)
        
        return optimizers

    def _create_noise_schedulers(self):
        """Create noise schedulers for different strategies."""
        schedulers = {}
        action_dim = self.env.get_action_dim()
        # Note: actual number of nodes is horizon_nodes + 1
        actual_nodes = self.config.horizon_nodes + 1
        
        for schedule_type in self.config.noise_schedule_types:
            if schedule_type == "hierarchical":
                # Use task-specific hierarchical scheduler
                if self.config.env_name == "navigation2d":
                    schedulers[schedule_type] = create_2d_navigation_scheduler(
                        actual_nodes, self.device
                    )
                elif self.config.env_name == "inverted_pendulum":
                    schedulers[schedule_type] = create_inverted_pendulum_scheduler(
                        actual_nodes, self.device
                    )
            else:
                schedulers[schedule_type] = create_noise_scheduler(
                    schedule_type,
                    actual_nodes,
                    action_dim,
                    self.device
                )
        
        return schedulers

    def generate_initial_trajectory(self) -> torch.Tensor:
        """Generate an initial trajectory for optimization."""
        # Note: actual number of nodes is horizon_nodes + 1
        actual_nodes = self.config.horizon_nodes + 1
        
        if self.config.env_name == "navigation2d":
            # Linear trajectory from start to goal (position waypoints)
            start_pos = self.env.start_pos
            goal_pos = self.env.goal_pos
            
            # Create waypoints with fixed start and end
            trajectory = torch.zeros(actual_nodes, 2, device=self.device)
            trajectory[0] = start_pos  # Fixed start
            trajectory[-1] = goal_pos  # Fixed end
            
            # Intermediate nodes interpolated between start and goal
            for i in range(1, actual_nodes - 1):
                t = i / (actual_nodes - 1)  # Progress from 0 to 1
                trajectory[i] = start_pos + t * (goal_pos - start_pos)
                
        elif self.config.env_name == "inverted_pendulum":
            # Zero force trajectory (let gravity act)
            trajectory = torch.zeros(actual_nodes, 1, device=self.device)
        
        return trajectory

    def run_single_optimization(self,
                              optimizer_name: str,
                              noise_scheduler_name: str,
                              initial_trajectory: torch.Tensor) -> Dict[str, Any]:
        """Run a single optimization trial.

        Args:
            optimizer_name: Name of the optimizer to use
            noise_scheduler_name: Name of the noise scheduler to use
            initial_trajectory: Initial trajectory

        Returns:
            Dictionary containing optimization results
        """
        optimizer = self.optimizers[optimizer_name]
        noise_scheduler = self.noise_schedulers[noise_scheduler_name]
        
        # Initialize tracking
        history = {
            'trajectories': [initial_trajectory.clone()],
            'costs': [],
            'times': [],
            'noise_scales': []
        }
        
        current_traj = initial_trajectory.clone()
        start_time = time.time()
        
        for iteration in range(self.config.num_iterations):
            iter_start = time.time()
            
            # Get noise scale for this iteration
            noise_scale = noise_scheduler.get_noise_scale(
                iteration, self.config.num_iterations
            )
            history['noise_scales'].append(noise_scale.clone())

            # Generate samples around current trajectory using the noise sampler
            # Note: current_traj has shape [horizon_nodes+1, action_dim]
            actual_nodes, action_dim = current_traj.shape
            
            # Generate noise using the configured noise sampler
            eps_shape = (self.config.num_samples, actual_nodes, action_dim)
            
            if self.noise_sampler and self.config.noise_distribution == 'normal':
                eps = self.noise_sampler.sample(eps_shape, mean=0.0, std=1.0)
            elif self.noise_sampler and self.config.noise_distribution == 'uniform':
                eps = self.noise_sampler.sample(eps_shape, low=-1.0, high=1.0)
            else:
                # Fallback to standard random sampling
                eps = torch.randn(eps_shape, device=self.device)
            
            # Apply noise scaling
            if noise_scale.dim() == 2:
                # Full [horizon_nodes+1, action_dim] noise scale
                samples = current_traj.unsqueeze(0) + eps * noise_scale.unsqueeze(0)
            else:
                # Scalar or simplified noise scale
                samples = current_traj.unsqueeze(0) + eps * noise_scale
            
            # For navigation2d: Keep start and end nodes fixed
            if self.config.env_name == "navigation2d":
                samples[:, 0, :] = self.env.start_pos  # Fix start node
                samples[:, -1, :] = self.env.goal_pos   # Fix end node

            # Ensure current trajectory also has fixed start/end before adding to samples
            if self.config.env_name == "navigation2d":
                current_traj[0, :] = self.env.start_pos  # Fix start node
                current_traj[-1, :] = self.env.goal_pos   # Fix end node
            
            # FIXME: this can change the initial state of pendulum
            initial_state = None
            if self.config.env_name == "inverted_pendulum":
                # State: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
                initial_state = torch.zeros(self.config.num_samples+1, 4, device=self.device)
                initial_state[:, 3] = -0.1
            
            # Add the current trajectory as the first sample
            all_samples = torch.cat([current_traj.unsqueeze(0), samples], dim=0)
            
            # Convert to dense trajectories for environment rollout
            dense_samples = optimizer.node2dense(all_samples)
            
            # Evaluate trajectories in environment
            rewards = self.env.batch_rollout(dense_samples)
            
            # Update trajectory using the optimizer
            if iteration>0:
                current_traj = optimizer.optimize(
                    current_traj,
                    all_samples,
                    rewards
                )
            
            # For navigation2d: Ensure start and end nodes remain fixed after optimization
            if self.config.env_name == "navigation2d":
                current_traj[0, :] = self.env.start_pos  # Fix start node
                current_traj[-1, :] = self.env.goal_pos   # Fix end node
            
            # Evaluate current trajectory performance
            dense_current = optimizer.node2dense(current_traj.unsqueeze(0))
            current_rewards = self.env.batch_rollout(dense_current)
            current_cost = torch.sum(current_rewards).item()
            
            # Store results
            history['trajectories'].append(current_traj.clone())
            history['costs'].append(current_cost)
            history['times'].append(time.time() - iter_start)
            
            print(f"{optimizer_name} + {noise_scheduler_name} "
                  f"Iteration {iteration+1}/{self.config.num_iterations}: "
                  f"Cost = {current_cost:.4f}")
        
        total_time = time.time() - start_time
        
        return {
            'optimizer': optimizer_name,
            'noise_scheduler': noise_scheduler_name,
            'history': history,
            'final_cost': history['costs'][-1],
            'total_time': total_time,
            'best_cost': max(history['costs']),
            'convergence_iteration': np.argmax(history['costs']),
            'avg_iteration_time': np.mean(history['times'])
        }

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete comparison experiment."""
        print("Starting trajectory optimization comparison experiment...")
        
        all_results = []
        
        for trial in range(self.config.num_trials):
            print(f"\n=== Trial {trial + 1}/{self.config.num_trials} ===")
            
            # Reset environment for each trial FIRST
            if hasattr(self.env, '_generate_scenario'):
                self.env._generate_scenario()
            
            # Generate new initial trajectory for each trial AFTER environment reset
            initial_traj = self.generate_initial_trajectory()
            
            for optimizer_name in self.optimizers.keys():
                for noise_scheduler_name in self.noise_schedulers.keys():
                    print(f"\nRunning {optimizer_name} with {noise_scheduler_name}...")
                    
                    result = self.run_single_optimization(
                        optimizer_name, noise_scheduler_name, initial_traj
                    )
                    result['trial'] = trial
                    all_results.append(result)
                    
                    # Store in organized structure
                    key = f"{optimizer_name}_{noise_scheduler_name}"
                    self.results[key]['costs'].append(result['final_cost'])
                    self.results[key]['times'].append(result['total_time'])
                    self.results[key]['convergence'].append(result['convergence_iteration'])
        
        # Compute aggregate statistics
        aggregate_results = self._compute_aggregate_statistics()
        
        # Save results
        if self.config.save_results:
            self._save_results(all_results, aggregate_results)
        
        return {
            'individual_results': all_results,
            'aggregate_results': aggregate_results
        }

    def _compute_aggregate_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across trials."""
        aggregate = {}
        
        for method_key, results in self.results.items():
            optimizer_name, noise_scheduler_name = method_key.split('_', 1)
            
            costs = np.array(results['costs'])
            times = np.array(results['times'])
            convergence = np.array(results['convergence'])
            
            aggregate[method_key] = {
                'optimizer': optimizer_name,
                'noise_scheduler': noise_scheduler_name,
                'cost_mean': np.mean(costs),
                'cost_std': np.std(costs),
                'cost_median': np.median(costs),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'convergence_mean': np.mean(convergence),
                'convergence_std': np.std(convergence),
                'success_rate': np.mean(costs > 0)  # Assuming positive cost is success
            }
        
        return aggregate

    def _save_results(self, individual_results: List[Dict], aggregate_results: Dict[str, Any]):
        """Save results to files."""
        
        def convert_to_serializable(obj):
            """Convert tensors and numpy types to JSON serializable types."""
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Save individual results
        individual_file = os.path.join(self.config.results_dir, "individual_results.json")
        with open(individual_file, 'w') as f:
            # Convert all data to JSON serializable format
            serializable_results = convert_to_serializable(individual_results)
            json.dump(serializable_results, f, indent=2)
        
        # Save aggregate results
        aggregate_file = os.path.join(self.config.results_dir, "aggregate_results.json")
        with open(aggregate_file, 'w') as f:
            serializable_aggregate = convert_to_serializable(aggregate_results)
            json.dump(serializable_aggregate, f, indent=2)
        
        # Save configuration
        config_file = os.path.join(self.config.results_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"Results saved to {self.config.results_dir}")

    def generate_plots(self, experiment_results: Dict[str, Any]):
        """Generate academic-quality plots for the results."""
        if not self.config.save_plots:
            return
        
        individual_results = experiment_results['individual_results']
        aggregate_results = experiment_results['aggregate_results']
        
        # Create plots directory
        plots_dir = os.path.join(self.config.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Performance comparison plot
        self._plot_performance_comparison(aggregate_results, plots_dir)
        
        # 2. Convergence analysis
        self._plot_convergence_analysis(individual_results, plots_dir)
        
        # 3. Computational efficiency analysis
        self._plot_computational_efficiency(aggregate_results, plots_dir)
        
        # 4. Noise scheduling impact
        self._plot_noise_scheduling_impact(aggregate_results, plots_dir)
        
        # 5. Trajectory visualization
        self._plot_trajectory_examples(individual_results, plots_dir)
        
        # 6. Optimization process visualization (if enabled)
        self._plot_optimization_process(individual_results, plots_dir)
        
        print(f"Plots saved to {plots_dir}")

    def _plot_performance_comparison(self, aggregate_results: Dict[str, Any], plots_dir: str):
        """Plot performance comparison across methods."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        methods = list(aggregate_results.keys())
        optimizers = [aggregate_results[m]['optimizer'] for m in methods]
        schedulers = [aggregate_results[m]['noise_scheduler'] for m in methods]
        costs = [aggregate_results[m]['cost_mean'] for m in methods]
        cost_stds = [aggregate_results[m]['cost_std'] for m in methods]
        times = [aggregate_results[m]['time_mean'] for m in methods]
        
        # Create method labels
        labels = [f"{opt}\n({sch})" for opt, sch in zip(optimizers, schedulers)]
        
        # Plot 1: Final cost comparison
        bars1 = axes[0, 0].bar(range(len(methods)), costs, yerr=cost_stds, capsize=5)
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('Final Cost (Higher is Better)')
        axes[0, 0].set_title('Performance Comparison: Final Cost')
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Color bars by optimizer
        colors = {'MPPI': 'red', 'WBFO': 'blue', 'AVWBFO': 'green'}
        for i, bar in enumerate(bars1):
            bar.set_color(colors.get(optimizers[i], 'gray'))
        
        # Plot 2: Computation time comparison
        bars2 = axes[0, 1].bar(range(len(methods)), times)
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Total Time (s)')
        axes[0, 1].set_title('Computational Efficiency')
        axes[0, 1].set_xticks(range(len(methods)))
        axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars2):
            bar.set_color(colors.get(optimizers[i], 'gray'))
        
        # Plot 3: Performance vs Time scatter
        for i, (opt, cost, time) in enumerate(zip(optimizers, costs, times)):
            axes[1, 0].scatter(time, cost, c=colors.get(opt, 'gray'), 
                             s=100, alpha=0.7, label=opt if opt not in axes[1, 0].get_legend_handles_labels()[1] else "")
            axes[1, 0].annotate(schedulers[i], (time, cost), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        
        axes[1, 0].set_xlabel('Total Time (s)')
        axes[1, 0].set_ylabel('Final Cost')
        axes[1, 0].set_title('Performance vs Computational Cost')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Success rate by noise scheduler
        scheduler_performance = defaultdict(list)
        for method in methods:
            sched = aggregate_results[method]['noise_scheduler']
            cost = aggregate_results[method]['cost_mean']
            scheduler_performance[sched].append(cost)
        
        sched_names = list(scheduler_performance.keys())
        sched_means = [np.mean(scheduler_performance[s]) for s in sched_names]
        
        axes[1, 1].bar(range(len(sched_names)), sched_means)
        axes[1, 1].set_xlabel('Noise Scheduler')
        axes[1, 1].set_ylabel('Average Final Cost')
        axes[1, 1].set_title('Noise Scheduler Impact')
        axes[1, 1].set_xticks(range(len(sched_names)))
        axes[1, 1].set_xticklabels(sched_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_convergence_analysis(self, individual_results: List[Dict], plots_dir: str):
        """Plot convergence analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Group results by method
        method_results = defaultdict(list)
        for result in individual_results:
            key = f"{result['optimizer']}_{result['noise_scheduler']}"
            method_results[key].append(result)
        
        colors = {'MPPI': 'red', 'WBFO': 'blue', 'AVWBFO': 'green'}
        linestyles = {'constant': '-', 'exponential_decay': '--', 'hierarchical': '-.'}
        
        # Plot 1: Average convergence curves
        for method_key, results in method_results.items():
            optimizer_name, noise_scheduler_name = method_key.split('_', 1)
            
            # Collect all cost histories
            all_costs = []
            for result in results:
                costs = result['history']['costs']
                all_costs.append(costs)
            
            # Compute mean and std
            min_length = min(len(costs) for costs in all_costs)
            truncated_costs = [costs[:min_length] for costs in all_costs]
            mean_costs = np.mean(truncated_costs, axis=0)
            std_costs = np.std(truncated_costs, axis=0)
            iterations = range(len(mean_costs))
            
            color = colors.get(optimizer_name, 'gray')
            linestyle = linestyles.get(noise_scheduler_name, '-')
            
            axes[0, 0].plot(iterations, mean_costs, color=color, linestyle=linestyle,
                          label=f"{optimizer_name} ({noise_scheduler_name})", linewidth=2)
            axes[0, 0].fill_between(iterations, mean_costs - std_costs, mean_costs + std_costs,
                                  color=color, alpha=0.2)
        
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Cost')
        axes[0, 0].set_title('Convergence Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Convergence rate (improvement per iteration)
        for method_key, results in method_results.items():
            optimizer_name, noise_scheduler_name = method_key.split('_', 1)
            
            improvements = []
            for result in results:
                costs = np.array(result['history']['costs'])
                if len(costs) > 1:
                    improvement = costs[-1] - costs[0]  # Total improvement
                    improvements.append(improvement)
            
            if improvements:
                color = colors.get(optimizer_name, 'gray')
                axes[0, 1].bar(method_key, np.mean(improvements), color=color, alpha=0.7,
                             yerr=np.std(improvements), capsize=5)
        
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Total Improvement')
        axes[0, 1].set_title('Convergence Improvement')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Sample efficiency (cost vs number of samples)
        sample_counts = []
        final_costs = []
        method_labels = []
        
        for method_key, results in method_results.items():
            for result in results:
                # Approximate total samples used
                total_samples = self.config.num_samples * self.config.num_iterations
                sample_counts.append(total_samples)
                final_costs.append(result['final_cost'])
                method_labels.append(method_key)
        
        # Create scatter plot with different markers for each optimizer
        markers = {'MPPI': 'o', 'WBFO': 's', 'AVWBFO': '^'}
        for method_key in method_results.keys():
            optimizer_name, noise_scheduler_name = method_key.split('_', 1)
            mask = [label == method_key for label in method_labels]
            x = [sample_counts[i] for i, m in enumerate(mask) if m]
            y = [final_costs[i] for i, m in enumerate(mask) if m]
            
            color = colors.get(optimizer_name, 'gray')
            marker = markers.get(optimizer_name, 'o')
            
            # Check if label already exists
            existing_labels = []
            if axes[1, 0].legend_:
                existing_labels = [t.get_text() for t in axes[1, 0].legend_.get_texts()]
            
            label = f"{optimizer_name}" if optimizer_name not in existing_labels else ""
            axes[1, 0].scatter(x, y, c=color, marker=marker, s=100, alpha=0.7, label=label)
        
        axes[1, 0].set_xlabel('Total Samples Used')
        axes[1, 0].set_ylabel('Final Cost')
        axes[1, 0].set_title('Sample Efficiency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Iteration time analysis
        for method_key, results in method_results.items():
            optimizer_name, noise_scheduler_name = method_key.split('_', 1)
            
            avg_times = []
            for result in results:
                avg_times.append(result['avg_iteration_time'])
            
            color = colors.get(optimizer_name, 'gray')
            axes[1, 1].bar(method_key, np.mean(avg_times), color=color, alpha=0.7,
                         yerr=np.std(avg_times), capsize=5)
        
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('Average Iteration Time (s)')
        axes[1, 1].set_title('Per-Iteration Computational Cost')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_computational_efficiency(self, aggregate_results: Dict[str, Any], plots_dir: str):
        """Plot computational efficiency analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Prepare data
        methods = list(aggregate_results.keys())
        optimizers = [aggregate_results[m]['optimizer'] for m in methods]
        times = [aggregate_results[m]['time_mean'] for m in methods]
        costs = [aggregate_results[m]['cost_mean'] for m in methods]
        
        colors = {'MPPI': 'red', 'WBFO': 'blue', 'AVWBFO': 'green'}
        
        # Plot 1: Time comparison by optimizer
        optimizer_times = defaultdict(list)
        for opt, time in zip(optimizers, times):
            optimizer_times[opt].append(time)
        
        opt_names = list(optimizer_times.keys())
        opt_means = [np.mean(optimizer_times[opt]) for opt in opt_names]
        opt_stds = [np.std(optimizer_times[opt]) for opt in opt_names]
        
        bars = axes[0].bar(range(len(opt_names)), opt_means, yerr=opt_stds, capsize=5)
        for i, bar in enumerate(bars):
            bar.set_color(colors.get(opt_names[i], 'gray'))
        
        axes[0].set_xlabel('Optimizer')
        axes[0].set_ylabel('Average Total Time (s)')
        axes[0].set_title('Computational Time by Optimizer')
        axes[0].set_xticks(range(len(opt_names)))
        axes[0].set_xticklabels(opt_names)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Efficiency ratio (performance per unit time)
        efficiency_ratios = [cost / time for cost, time in zip(costs, times)]
        
        bars = axes[1].bar(range(len(methods)), efficiency_ratios)
        for i, bar in enumerate(bars):
            bar.set_color(colors.get(optimizers[i], 'gray'))
        
        axes[1].set_xlabel('Method')
        axes[1].set_ylabel('Performance / Time Ratio')
        axes[1].set_title('Computational Efficiency')
        axes[1].set_xticks(range(len(methods)))
        axes[1].set_xticklabels([f"{opt}\n({aggregate_results[m]['noise_scheduler']})" 
                               for m, opt in zip(methods, optimizers)], rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Pareto frontier (performance vs time)
        for opt in set(optimizers):
            opt_indices = [i for i, o in enumerate(optimizers) if o == opt]
            opt_times = [times[i] for i in opt_indices]
            opt_costs = [costs[i] for i in opt_indices]
            
            axes[2].scatter(opt_times, opt_costs, c=colors.get(opt, 'gray'), 
                          label=opt, s=100, alpha=0.7)
        
        axes[2].set_xlabel('Total Time (s)')
        axes[2].set_ylabel('Final Cost')
        axes[2].set_title('Performance vs Computational Cost')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'computational_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_noise_scheduling_impact(self, aggregate_results: Dict[str, Any], plots_dir: str):
        """Plot the impact of different noise scheduling strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Group results by noise scheduler
        scheduler_results = defaultdict(list)
        for method, result in aggregate_results.items():
            scheduler = result['noise_scheduler']
            scheduler_results[scheduler].append(result)
        
        # Plot 1: Performance by noise scheduler
        scheduler_names = list(scheduler_results.keys())
        scheduler_costs = []
        scheduler_stds = []
        
        for scheduler in scheduler_names:
            costs = [r['cost_mean'] for r in scheduler_results[scheduler]]
            scheduler_costs.append(np.mean(costs))
            scheduler_stds.append(np.std(costs))
        
        bars = axes[0, 0].bar(range(len(scheduler_names)), scheduler_costs, 
                            yerr=scheduler_stds, capsize=5)
        axes[0, 0].set_xlabel('Noise Scheduler')
        axes[0, 0].set_ylabel('Average Final Cost')
        axes[0, 0].set_title('Performance by Noise Scheduling Strategy')
        axes[0, 0].set_xticks(range(len(scheduler_names)))
        axes[0, 0].set_xticklabels(scheduler_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Convergence speed by noise scheduler
        scheduler_convergence = []
        conv_stds = []
        
        for scheduler in scheduler_names:
            conv_rates = [r['convergence_mean'] for r in scheduler_results[scheduler]]
            scheduler_convergence.append(np.mean(conv_rates))
            conv_stds.append(np.std(conv_rates))
        
        bars = axes[0, 1].bar(range(len(scheduler_names)), scheduler_convergence,
                            yerr=conv_stds, capsize=5)
        axes[0, 1].set_xlabel('Noise Scheduler')
        axes[0, 1].set_ylabel('Average Convergence Iteration')
        axes[0, 1].set_title('Convergence Speed by Noise Scheduler')
        axes[0, 1].set_xticks(range(len(scheduler_names)))
        axes[0, 1].set_xticklabels(scheduler_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Heatmap of optimizer-scheduler combinations
        optimizers = list(set(r['optimizer'] for results in scheduler_results.values() for r in results))
        
        heatmap_data = np.zeros((len(optimizers), len(scheduler_names)))
        
        for method, result in aggregate_results.items():
            opt_idx = optimizers.index(result['optimizer'])
            sched_idx = scheduler_names.index(result['noise_scheduler'])
            heatmap_data[opt_idx, sched_idx] = result['cost_mean']
        
        im = axes[1, 0].imshow(heatmap_data, cmap='viridis', aspect='auto')
        axes[1, 0].set_xticks(range(len(scheduler_names)))
        axes[1, 0].set_xticklabels(scheduler_names, rotation=45, ha='right')
        axes[1, 0].set_yticks(range(len(optimizers)))
        axes[1, 0].set_yticklabels(optimizers)
        axes[1, 0].set_title('Performance Heatmap: Optimizer × Noise Scheduler')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 0])
        cbar.set_label('Final Cost')
        
        # Add text annotations
        for i in range(len(optimizers)):
            for j in range(len(scheduler_names)):
                text = axes[1, 0].text(j, i, f'{heatmap_data[i, j]:.2f}',
                                     ha="center", va="center", color="white")
        
        # Plot 4: Time efficiency by scheduler
        scheduler_times = []
        time_stds = []
        
        for scheduler in scheduler_names:
            times = [r['time_mean'] for r in scheduler_results[scheduler]]
            scheduler_times.append(np.mean(times))
            time_stds.append(np.std(times))
        
        bars = axes[1, 1].bar(range(len(scheduler_names)), scheduler_times,
                            yerr=time_stds, capsize=5)
        axes[1, 1].set_xlabel('Noise Scheduler')
        axes[1, 1].set_ylabel('Average Total Time (s)')
        axes[1, 1].set_title('Computational Time by Noise Scheduler')
        axes[1, 1].set_xticks(range(len(scheduler_names)))
        axes[1, 1].set_xticklabels(scheduler_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'noise_scheduling_impact.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_trajectory_examples(self, individual_results: List[Dict], plots_dir: str):
        """Plot example trajectories for visualization."""
        # Get best result for each optimizer
        best_results = {}
        for result in individual_results:
            optimizer = result['optimizer']
            if optimizer not in best_results or result['final_cost'] > best_results[optimizer]['final_cost']:
                best_results[optimizer] = result
        
        # Create individual plots for each optimizer and save them
        for optimizer, result in best_results.items():
            # Get final trajectory (control nodes)
            final_trajectory_nodes = result['history']['trajectories'][-1]
            
            # Convert to dense trajectory for visualization
            opt = self.optimizers[optimizer]
            final_trajectory_dense = opt.node2dense(final_trajectory_nodes.unsqueeze(0)).squeeze(0)
            
            # Create trajectory visualization
            fig = self.env.visualize_trajectory(
                final_trajectory_dense, 
                f"Best {optimizer} Trajectory (Cost: {result['final_cost']:.2f})"
            )
            
            # Save individual trajectory plot
            filename = f'trajectory_{optimizer.lower()}.svg'
            plt.figure(fig.number)
            plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Create a combined figure using direct plotting instead of copying patches
        fig_size = (15, 5 * len(best_results))
        fig, axes = plt.subplots(len(best_results), 1, figsize=fig_size)
        if len(best_results) == 1:
            axes = [axes]
        
        for i, (optimizer, result) in enumerate(best_results.items()):
            # Get final trajectory (control nodes)
            final_trajectory_nodes = result['history']['trajectories'][-1]
            
            # Convert to dense trajectory for visualization
            opt = self.optimizers[optimizer]
            final_trajectory_dense = opt.node2dense(final_trajectory_nodes.unsqueeze(0)).squeeze(0)
            
            # Plot directly on the subplot for navigation environment
            if self.config.env_name == "navigation2d":
                self._plot_navigation_trajectory(axes[i], final_trajectory_dense, 
                                               f"Best {optimizer} Trajectory (Cost: {result['final_cost']:.2f})")
            elif self.config.env_name == "inverted_pendulum":
                # For pendulum, just plot a simple time series
                if final_trajectory_dense.dim() == 2:
                    trajectory = final_trajectory_dense.cpu().numpy()
                else:
                    trajectory = final_trajectory_dense.cpu().numpy()
                
                times = np.arange(len(trajectory)) * self.env.dt
                axes[i].plot(times, trajectory)
                axes[i].set_xlabel('Time [s]')
                axes[i].set_ylabel('Control Force [N]')
                axes[i].set_title(f"Best {optimizer} Control Trajectory (Cost: {result['final_cost']:.2f})")
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'trajectory_examples_combined.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_navigation_trajectory(self, ax, trajectory, title):
        """Plot navigation trajectory directly on given axes."""
        # Ensure trajectory is on correct device
        trajectory = trajectory.to(self.device)
        
        # Trajectory already includes all nodes (start to end)
        positions = trajectory.cpu().numpy()
        
        # Plot workspace
        ax.set_xlim(-2, self.env.workspace_size[0]+2)
        ax.set_ylim(-2, self.env.workspace_size[1]+2)
        
        # Plot obstacles (create new patches for this figure)
        for obs_x, obs_y, obs_r in self.env.obstacles:
            import matplotlib.patches as patches
            circle = patches.Circle((obs_x, obs_y), obs_r, fill=True, alpha=0.3, color='red')
            ax.add_patch(circle)
        
        # Plot goal (create new patch)
        import matplotlib.patches as patches
        goal_circle = patches.Circle(self.env.goal_pos.cpu().numpy(), self.env.goal_radius, 
                                   fill=True, alpha=0.3, color='green')
        ax.add_patch(goal_circle)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
        
        # Plot trajectory nodes (control points)
        ax.plot(positions[:, 0], positions[:, 1], 'ko', markersize=6, alpha=0.7, label='Nodes')
        
        # Mark start and end nodes specially
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start (Fixed)')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End (Fixed)')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

    def _plot_optimization_process(self, results: List[Dict[str, Any]], plots_dir: str):
        """Create optimization process visualization showing trajectory evolution."""
        if not self.config.visualize_optimization:
            return
            
        print("Generating optimization process visualizations...")
        
        # Create directory for optimization visualizations
        opt_viz_dir = os.path.join(plots_dir, 'optimization_process')
        os.makedirs(opt_viz_dir, exist_ok=True)
        
        # Group results by optimizer and noise scheduler
        grouped_results = {}
        for result in results:
            key = f"{result['optimizer']}_{result['noise_scheduler']}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Create visualization for each optimizer-scheduler combination
        for combo_name, combo_results in grouped_results.items():
            if self.config.env_name == "navigation2d":
                self._plot_navigation_optimization_process(combo_results, combo_name, opt_viz_dir)
            elif self.config.env_name == "inverted_pendulum":
                self._plot_pendulum_optimization_process(combo_results, combo_name, opt_viz_dir)
    
    def _plot_navigation_optimization_process(self, results: List[Dict[str, Any]], combo_name: str, plots_dir: str):
        """Create optimization process visualization for navigation task."""
        # Select iterations to visualize (start, middle, end)
        max_iterations = len(results[0]['history']['trajectories']) - 1  # -1 because first is initial
        iteration_indices = []
        
        # Always include first and last
        iteration_indices.append(0)  # Initial trajectory
        if max_iterations > 0:
            iteration_indices.append(max_iterations)  # Final trajectory
        
        # Add some intermediate iterations
        if max_iterations > 10:
            iteration_indices.extend([max_iterations//4, max_iterations//2, 3*max_iterations//4])
        elif max_iterations > 2:
            iteration_indices.append(max_iterations//2)
        
        iteration_indices = sorted(set(iteration_indices))
        
        # Limit number of trials to visualize
        max_trials = min(len(results), self.config.viz_max_trajectories // len(iteration_indices))
        if max_trials == 0:
            max_trials = 1
        
        # Create subplots
        num_iterations = len(iteration_indices)
        fig, axes = plt.subplots(1, num_iterations, figsize=(4*num_iterations, 4))
        if num_iterations == 1:
            axes = [axes]
        
        colors = plt.cm.viridis(np.linspace(0, 1, max_trials))
        
        for i, iteration_idx in enumerate(iteration_indices):
            ax = axes[i]
            
            # Plot environment
            ax.set_xlim(0, self.env.workspace_size[0])
            ax.set_ylim(0, self.env.workspace_size[1])
            
            # Plot obstacles
            for obs_x, obs_y, obs_r in self.env.obstacles:
                circle = patches.Circle((obs_x, obs_y), obs_r, fill=True, alpha=0.3, color='red')
                ax.add_patch(circle)
            
            # Plot goal
            goal_circle = patches.Circle(self.env.goal_pos.cpu().numpy(), self.env.goal_radius, 
                                       fill=True, alpha=0.3, color='green')
            ax.add_patch(goal_circle)
            
            # Plot trajectories from multiple trials
            for trial_idx in range(max_trials):
                result = results[trial_idx]
                trajectory = result['history']['trajectories'][iteration_idx]
                
                # Trajectory already includes all nodes (start to end)
                positions = trajectory.cpu().numpy()
                
                # Plot trajectory
                alpha = 0.7 if max_trials == 1 else 0.5
                ax.plot(positions[:, 0], positions[:, 1], '-', 
                       color=colors[trial_idx], alpha=alpha, linewidth=2,
                       label=f'Trial {trial_idx+1}' if i == 0 else "")
                
                # Plot trajectory nodes
                ax.plot(positions[:, 0], positions[:, 1], 'o', 
                       color=colors[trial_idx], alpha=alpha*0.8, markersize=4)
                
                # Mark start and end nodes
                if trial_idx == 0:  # Only mark for first trial to avoid clutter
                    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, 
                           label='Start (Fixed)' if i == 0 else "")
                    ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10,
                           label='End (Fixed)' if i == 0 else "")
            
            # Set title
            if iteration_idx == 0:
                ax.set_title(f'Initial\n(Iteration 0)')
            else:
                ax.set_title(f'Optimized\n(Iteration {iteration_idx})')
            
            ax.set_xlabel('X Position')
            if i == 0:
                ax.set_ylabel('Y Position')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.suptitle(f'Optimization Process: {combo_name.replace("_", " + ")}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'optimization_process_{combo_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pendulum_optimization_process(self, results: List[Dict[str, Any]], combo_name: str, plots_dir: str):
        """Create optimization process visualization for pendulum task."""
        # Select iterations to visualize
        max_iterations = len(results[0]['history']['trajectories']) - 1
        iteration_indices = [0]  # Initial
        if max_iterations > 0:
            iteration_indices.append(max_iterations)  # Final
        if max_iterations > 10:
            iteration_indices.extend([max_iterations//4, max_iterations//2, 3*max_iterations//4])
        elif max_iterations > 2:
            iteration_indices.append(max_iterations//2)
        
        iteration_indices = sorted(set(iteration_indices))
        
        # Limit trials
        max_trials = min(len(results), self.config.viz_max_trajectories)
        
        # Create figure
        fig, axes = plt.subplots(len(iteration_indices), 1, figsize=(12, 3*len(iteration_indices)))
        if len(iteration_indices) == 1:
            axes = [axes]
        
        colors = plt.cm.viridis(np.linspace(0, 1, max_trials))
        
        for i, iteration_idx in enumerate(iteration_indices):
            ax = axes[i]
            
            for trial_idx in range(max_trials):
                result = results[trial_idx]
                trajectory = result['history']['trajectories'][iteration_idx]
                
                # Convert to numpy for plotting
                control_inputs = trajectory.cpu().numpy().squeeze()
                times = np.arange(len(control_inputs)) * self.config.dt
                
                alpha = 0.7 if max_trials == 1 else 0.5
                ax.plot(times, control_inputs, '-', color=colors[trial_idx], 
                       alpha=alpha, linewidth=2, label=f'Trial {trial_idx+1}' if i == 0 else "")
            
            ax.set_ylabel('Control Force [N]')
            ax.set_title(f'Iteration {iteration_idx}')
            ax.grid(True, alpha=0.3)
            if i == len(iteration_indices) - 1:
                ax.set_xlabel('Time [s]')
            if i == 0 and max_trials > 1:
                ax.legend()
        
        plt.suptitle(f'Control Optimization Process: {combo_name.replace("_", " + ")}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'optimization_process_{combo_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the trajectory optimization comparison."""
    parser = argparse.ArgumentParser(description="Trajectory Optimization Methods Comparison")
    
    # Environment arguments
    parser.add_argument('--env', type=str, default='navigation2d', 
                       choices=['navigation2d', 'inverted_pendulum'],
                       help='Environment for comparison')
    parser.add_argument('--horizon_nodes', type=int, default=8,
                       help='Number of trajectory nodes')
    parser.add_argument('--horizon_samples', type=int, default=64,
                       help='Number of trajectory samples')
    parser.add_argument('--dt', type=float, default=0.02,
                       help='Time step')
    
    # Optimization arguments
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples per iteration')
    parser.add_argument('--num_iterations', type=int, default=50,
                       help='Number of optimization iterations')
    parser.add_argument('--temp_sample', type=float, default=0.1,
                       help='Temperature for sampling')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor for AVWBFO')
    
    # Noise scheduling arguments
    parser.add_argument('--noise_schedules', nargs='+', 
                       default=['constant', 'exponential_decay', 'hierarchical'],
                       help='Noise scheduling strategies to compare')
    
    # Noise sampling arguments
    parser.add_argument('--noise_sampler_type', type=str, default='lhs',
                       choices=['mc', 'lhs', 'halton'],
                       help='Type of noise sampler to use')
    parser.add_argument('--noise_distribution', type=str, default='normal',
                       choices=['normal', 'uniform'],
                       help='Distribution for noise sampling')
    parser.add_argument('--noise_sampler_seed', type=int, default=None,
                       help='Random seed for noise sampler')
    
    # Experiment arguments
    parser.add_argument('--num_trials', type=int, default=5,
                       help='Number of trials for statistical analysis')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--no_save_results', action='store_true',
                       help='Do not save results to files')
    parser.add_argument('--no_save_plots', action='store_true',
                       help='Do not save plots')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create experiment configuration
    config = ExperimentConfig(
        env_name=args.env,
        horizon_nodes=args.horizon_nodes,
        horizon_samples=args.horizon_samples,
        dt=args.dt,
        num_samples=args.num_samples,
        num_iterations=args.num_iterations,
        temp_sample=args.temp_sample,
        gamma=args.gamma,
        noise_schedule_types=args.noise_schedules,
        noise_sampler_type=args.noise_sampler_type,
        noise_distribution=args.noise_distribution,
        noise_sampler_seed=args.noise_sampler_seed,
        num_trials=args.num_trials,
        save_results=not args.no_save_results,
        save_plots=not args.no_save_plots,
        results_dir=args.results_dir,
        device=device
    )
    
    print("=== Trajectory Optimization Comparison ===")
    print(f"Environment: {config.env_name}")
    print(f"Device: {config.device}")
    print(f"Noise schedulers: {config.noise_schedule_types}")
    print(f"Noise sampler: {config.noise_sampler_type} ({config.noise_distribution})")
    print(f"Trials: {config.num_trials}")
    print(f"Iterations per trial: {config.num_iterations}")
    print(f"Samples per iteration: {config.num_samples}")
    
    # Initialize and run comparison
    comparison = TrajectoryOptimizationComparison(config)
    results = comparison.run_experiment()
    
    # Generate plots
    print("\nGenerating plots...")
    comparison.generate_plots(results)
    
    # Print summary
    print("\n=== Experiment Summary ===")
    for method_key, result in results['aggregate_results'].items():
        print(f"{method_key}:")
        print(f"  Final Cost: {result['cost_mean']:.4f} ± {result['cost_std']:.4f}")
        print(f"  Total Time: {result['time_mean']:.2f} ± {result['time_std']:.2f} s")
        print(f"  Convergence: {result['convergence_mean']:.1f} ± {result['convergence_std']:.1f} iterations")
    
    print(f"\nResults saved to: {config.results_dir}")


if __name__ == "__main__":
    main()
