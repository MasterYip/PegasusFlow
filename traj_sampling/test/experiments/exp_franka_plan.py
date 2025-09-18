#!/usr/bin/env python3
"""
Franka Arm Trajectory Optimization Experiment: Hierarchical Noise Scheduling Evaluation

This experiment compares different noise scheduling methods for Franka arm reach planning:
- Constant noise scheduling
- Linear noise scheduling  
- Hierarchical noise scheduling
- Adaptive noise scheduling

The experiment evaluates:
1. Task completion efficiency (steps to completion)
2. Solution quality (reward profiles)
3. Optimization convergence properties
4. Noise scheduling effectiveness

Results are saved with academic-quality plots and comprehensive data analysis.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import pickle
import multiprocessing as mp
import isaacgym
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Set plotting style for academic figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

from legged_gym.envs.base.legged_robot import LeggedRobot
from traj_sampling.config.trajectory_optimization_config import FrankaTrajectoryOptCfg
from traj_sampling.env_runner.legged_gym.franka_plan_envrunner import FrankaPlanEnvRunner


class AVWBFO_MC_Linear_CFG(FrankaTrajectoryOptCfg):
    class trajectory_opt(FrankaTrajectoryOptCfg.trajectory_opt):
        update_method = "avwbfo"
        noise_scheduler_type = 's2'
        noise_shape_fn = 'linear'
        noise_sampler_type = None  # Monte Carlo sampling

    class rl_warmstart(FrankaTrajectoryOptCfg.rl_warmstart):
        enable = False


class AVWBFO_LHS_Linear_CFG(FrankaTrajectoryOptCfg):

    class trajectory_opt(FrankaTrajectoryOptCfg.trajectory_opt):
        update_method = "avwbfo"
        noise_scheduler_type = 's2'
        noise_shape_fn = 'linear'
        noise_sampler_type = "lhs"

    class rl_warmstart(FrankaTrajectoryOptCfg.rl_warmstart):
        enable = False


class MPPI_MC_Linear_CFG(FrankaTrajectoryOptCfg):

    class trajectory_opt(FrankaTrajectoryOptCfg.trajectory_opt):
        update_method = "mppi"
        noise_scheduler_type = 's2'
        noise_shape_fn = 'linear'
        noise_sampler_type = None
    
    class rl_warmstart(FrankaTrajectoryOptCfg.rl_warmstart):
        enable = False


class MPPI_LHS_Linear_CFG(FrankaTrajectoryOptCfg):

    class trajectory_opt(FrankaTrajectoryOptCfg.trajectory_opt):
        update_method = "mppi"
        noise_scheduler_type = 's2'
        noise_shape_fn = 'linear'
        noise_sampler_type = 'lhs'
    
    class rl_warmstart(FrankaTrajectoryOptCfg.rl_warmstart):
        enable = False


def get_franka_commands():
    """Get predefined Franka arm commands for different reaching tasks."""
    commands = {
        'reach_forward': [0.6, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],   # Forward reach
        'reach_backward': [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], # Backward reach
        'reach_left': [0.4, 0.4, 0.5, 0.0, 0.0, 0.0, 1.0],      # Left reach
        'reach_right': [0.4, -0.4, 0.5, 0.0, 0.0, 0.0, 1.0],    # Right reach
        'reach_up': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],        # Upward reach
        'reach_down': [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],      # Downward reach
        'home': [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],           # Home position
    }
    return commands


def create_noise_scheduling_configs():
    """Create different noise scheduling configurations for comparison."""
    configs = {}

    configs['AVWBFO_MC_Linear'] = AVWBFO_MC_Linear_CFG()
    configs['AVWBFO_LHS_Linear'] = AVWBFO_LHS_Linear_CFG()
    configs['MPPI_MC_Linear'] = MPPI_MC_Linear_CFG()
    configs['MPPI_LHS_Linear'] = MPPI_LHS_Linear_CFG()

    return configs


def run_single_experiment_worker(method: str, config, command: List[float], 
                                seed: int, max_steps: int, args_dict: Dict, 
                                return_dict: Dict, key: str):
    """Worker function to run a single experiment in a separate process.
    
    Args:
        method: Noise scheduling method name
        config: Trajectory optimization config
        command: Command for the robot
        seed: Random seed
        max_steps: Maximum steps to run
        args_dict: Arguments dictionary
        return_dict: Shared dictionary for results
        key: Key for storing results
    """
    try:
        # Import inside worker to avoid Isaac Gym conflicts
        
        print(f"\n{'='*60}")
        print(f"Worker Process - Running experiment: {method} noise scheduling (seed={seed})")
        print(f"{'='*60}")
        
        # Create environment runner
        env_runner = FrankaPlanEnvRunner(
            task_name="franka_batch_rollout",
            num_main_envs=args_dict['num_envs'],
            num_rollout_per_main=args_dict['rollout_envs'],
            device=args_dict['device'],
            max_steps=max_steps,
            optimize_interval=1,
            seed=seed,
            headless=args_dict['headless'],
            enable_trajectory_optimization=True,
            trajectory_opt_config=config,
            command=command,
            debug_viz=args_dict['debug_viz'],
            experiment_name=f"{method}_noise_experiment"
        )
        
        # Set the experiment name
        env_runner.experiment_name = f"{method}_noise_seed_{seed}"
        
        # Run trajectory optimization
        results = env_runner.run_with_trajectory_optimization(seed=seed)
        
        # Add metadata
        results['noise_method'] = method
        results['seed'] = seed
        results['max_steps_allowed'] = max_steps
        
        # Print summary
        completion_stats = results['completion_stats']
        print(f"\nWorker Process - Experiment Summary - {method}:")
        print(f"  Completion rate: {completion_stats['completion_rate']:.2%}")
        print(f"  Mean completion steps: {completion_stats['mean_completion_steps']:.1f}")
        print(f"  Min completion steps: {completion_stats['min_completion_steps']}")
        print(f"  Max completion steps: {completion_stats['max_completion_steps']}")
        print(f"  Average reward: {results['average_reward']:.3f}")
        print(f"  Optimization time: {results['optimization_time']:.3f}s")
        
        # Clean up
        if hasattr(env_runner.env, 'end'):
            env_runner.env.end()
        del env_runner
        
        # Store results
        return_dict[key] = results
        
    except Exception as e:
        print(f"Error in worker process for {method}, seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return_dict[key] = None


def run_single_experiment(method: str, config, command: List[float], 
                         seed: int, max_steps: int, args) -> Dict[str, Any]:
    """Run a single experiment with specified noise scheduling method using multiprocessing.
    
    Args:
        method: Noise scheduling method name
        config: Trajectory optimization config
        command: Command for the robot
        seed: Random seed for reproducibility
        max_steps: Maximum steps to run
        args: Command line arguments
        
    Returns:
        Dictionary with experiment results
    """
    # Convert config to dictionary for serialization
    
    # Convert args to dictionary
    args_dict = {
        'num_envs': args.num_envs,
        'rollout_envs': args.rollout_envs,
        'device': args.device,
        'headless': args.headless,
        'debug_viz': args.debug_viz
    }
    
    # Use multiprocessing to run experiment in separate process
    manager = mp.Manager()
    return_dict = manager.dict()
    key = f"{method}_{seed}"
    
    # Create and start process
    p = mp.Process(target=run_single_experiment_worker, 
                   args=(method, config, command, seed, max_steps, 
                         args_dict, return_dict, key))
    p.start()
    p.join()
    
    # Get result
    if key in return_dict and return_dict[key] is not None:
        return return_dict[key]
    else:
        print(f"Failed to get result for {method}, seed {seed}")
        return None


def analyze_experiment_results(all_results: Dict[str, List[Dict]], 
                              output_dir: str) -> Dict[str, Any]:
    """Analyze and summarize experiment results across all methods and seeds.
    
    Args:
        all_results: Dictionary mapping noise methods to list of results
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary with analysis summary
    """
    print(f"\n{'='*60}")
    print("ANALYZING EXPERIMENT RESULTS")
    print(f"{'='*60}")
    
    analysis = {}
    
    for noise_method, results_list in all_results.items():
        method_analysis = {
            'completion_rates': [],
            'completion_steps_raw': [],  # Raw completion steps across all envs and runs
            'average_rewards': [],
            'optimization_times': [],
            'total_steps': [],
            'reward_histories': [],
            'completion_stats_list': []
        }
        
        for result in results_list:
            # Extract completion statistics
            completion_stats = result['completion_stats']
            method_analysis['completion_rates'].append(completion_stats['completion_rate'])
            
            # Extract raw completion steps from this run
            raw_completion_steps = completion_stats['completion_steps']  # This is the list of actual completion steps
            method_analysis['completion_steps_raw'].extend(raw_completion_steps)  # Add all steps to the raw list
            
            # Extract performance metrics
            method_analysis['average_rewards'].append(result['average_reward'])
            method_analysis['optimization_times'].append(result['optimization_time'])
            method_analysis['total_steps'].append(result['steps'])
            method_analysis['reward_histories'].append(result['rewards_history'])
            method_analysis['completion_stats_list'].append(completion_stats)
        
        # Calculate statistics across multiple seeds/runs
        method_summary = {}
        
        # For completion rates (percentage of envs that completed)
        completion_rates = method_analysis['completion_rates']
        if completion_rates:
            method_summary['completion_rates'] = {
                'mean': float(np.mean(completion_rates)),
                'std': float(np.std(completion_rates)),
                'min': float(np.min(completion_rates)),
                'max': float(np.max(completion_rates)),
                'median': float(np.median(completion_rates))
            }
        else:
            method_summary['completion_rates'] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        # For completion steps (raw steps from all completed environments across all runs)
        completion_steps_raw = method_analysis['completion_steps_raw']
        if completion_steps_raw:
            method_summary['completion_steps'] = {
                'mean': float(np.mean(completion_steps_raw)),
                'std': float(np.std(completion_steps_raw)),
                'min': float(np.min(completion_steps_raw)),
                'max': float(np.max(completion_steps_raw)),
                'median': float(np.median(completion_steps_raw))
            }
        else:
            method_summary['completion_steps'] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        # For other metrics that are already single values per run
        for metric in ['average_rewards', 'optimization_times']:
            values = method_analysis[metric]
            if values:
                method_summary[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
            else:
                method_summary[metric] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        analysis[noise_method] = {
            'raw_data': method_analysis,
            'summary': method_summary
        }
        
        # Print summary
        print(f"\n{noise_method.title()} Noise Scheduling:")
        print(f"  Completion Rate: {method_summary['completion_rates']['mean']:.3f} ± {method_summary['completion_rates']['std']:.3f}")
        print(f"  Completion Steps: {method_summary['completion_steps']['mean']:.1f} ± {method_summary['completion_steps']['std']:.1f}")
        print(f"  Average Reward: {method_summary['average_rewards']['mean']:.3f} ± {method_summary['average_rewards']['std']:.3f}")
        print(f"  Optimization Time: {method_summary['optimization_times']['mean']:.3f} ± {method_summary['optimization_times']['std']:.3f}")
        print(f"  Number of runs: {len(results_list)}")
        print(f"  Total completed environments: {len(completion_steps_raw)}")
    
    # Save analysis results
    analysis_file = os.path.join(output_dir, 'experiment_analysis.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists and numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        analysis_json = {}
        for method, data in analysis.items():
            analysis_json[method] = {
                'summary': convert_to_json_serializable(data['summary'])
            }
        json.dump(analysis_json, f, indent=2)
    
    # Save raw data as pickle
    raw_data_file = os.path.join(output_dir, 'experiment_raw_data.pkl')
    with open(raw_data_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nAnalysis saved to {analysis_file}")
    print(f"Raw data saved to {raw_data_file}")
    
    return analysis


def create_academic_plots(all_results: Dict[str, List[Dict]], 
                         analysis: Dict[str, Any],
                         output_dir: str):
    """Create academic-quality plots for the experiment results.
    
    Args:
        all_results: Dictionary mapping noise methods to list of results
        analysis: Analysis summary from analyze_experiment_results
        output_dir: Directory to save plots
    """
    print(f"\n{'='*60}")
    print("CREATING ACADEMIC PLOTS")
    print(f"{'='*60}")
    
    # Set up plotting parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Color palette for different methods
    method_colors = {
        'AVWBFO_MC_Linear': '#E74C3C',      # Red
        'AVWBFO_LHS_Linear': '#3498DB',        # Blue  
        'MPPI_MC_Linear': '#2ECC71',  # Green
        'MPPI_LHS_Linear': '#F39C12'       # Orange
    }
    
    # Plot 1: Completion Rate Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    methods = list(all_results.keys())
    completion_rates = [analysis[method]['summary']['completion_rates']['mean'] for method in methods]
    completion_stds = [analysis[method]['summary']['completion_rates']['std'] for method in methods]
    
    bars = ax1.bar(methods, completion_rates, yerr=completion_stds, 
                   color=[method_colors.get(m, '#BDC3C7') for m in methods],
                   capsize=5, alpha=0.8)
    ax1.set_ylabel('Task Completion Rate')
    ax1.set_title('Task Completion Rate by Noise Scheduling Method')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, rate, std in zip(bars, completion_rates, completion_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{rate:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Mean Steps to Completion
    completion_steps = [analysis[method]['summary']['completion_steps']['mean'] for method in methods]
    completion_steps_stds = [analysis[method]['summary']['completion_steps']['std'] for method in methods]
    
    bars2 = ax2.bar(methods, completion_steps, yerr=completion_steps_stds,
                    color=[method_colors.get(m, '#BDC3C7') for m in methods],
                    capsize=5, alpha=0.8)
    ax2.set_ylabel('Mean Steps to Task Completion')
    ax2.set_title('Efficiency: Steps Required for Task Completion')
    
    # Add value labels on bars
    for bar, steps, std in zip(bars2, completion_steps, completion_steps_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{steps:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'completion_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Reward Profiles Over Time
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in methods:
        reward_histories = analysis[method]['raw_data']['reward_histories']
        color = method_colors.get(method, '#BDC3C7')
        
        # Calculate mean and std across runs
        max_len = max(len(hist) for hist in reward_histories)
        padded_histories = []
        for hist in reward_histories:
            padded = hist + [hist[-1]] * (max_len - len(hist))  # Pad with last value
            padded_histories.append(padded)
        
        mean_rewards = np.mean(padded_histories, axis=-1).flatten()
        std_rewards = np.std(padded_histories, axis=-1).flatten()
        steps = np.arange(len(mean_rewards))
        
        ax.plot(steps, mean_rewards, label=f'{method.title()} Noise', 
                color=color, linewidth=2)
        ax.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                       alpha=0.2, color=color)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Step Reward')
    ax.set_title('Reward Profiles: Learning Curves by Noise Scheduling Method')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Distribution of Completion Times
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_completion_data = []
    for method in methods:
        completion_steps_raw = analysis[method]['raw_data']['completion_steps_raw']
        for steps in completion_steps_raw:
            all_completion_data.append({'Method': method.title(), 'Completion Steps': steps})
    
    if all_completion_data:
        df = pd.DataFrame(all_completion_data)
        sns.boxplot(data=df, x='Method', y='Completion Steps', ax=ax,
                   palette=[method_colors.get(m.lower(), '#BDC3C7') for m in df['Method'].unique()])
        ax.set_title('Distribution of Task Completion Times')
        ax.set_ylabel('Steps to Task Completion')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'completion_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Performance vs Computational Cost
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for method in methods:
        completion_rates = analysis[method]['raw_data']['completion_rates']
        opt_times = analysis[method]['raw_data']['optimization_times']
        color = method_colors.get(method, '#BDC3C7')
        
        ax.scatter(opt_times, completion_rates, label=f'{method.title()} Noise',
                  color=color, s=100, alpha=0.7)
        
        # Add error bars for mean
        mean_time = np.mean(opt_times)
        mean_rate = np.mean(completion_rates)
        std_time = np.std(opt_times)
        std_rate = np.std(completion_rates)
        
        ax.errorbar(mean_time, mean_rate, xerr=std_time, yerr=std_rate,
                   color=color, capsize=5, linewidth=2, capthick=2)
    
    ax.set_xlabel('Average Optimization Time (seconds)')
    ax.set_ylabel('Task Completion Rate')
    ax.set_title('Performance vs Computational Cost Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_cost.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Summary Statistics Table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table data
    table_data = []
    headers = ['Noise Method', 'Completion Rate', 'Mean Steps', 'Avg Reward', 'Opt Time (s)']
    
    for method in methods:
        summary = analysis[method]['summary']
        row = [
            method.title(),
            f"{summary['completion_rates']['mean']:.3f} ± {summary['completion_rates']['std']:.3f}",
            f"{summary['completion_steps']['mean']:.1f} ± {summary['completion_steps']['std']:.1f}",
            f"{summary['average_rewards']['mean']:.3f} ± {summary['average_rewards']['std']:.3f}",
            f"{summary['optimization_times']['mean']:.3f} ± {summary['optimization_times']['std']:.3f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    ax.set_title('Experimental Results Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Academic plots saved to {output_dir}")
    print("Generated plots:")
    print("  - completion_metrics.png: Task completion rates and efficiency")
    print("  - reward_profiles.png: Learning curves by method")
    print("  - completion_distribution.png: Distribution of completion times")
    print("  - performance_vs_cost.png: Performance vs computational cost")
    print("  - summary_table.png: Summary statistics table")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Franka Arm Noise Scheduling Experiment")
    
    # Experiment parameters
    parser.add_argument('--task_command', type=str, default='reach_backward',
                        choices=['reach_forward', 'reach_backward', 'reach_left', 
                                'reach_right', 'reach_up', 'reach_down', 'home'],
                        help='Franka arm reaching task to test')
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='Number of random seeds to test per method')
    parser.add_argument('--num_envs', type=int, default=30,
                        help='Number of parallel environments')
    parser.add_argument('--rollout_envs', type=int, default=64,
                        help='Number of rollout environments per main environment')
    parser.add_argument('--max_steps', type=int, default=150,
                        help='Maximum steps per experiment run')
    
    # Noise scheduling methods to test
    parser.add_argument('--methods', nargs='+', 
                        choices=['AVWBFO_MC_Linear', 'AVWBFO_LHS_Linear', 'MPPI_MC_Linear', 'MPPI_LHS_Linear'],
                        default=['AVWBFO_MC_Linear', 'AVWBFO_LHS_Linear', 'MPPI_MC_Linear', 'MPPI_LHS_Linear'],
                        # default=['AVWBFO_MC_Linear', 'AVWBFO_LHS_Linear', 'MPPI_MC_Linear'],
                        help='Noise scheduling methods to compare')
    
    # Environment parameters
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run in headless mode')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for computation')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated if not specified)')
    parser.add_argument('--save_data', action='store_true', default=True,
                        help='Save experimental data')
    
    # Misc
    parser.add_argument('--debug_viz', action='store_true', default=True,
                    help='Enable debug visualization')
    return parser.parse_args()


def main():
    """Run the Franka arm noise scheduling experiment."""
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    args = parse_arguments()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./franka_noise_experiment_{args.task_command}_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save experiment configuration
    config_file = os.path.join(args.output_dir, 'experiment_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n{'='*80}")
    print("FRANKA ARM NOISE SCHEDULING EXPERIMENT")
    print(f"{'='*80}")
    print(f"Task: {args.task_command}")
    print(f"Methods: {args.methods}")
    print(f"Seeds per method: {args.num_seeds}")
    print(f"Environments: {args.num_envs}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using multiprocessing to avoid Isaac Gym conflicts")
    print(f"{'='*80}")
    
    # Get task command
    commands = get_franka_commands()
    command = commands[args.task_command]
    
    # Get noise scheduling configurations
    noise_configs = create_noise_scheduling_configs()
    
    # Run experiments using multiprocessing
    all_results = {}
    
    for method in args.methods:
        if method not in noise_configs:
            print(f"Warning: Unknown noise method {method}, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"TESTING {method.upper()} NOISE SCHEDULING")
        print(f"{'='*60}")
        
        method_results = []
        config = noise_configs[method]
        
        for seed in range(args.num_seeds):
            print(f"\nStarting experiment: {method} with seed {seed}")
            
            # Run experiment in separate process
            result = run_single_experiment(method, config, command, seed, args.max_steps, args)
            
            if result is not None:
                method_results.append(result)
                print(f"Completed experiment: {method} with seed {seed}")
            else:
                print(f"Failed experiment: {method} with seed {seed}")
            
            # Small delay between experiments to ensure clean process cleanup
            time.sleep(1)
        
        if method_results:
            all_results[method] = method_results
            print(f"Completed {len(method_results)} runs for {method} method")
        else:
            print(f"No successful runs for {method} method")
    
    if not all_results:
        print("No successful experiments completed!")
        return
    
    # Analyze results
    print(f"\nAll experiments completed. Analyzing results...")
    analysis = analyze_experiment_results(all_results, args.output_dir)
    
    # Create plots
    create_academic_plots(all_results, analysis, args.output_dir)
    
    # Save final summary
    summary_file = os.path.join(args.output_dir, 'experiment_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Franka Arm Noise Scheduling Experiment Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Task: {args.task_command}\n")
        f.write(f"Methods tested: {list(all_results.keys())}\n")
        f.write(f"Seeds per method: {args.num_seeds}\n")
        f.write(f"Environments: {args.num_envs}\n")
        f.write(f"Max steps: {args.max_steps}\n\n")
        
        f.write("Results Summary:\n")
        f.write("-" * 20 + "\n")
        for method, method_analysis in analysis.items():
            summary = method_analysis['summary']
            f.write(f"\n{method.title()} Noise Scheduling:\n")
            f.write(f"  Completion Rate: {summary['completion_rates']['mean']:.3f} ± {summary['completion_rates']['std']:.3f}\n")
            f.write(f"  Mean Steps: {summary['completion_steps']['mean']:.1f} ± {summary['completion_steps']['std']:.1f}\n")
            f.write(f"  Average Reward: {summary['average_rewards']['mean']:.3f} ± {summary['average_rewards']['std']:.3f}\n")
            f.write(f"  Optimization Time: {summary['optimization_times']['mean']:.3f} ± {summary['optimization_times']['std']:.3f}\n")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
