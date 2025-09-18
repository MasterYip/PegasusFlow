#!/usr/bin/env python3
"""
Robot Navigation Trajectory Optimization Comparison Experiment

This experiment compares different trajectory optimization configurations for robot navigation tasks:
- Level 1: Different trajectory optimization methods/configurations
- Level 2: Different navigation tasks/scenes

The experiment evaluates:
1. Navigation success rate
2. Goal reaching efficiency (steps to completion)
3. Solution quality (reward profiles)
4. Optimization convergence properties
5. Performance across different task scenarios

Results are saved with academic-quality plots and comprehensive data analysis.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import multiprocessing as mp
import isaacgym
import torch
from datetime import datetime
from typing import Dict, List, Any

# Set plotting style for academic figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

from legged_gym.envs.base.legged_robot import LeggedRobot
from traj_sampling.config.trajectory_optimization_config import (
    TrajectoryOptimizationCfg,
    ElSpiderAirTrajectoryOptCfg,
    AnymalCTrajectoryOptCfg,
    Go2TrajectoryOptCfg,
    CassieTrajectoryOptCfg,
    FrankaTrajectoryOptCfg
)
from traj_sampling.env_runner.legged_gym.robot_nav_envrunner import RobotNavEnvRunner


# ============================================================================
# TRAJECTORY OPTIMIZATION CONFIGURATIONS
# ============================================================================

class ElAir_VanillaRL_CFG(ElSpiderAirTrajectoryOptCfg):
    class trajectory_opt(ElSpiderAirTrajectoryOptCfg.trajectory_opt):
        update_method = "avwbfo"
        noise_scheduler_type = 's2'
        noise_shape_fn = 'linear'
        noise_base_scale = 0.0  # Not make changes to RL policy generated traj
        noise_sampler_type = None

    class rl_warmstart(ElSpiderAirTrajectoryOptCfg.rl_warmstart):
        enable = True


class ElAir_AVWBFO_RL_CFG(ElSpiderAirTrajectoryOptCfg):

    class trajectory_opt(ElSpiderAirTrajectoryOptCfg.trajectory_opt):
        update_method = "avwbfo"
        noise_scheduler_type = 's2'
        noise_shape_fn = 'linear'
        noise_sampler_type = 'lhs'

    class rl_warmstart(ElSpiderAirTrajectoryOptCfg.rl_warmstart):
        enable = True


class ElAir_AVWBFO_CFG(ElSpiderAirTrajectoryOptCfg):

    class trajectory_opt(ElSpiderAirTrajectoryOptCfg.trajectory_opt):
        update_method = "avwbfo"
        noise_scheduler_type = 's2'
        noise_shape_fn = 'linear'
        noise_sampler_type = 'lhs'

    class rl_warmstart(ElSpiderAirTrajectoryOptCfg.rl_warmstart):
        enable = False


class ElAir_MPPI_RL_CFG(ElSpiderAirTrajectoryOptCfg):
    """MPPI with Latin Hypercube sampling and linear noise scheduling."""

    class trajectory_opt(ElSpiderAirTrajectoryOptCfg.trajectory_opt):
        update_method = "mppi"
        noise_scheduler_type = 's2'
        noise_shape_fn = 'linear'
        noise_sampler_type = 'lhs'

    class rl_warmstart(ElSpiderAirTrajectoryOptCfg.rl_warmstart):
        enable = True


class ElAir_MPPI_CFG(ElSpiderAirTrajectoryOptCfg):
    """MPPI with Latin Hypercube sampling and linear noise scheduling."""

    class trajectory_opt(ElSpiderAirTrajectoryOptCfg.trajectory_opt):
        update_method = "mppi"
        noise_scheduler_type = 's2'
        noise_shape_fn = 'linear'
        noise_sampler_type = 'lhs'

    class rl_warmstart(ElSpiderAirTrajectoryOptCfg.rl_warmstart):
        enable = False


def create_trajectory_optimization_configs(robot_type: str) -> Dict[str, TrajectoryOptimizationCfg]:
    """Create different trajectory optimization configurations for comparison."""
    configs = {}

    if robot_type == 'elspider_air':
        configs['VanillaRL'] = ElAir_VanillaRL_CFG()
        configs['AVWBFO_RL'] = ElAir_AVWBFO_RL_CFG()
        configs['AVWBFO'] = ElAir_AVWBFO_CFG()
        configs['MPPI_RL'] = ElAir_MPPI_RL_CFG()
        configs['MPPI'] = ElAir_MPPI_CFG()

    elif robot_type == 'anymal_c':
        # Similar configurations for Anymal C (would need to create derived classes)
        base_config = AnymalCTrajectoryOptCfg()
        configs['AVWBFO_MC_Linear'] = base_config
        configs['AVWBFO_LHS_Linear'] = base_config
        configs['MPPI_MC_Linear'] = base_config
        configs['MPPI_LHS_Linear'] = base_config

    else:
        # Default configurations for other robots
        base_config = TrajectoryOptimizationCfg()
        configs['AVWBFO_Default'] = base_config
        configs['MPPI_Default'] = base_config

    return configs


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_single_experiment_worker(method: str, config,
                                 robot_type: str, task_name: str, seed: int, args_dict: Dict,
                                 return_dict: Dict, key: str):
    """Worker function to run a single experiment in a separate process."""
    try:
        print(f"\n{'='*60}")
        print(f"Worker Process - Running experiment: {method} (seed={seed}) on task: {task_name}")
        print(f"{'='*60}")

        # Use provided task_name directly (was previously derived from robot_type)
        # task_name = f"{robot_type}_nav" if robot_type == 'elspider_air' else f"{robot_type}_batch_rollout"

        # Create environment runner
        # FIXME: rollout_envs=0 will cause some issues (e.g. start/goal gen not correct)
        rollout_envs = 1 if config.rl_warmstart.enable and config.trajectory_opt.noise_base_scale == 0.0 else args_dict[
            'rollout_envs']
        env_runner = RobotNavEnvRunner(
            task_name=task_name,
            num_main_envs=args_dict['num_envs'],
            num_rollout_per_main=rollout_envs,
            device=args_dict['device'],
            max_steps=args_dict['max_steps'],
            optimize_interval=1,
            seed=seed,
            headless=args_dict['headless'],
            enable_trajectory_optimization=True,
            trajectory_opt_config=config,
            command=[1.0, 0.0, 0.0, 0.0],  # Default forward command
            debug_viz=args_dict['debug_viz'],
            debug_viz_origins=args_dict['debug_viz_origins'],
            experiment_name=f"{method}_experiment",
            save_step_snapshots=args_dict['save_step_snapshots'],
            snapshot_interval=args_dict['snapshot_interval'],
            snapshot_output_dir=args_dict['snapshot_output_dir']
        )

        # Set the experiment name
        env_runner.experiment_name = f"{method}_seed_{seed}_{task_name}"

        # Run trajectory optimization
        if config.rl_warmstart.enable and config.trajectory_opt.noise_base_scale == 0.0:
            results = env_runner.run(seed=seed)
        else:
            results = env_runner.run_with_trajectory_optimization(seed=seed)

        # Add metadata (include task_name for clarity)
        results['trajectory_method'] = method
        results['robot_type'] = robot_type
        results['task_name'] = task_name
        results['seed'] = seed
        results['max_steps_allowed'] = args_dict['max_steps']

        # Extract navigation-specific metrics
        if hasattr(env_runner.env, 'get_goal_reached_status'):
            goal_reached = env_runner.env.get_goal_reached_status()
            if hasattr(env_runner.env, 'get_distance_to_goal'):
                distances = env_runner.env.get_distance_to_goal()
                results['final_distances'] = distances.cpu().numpy().tolist()

            # Calculate navigation success rate
            main_env_indices = env_runner.env.main_env_indices
            main_env_goal_reached = goal_reached
            results['navigation_success_rate'] = float(torch.mean(main_env_goal_reached.float()))

            # Find first successful completion step for each main environment
            completion_steps = []
            completion_map = {}
            for env_idx, step in results['per_env_completion_steps']:
                if env_idx not in completion_map:  # Only record first completion
                    completion_map[env_idx] = step
            
            # Extract completion steps for main environments
            for i, main_idx in enumerate(main_env_indices):
                main_idx_item = main_idx.item() if hasattr(main_idx, 'item') else main_idx
                if main_idx_item in completion_map:
                    completion_steps.append(completion_map[main_idx_item])

            if completion_steps:
                results['mean_completion_steps'] = float(np.mean(completion_steps))
                results['min_completion_steps'] = int(np.min(completion_steps))
                results['max_completion_steps'] = int(np.max(completion_steps))
            else:
                results['mean_completion_steps'] = args_dict['max_steps']
                results['min_completion_steps'] = args_dict['max_steps']
                results['max_completion_steps'] = args_dict['max_steps']

        # Print summary
        print(f"\nWorker Process - Experiment Summary - {method}:")
        print(f"  Navigation success rate: {results.get('navigation_success_rate', 0):.2%}")
        print(f"  Mean completion steps: {results.get('mean_completion_steps', 0):.1f}")
        print(f"  Average reward: {results.get('average_reward', 0):.3f}")
        print(f"  Optimization time: {results.get('optimization_time', 0):.3f}s")

        # Clean up
        if hasattr(env_runner.env, 'end'):
            env_runner.env.end()
        del env_runner

        # Store results
        return_dict[key] = results

    except Exception as e:
        print(f"Error in worker process for {method}, seed {seed}, task {task_name}: {e}")
        import traceback
        traceback.print_exc()
        return_dict[key] = None


def run_single_experiment(method: str, config,
                          robot_type: str, task_name: str, seed: int, args) -> Dict[str, Any]:
    """Run a single experiment with specified method using multiprocessing."""
    # Convert args to dictionary
    args_dict = {
        'num_envs': args.num_envs,
        'rollout_envs': args.rollout_envs,
        'device': args.device,
        'headless': args.headless,
        'debug_viz': args.debug_viz,
        'debug_viz_origins': args.debug_viz_origins,
        'max_steps': args.max_steps,
        'save_step_snapshots': args.save_step_snapshots,
        'snapshot_interval': args.snapshot_interval,
        'snapshot_output_dir': args.snapshot_output_dir
    }

    # Use multiprocessing to run experiment in separate process
    manager = mp.Manager()
    return_dict = manager.dict()
    key = f"{task_name}_{method}_{seed}"

    # Create and start process
    p = mp.Process(target=run_single_experiment_worker,
                   args=(method, config, robot_type, task_name, seed,
                         args_dict, return_dict, key))
    p.start()
    p.join()

    # Get result
    if key in return_dict and return_dict[key] is not None:
        return return_dict[key]
    else:
        print(f"Failed to get result for {method}, seed {seed}, task {task_name}")
        return None


# ============================================================================
# DATA ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_experiment_results(all_results: Dict[str, List[Dict]],
                               output_dir: str) -> Dict[str, Any]:
    """Analyze and summarize experiment results across all methods."""
    print(f"\n{'='*60}")
    print("ANALYZING EXPERIMENT RESULTS")
    print(f"{'='*60}")

    analysis = {}

    for method, results_list in all_results.items():
        method_analysis = {
            'navigation_success_rates': [],
            'completion_steps': [],
            'average_rewards': [],
            'optimization_times': [],
            'total_steps': [],
            'reward_histories': [],
            'final_distances': []
        }

        for result in results_list:
            # Extract navigation-specific metrics
            method_analysis['navigation_success_rates'].append(result.get('navigation_success_rate', 0.0))
            method_analysis['completion_steps'].append(result.get('mean_completion_steps', result.get('max_steps_allowed', 0)))
            method_analysis['average_rewards'].append(result.get('average_reward', 0.0))
            method_analysis['optimization_times'].append(result.get('optimization_time', 0.0))
            method_analysis['total_steps'].append(result.get('steps', 0))
            method_analysis['reward_histories'].append(result.get('rewards_history', []))
            method_analysis['final_distances'].extend(result.get('final_distances', []))

        # Calculate statistics for this method
        method_summary = {}

        # Navigation success rates
        success_rates = method_analysis['navigation_success_rates']
        if success_rates:
            method_summary['navigation_success_rate'] = {
                'mean': float(np.mean(success_rates)),
                'std': float(np.std(success_rates)),
                'min': float(np.min(success_rates)),
                'max': float(np.max(success_rates)),
                'median': float(np.median(success_rates))
            }
        else:
            method_summary['navigation_success_rate'] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}

        # Completion steps - collect all individual completion steps, not means
        all_completion_steps = []
        for result in results_list:
            # Extract individual completion steps from per_env_completion_steps if available
            if 'per_env_completion_steps' in result:
                completion_steps_data = [step for _, step in result['per_env_completion_steps']]
                all_completion_steps.extend(completion_steps_data)
            # Fallback: if only mean is available, we can't compute proper std
            elif 'mean_completion_steps' in result and result.get('navigation_success_rate', 0) > 0:
                # This is a fallback - ideally we should have individual steps
                all_completion_steps.append(result['mean_completion_steps'])
        
        if all_completion_steps:
            method_summary['completion_steps'] = {
                'mean': float(np.mean(all_completion_steps)),
                'std': float(np.std(all_completion_steps)) if len(all_completion_steps) > 1 else 0.0,
                'min': float(np.min(all_completion_steps)),
                'max': float(np.max(all_completion_steps)),
                'median': float(np.median(all_completion_steps)),
                'count': len(all_completion_steps)
            }
        else:
            method_summary['completion_steps'] = {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'count': 0
            }

        # Other metrics
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

        analysis[method] = {
            'raw_data': method_analysis,
            'summary': method_summary
        }

        # Print summary for this method
        print(f"\n{method.title()}:")
        print(
            f"  Navigation Success Rate: {method_summary['navigation_success_rate']['mean']:.3f} ± {method_summary['navigation_success_rate']['std']:.3f}")
        print(
            f"  Mean Completion Steps: {method_summary['completion_steps']['mean']:.1f} ± {method_summary['completion_steps']['std']:.1f}")
        print(
            f"  Average Reward: {method_summary['average_rewards']['mean']:.3f} ± {method_summary['average_rewards']['std']:.3f}")
        print(
            f"  Optimization Time: {method_summary['optimization_times']['mean']:.3f} ± {method_summary['optimization_times']['std']:.3f}")
        print(f"  Number of runs: {len(results_list)}")

    # Save analysis results
    analysis_file = os.path.join(output_dir, 'experiment_analysis.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
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
        for method, method_data in analysis.items():
            analysis_json[method] = {
                'summary': convert_to_json_serializable(method_data['summary'])
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
    """Create academic-quality plots for the experiment results."""
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
        'VanillaRL': '#E74C3C',      # Red
        'AVWBFO_RL': '#3498DB',     # Blue
        'AVWBFO': '#2ECC71',        # Green
        'MPPI_RL': '#F39C12',       # Orange
        'MPPI': '#9B59B6',   # Purple
        'Others': '#E67E22'        # Dark Orange
    }

    # Get all methods
    methods = list(all_results.keys())

    # Plot 1: Navigation Success Rate Comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    success_rates = [analysis[method]['summary']['navigation_success_rate']['mean'] for method in methods]
    success_stds = [analysis[method]['summary']['navigation_success_rate']['std'] for method in methods]

    bars = ax.bar(methods, success_rates, yerr=success_stds,
                  color=[method_colors.get(m, '#BDC3C7') for m in methods],
                  capsize=5, alpha=0.8)
    ax.set_ylabel('Navigation Success Rate')
    ax.set_title('Navigation Success Rate by Trajectory Optimization Method')
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, rate, std in zip(bars, success_rates, success_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{rate:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'navigation_success_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Completion Steps Comparison (Completed vs All Environments)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate completion steps for completed environments only
    completed_only_steps = []
    completed_only_stds = []
    
    # Calculate completion steps for all environments (non-completed = max_steps)
    all_envs_steps = []
    all_envs_stds = []
    
    for method in methods:
        # Get max_steps from the first result (assuming all have same max_steps)
        max_steps = None
        for result in all_results[method]:
            if 'max_steps_allowed' in result:
                max_steps = result['max_steps_allowed']
                break
        if max_steps is None:
            max_steps = 200  # fallback default
        
        # Collect all individual completion steps and calculate for both scenarios
        method_completed_steps = []
        method_all_steps = []
        
        for result in all_results[method]:
            # Extract individual completion steps from per_env_completion_steps if available
            if 'per_env_completion_steps' in result:
                completion_steps_data = [step for _, step in result['per_env_completion_steps']]
                method_completed_steps.extend(completion_steps_data)
                
                # For all environments calculation, need to know total envs and successful ones
                num_main_envs = result.get('num_main_envs')  # fallback
                num_completed = len(completion_steps_data)
                num_not_completed = num_main_envs - num_completed
                
                # Add completed steps
                method_all_steps.extend(completion_steps_data)
                # Add max_steps for non-completed environments
                method_all_steps.extend([max_steps] * num_not_completed)
            
            # Fallback: if only mean is available
            elif 'mean_completion_steps' in result and result.get('navigation_success_rate', 0) > 0:
                method_completed_steps.append(result['mean_completion_steps'])
                
                # Estimate all environments based on success rate
                success_rate = result.get('navigation_success_rate', 0)
                num_main_envs = result.get('num_main_envs')
                num_completed = int(success_rate * num_main_envs)
                num_not_completed = num_main_envs - num_completed
                
                method_all_steps.extend([result['mean_completion_steps']] * num_completed)
                method_all_steps.extend([max_steps] * num_not_completed)
        
        # Calculate statistics
        if method_completed_steps:
            completed_only_steps.append(float(np.mean(method_completed_steps)))
            completed_only_stds.append(float(np.std(method_completed_steps)) if len(method_completed_steps) > 1 else 0.0)
        else:
            completed_only_steps.append(0.0)
            completed_only_stds.append(0.0)
        
        if method_all_steps:
            all_envs_steps.append(float(np.mean(method_all_steps)))
            all_envs_stds.append(float(np.std(method_all_steps)) if len(method_all_steps) > 1 else 0.0)
        else:
            all_envs_steps.append(max_steps)
            all_envs_stds.append(0.0)

    # Set up bar positions
    x = np.arange(len(methods))
    width = 0.35  # width of bars

    # Create bars
    bars1 = ax.bar(x - width/2, completed_only_steps, width, yerr=completed_only_stds,
                   color=[method_colors.get(m, '#BDC3C7') for m in methods],
                   capsize=5, alpha=0.8, label='Completed Environments Only')
    
    bars2 = ax.bar(x + width/2, all_envs_steps, width, yerr=all_envs_stds,
                   color=[method_colors.get(m, '#BDC3C7') for m in methods],
                   capsize=5, alpha=0.5, label='All Environments (Failed = Max Steps)')

    ax.set_ylabel('Mean Steps to Completion')
    ax.set_title('Task Completion Efficiency by Method')
    ax.set_xlabel('Trajectory Optimization Method')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, ha='center')
    ax.legend()

    # Add value labels on bars
    for i, (bar1, bar2, steps1, std1, steps2, std2) in enumerate(zip(
        bars1, bars2, completed_only_steps, completed_only_stds, all_envs_steps, all_envs_stds)):
        
        # Labels for completed only bars
        height1 = bar1.get_height()
        if height1 > 0:  # Only show label if there were completions
            ax.text(bar1.get_x() + bar1.get_width()/2., height1 + std1 + 1,
                    f'{steps1:.1f}±{std1:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Labels for all environments bars
        height2 = bar2.get_height()
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + std2 + 1,
                f'{steps2:.1f}±{std2:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'completion_steps_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Summary Statistics Table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')

    # Create summary table data
    table_data = []
    headers = ['Method', 'Success Rate', 'Mean Steps', 'Avg Reward', 'Opt Time (s)']

    for method in methods:
        summary = analysis[method]['summary']
        row = [
            method.replace('_', ' ').title(),
            f"{summary['navigation_success_rate']['mean']:.3f} ± {summary['navigation_success_rate']['std']:.3f}",
            f"{summary['completion_steps']['mean']:.1f} ± {summary['completion_steps']['std']:.1f}",
            f"{summary['average_rewards']['mean']:.3f} ± {summary['average_rewards']['std']:.3f}",
            f"{summary['optimization_times']['mean']:.3f} ± {summary['optimization_times']['std']:.3f}"
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')

    ax.set_title('Robot Navigation Experiment Results Summary', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Academic plots saved to {output_dir}")
    print("Generated plots:")
    print("  - navigation_success_comparison.png: Success rates by method")
    print("  - completion_steps_comparison.png: Task completion efficiency")
    print("  - summary_table.png: Summary statistics table")


# ============================================================================
# ARGUMENT PARSING AND MAIN EXECUTION
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Robot Navigation Trajectory Optimization Experiment")

    # Robot and experiment parameters
    parser.add_argument('--robot', type=str, default='elspider_air',
                        choices=['elspider_air', 'anymal_c', 'go2', 'cassie', 'franka'],
                        help='Robot type to use')
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='Number of random seeds to test per method')

    # New: list of task names to run (overrides automatic derivation)
    parser.add_argument('--tasks', nargs='+', default=["elair_barrier_nav"],
                        help='List of task names to run (if not provided, task is derived from robot)')

    # Environment parameters
    parser.add_argument('--num_envs', type=int, default=20,
                        help='Number of main environments')
    parser.add_argument('--rollout_envs', type=int, default=128,
                        help='Number of rollout environments per main environment')
    parser.add_argument('--max_steps', type=int, default=300,
                        help='Maximum steps per experiment run')

    # Trajectory optimization methods to test
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Specific trajectory optimization methods to test (default: all available)')

    # Environment parameters
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Run in headless mode')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for computation')
    parser.add_argument('--debug_viz', action='store_true', default=True,
                        help='Enable debug visualization')
    parser.add_argument('--debug_viz_origins', action='store_true', default=False,
                        help='Enable visualization of environment origins')

    # Snapshot and recording options
    parser.add_argument('--save_step_snapshots', action='store_true', default=True,
                        help='Save viewer snapshots at each step (requires GUI mode)')
    parser.add_argument('--snapshot_interval', type=int, default=1,
                        help='Interval between snapshots (save every N steps)')
    parser.add_argument('--snapshot_output_dir', type=str, default=None,
                        help='Directory for saving snapshots (auto-generated if not specified)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated if not specified)')

    return parser.parse_args()


def main():
    """Run the robot navigation trajectory optimization comparison experiment."""
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    args = parse_arguments()

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./robot_nav_experiment_{args.robot}_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Create snapshot output directory if snapshots are enabled
    if args.save_step_snapshots:
        if args.headless:
            print("Warning: Snapshots require GUI mode. Disabling snapshots for headless run.")
            args.save_step_snapshots = False
        else:
            if args.snapshot_output_dir is None:
                args.snapshot_output_dir = os.path.join(args.output_dir, "step_snapshots")
            os.makedirs(args.snapshot_output_dir, exist_ok=True)
            print(f"Step snapshots will be saved to: {args.snapshot_output_dir}")
            print(f"Snapshot interval: every {args.snapshot_interval} steps")

    # Save experiment configuration
    config_file = os.path.join(args.output_dir, 'experiment_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*80}")
    print("ROBOT NAVIGATION TRAJECTORY OPTIMIZATION EXPERIMENT")
    print(f"{'='*80}")
    print(f"Robot: {args.robot}")
    print(f"Seeds per method: {args.num_seeds}")
    print(f"Environments: {args.num_envs}")
    print(f"Rollout environments per main: {args.rollout_envs}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"Headless mode: {args.headless}")
    if args.save_step_snapshots and not args.headless:
        print(f"Step snapshots: Enabled (every {args.snapshot_interval} steps)")
        print(f"Snapshot output directory: {args.snapshot_output_dir}")
    else:
        print(f"Step snapshots: Disabled")
    print(f"Using multiprocessing to avoid Isaac Gym conflicts")
    print(f"{'='*80}")

    # Get trajectory optimization configurations
    all_configs = create_trajectory_optimization_configs(args.robot)
    if args.methods:
        configs = {name: config for name, config in all_configs.items() if name in args.methods}
        if not configs:
            print(f"Warning: No valid methods found in {args.methods}")
            configs = all_configs
    else:
        configs = all_configs

    print(f"Trajectory optimization methods: {list(configs.keys())}")

    # Determine tasks to run
    if args.tasks is None:
        # default behavior: derive single task from robot type
        if args.robot == 'elspider_air':
            tasks = [f"{args.robot}_nav"]
        else:
            tasks = [f"{args.robot}_batch_rollout"]
    else:
        tasks = args.tasks

    print(f"Tasks to run: {tasks}")

    # Run experiments using multiprocessing
    all_results = {}

    total_experiments = len(configs) * args.num_seeds * len(tasks)
    current_experiment = 0

    for task_name in tasks:
        for method_name, config in configs.items():
            composite_name = f"{task_name}_{method_name}"
            print(f"\n{'='*60}")
            print(f"TESTING {method_name.upper()} METHOD ON TASK {task_name}")
            print(f"{'='*60}")

            all_results[composite_name] = []

            for seed in range(args.num_seeds):
                current_experiment += 1
                print(f"\nProgress: {current_experiment}/{total_experiments}")
                print(f"Starting experiment: {method_name} with seed {seed} on task {task_name}")

                # Run experiment in separate process (pass task_name)
                result = run_single_experiment(method_name, config, args.robot, task_name, seed, args)

                if result is not None:
                    all_results[composite_name].append(result)
                    print(f"Completed experiment: {method_name} with seed {seed} on task {task_name}")
                else:
                    print(f"Failed experiment: {method_name} with seed {seed} on task {task_name}")

                # Small delay between experiments to ensure clean process cleanup
                time.sleep(1)

            print(f"Completed {len(all_results[composite_name])} runs for {composite_name}")

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
        f.write(f"Robot Navigation Trajectory Optimization Experiment Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Robot: {args.robot}\n")
        f.write(f"Methods tested: {list(all_results.keys())}\n")
        f.write(f"Seeds per method: {args.num_seeds}\n")
        f.write(f"Environments: {args.num_envs}\n")
        f.write(f"Rollout environments per main: {args.rollout_envs}\n")
        f.write(f"Max steps: {args.max_steps}\n")
        f.write(f"Step snapshots: {'Enabled' if args.save_step_snapshots else 'Disabled'}\n")
        if args.save_step_snapshots:
            f.write(f"Snapshot interval: every {args.snapshot_interval} steps\n")
            f.write(f"Snapshot output directory: {args.snapshot_output_dir}\n")
        f.write("\n")

        f.write("Results Summary:\n")
        f.write("-" * 30 + "\n")
        for method, method_results in analysis.items():
            summary = method_results['summary']
            f.write(f"\n{method.title()} Method:\n")
            f.write(
                f"  Success Rate: {summary['navigation_success_rate']['mean']:.3f} ± {summary['navigation_success_rate']['std']:.3f}\n")
            f.write(f"  Mean Steps: {summary['completion_steps']['mean']:.1f} ± {summary['completion_steps']['std']:.1f}\n")
            f.write(f"  Avg Reward: {summary['average_rewards']['mean']:.3f} ± {summary['average_rewards']['std']:.3f}\n")
            f.write(
                f"  Opt Time: {summary['optimization_times']['mean']:.3f} ± {summary['optimization_times']['std']:.3f}s\n")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")


def load_data_and_plot(output_dir: str):
    """Load previously saved experiment data and regenerate plots."""
    raw_data_file = os.path.join(output_dir, 'experiment_raw_data.pkl')
    if not os.path.exists(raw_data_file):
        print(f"Raw data file not found: {raw_data_file}")
        return

    with open(raw_data_file, 'rb') as f:
        all_results = pickle.load(f)

    analysis_file = os.path.join(output_dir, 'experiment_analysis.json')
    if not os.path.exists(analysis_file):
        print(f"Analysis file not found: {analysis_file}")
        return

    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    create_academic_plots(all_results, analysis, output_dir)
    print(f"Plots regenerated and saved to: {output_dir}")


if __name__ == "__main__":
    # main()
    load_data_and_plot("PredictiveDiffusionPlanner_Dev/doc/records/20250902 Exp2ElAirBarrierNav/new_rew2")
