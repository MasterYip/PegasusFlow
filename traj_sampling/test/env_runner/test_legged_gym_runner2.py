#!/usr/bin/env python3

"""
Test script for LeggedGym Environment Runner v2 with trajectory optimization.

This script demonstrates the improved trajectory optimization implementation that:
1. Uses batch rollout environments directly (no traj_grad_sampling environments)
2. Rigorously follows the original trajectory optimization algorithm
3. Features clean separation between environment and optimization logic
4. Uses standalone trajectory optimization configuration
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import isaacgym
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from traj_sampling.config.trajectory_optimization_config import (
    TrajectoryOptimizationCfg,
    ElSpiderAirTrajectoryOptCfg,
    AnymalCTrajectoryOptCfg,
    Go2TrajectoryOptCfg,
    CassieTrajectoryOptCfg,
    FrankaTrajectoryOptCfg
)
from traj_sampling.env_runner.legged_gym.legged_gym_envrunner2 import LeggedGymEnvRunner2


def get_command(command_name, robot_type='legged'):
    """Get command vector based on command name and robot type."""
    if robot_type == 'franka':
        # Task space position commands for Franka robot arm
        commands = {
            'reach_forward': [0.6, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],  # [x, y, z, qx, qy, qz, qw]
            'reach_backward': [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
            'reach_left': [0.4, 0.4, 0.5, 0.0, 0.0, 0.0, 1.0],
            'reach_right': [0.4, -0.4, 0.5, 0.0, 0.0, 0.0, 1.0],
            'reach_up': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'reach_down': [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            'home': [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],  # Default home position
        }
        return commands.get(command_name, [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])  # Default to home
    else:
        # Standard locomotion commands for legged robots
        commands = {
            'walk_forward': [1.0, 0.0, 0.0, 0.0],
            'walk_backward': [-1.0, 0.0, 0.0, 0.0],
            'strafe_left': [0.0, 0.5, 0.0, 0.0],
            'strafe_right': [0.0, -0.5, 0.0, 0.0],
            'turn_left': [0.0, 0.0, 0.5, 0.0],
            'turn_right': [0.0, 0.0, -0.5, 0.0],
            'stop': [0.0, 0.0, 0.0, 0.0],
        }
        return commands.get(command_name, [1.0, 0.0, 0.0, 0.0])


def get_task_name(robot):
    """Get the batch rollout task name for the given robot."""
    task_mapping = {
        'elspider_air': 'elspider_air_batch_rollout',
        'elspider_air_nav': 'elspider_air_nav',
        'elspider_air_barrier_nav': 'elair_barrier_nav',
        'elspider_air_timberpile_nav': 'elair_timberpile_nav',
        'anymal_c': 'anymal_c_batch_rollout',
        'anymal_c_nav': 'anymal_c_nav',
        'anymal_c_barrier_nav': 'anymal_c_barrier_nav',
        'anymal_c_timberpile_nav': 'anymal_c_timberpile_nav',
        'go2': 'go2_batch_rollout', 
        'cassie': 'cassie_batch_rollout',
        'franka': 'franka_batch_rollout',
    }
    return task_mapping.get(robot, 'elspider_air_batch_rollout')


def get_trajectory_config(robot, policy_checkpoint=None):
    """Get trajectory optimization config for the given robot."""
    if robot in ['elspider_air', 'elspider_air_nav', 'elspider_air_barrier_nav', 'elspider_air_timberpile_nav']:
        config = ElSpiderAirTrajectoryOptCfg()
        if policy_checkpoint:
            config.rl_warmstart.policy_checkpoint = policy_checkpoint
    elif robot in ['anymal_c', 'anymal_c_nav', 'anymal_c_barrier_nav', 'anymal_c_timberpile_nav']:
        config = AnymalCTrajectoryOptCfg()
        if policy_checkpoint:
            config.rl_warmstart.policy_checkpoint = policy_checkpoint
    elif robot == 'go2':
        config = Go2TrajectoryOptCfg()
        if policy_checkpoint:
            config.rl_warmstart.policy_checkpoint = policy_checkpoint
    elif robot == 'cassie':
        config = CassieTrajectoryOptCfg()
        if policy_checkpoint:
            config.rl_warmstart.policy_checkpoint = policy_checkpoint
    elif robot == 'franka':
        config = FrankaTrajectoryOptCfg()
        if policy_checkpoint:
            config.rl_warmstart.policy_checkpoint = policy_checkpoint
    else:
        config = TrajectoryOptimizationCfg()
        if policy_checkpoint:
            config.rl_warmstart.policy_checkpoint = policy_checkpoint
    
    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test LeggedGym v2 with trajectory optimization")
    
    # Robot and task selection
    parser.add_argument('--robot', type=str, default='elspider_air_timberpile_nav',
                        choices=['anymal_c', 'anymal_c_nav', 'anymal_c_barrier_nav', 'anymal_c_timberpile_nav', 
                                 'go2', 'cassie', 'elspider_air', 'elspider_air_nav', 
                                 'elspider_air_barrier_nav', 'elspider_air_timberpile_nav', 'franka'],
                        help='Robot type to use')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Run in headless mode (no GUI)')
                        
    # Environment parameters
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of main environments')
    parser.add_argument('--rollout_envs', type=int, default=256,
                        help='Number of rollout environments per main environment')
    
    # Trajectory optimization parameters
    parser.add_argument('--horizon_nodes', type=int, default=None,
                        help='Number of control nodes in the horizon (None uses config default)')
    parser.add_argument('--horizon_samples', type=int, default=None,
                        help='Number of samples in the horizon (None uses config default)')
    parser.add_argument('--num_diffuse_steps', type=int, default=None,
                        help='Number of diffusion steps (None uses config default)')
    parser.add_argument('--num_diffuse_steps_init', type=int, default=None,
                        help='Number of initial diffusion steps (None uses config default)')
    
    # Simulation parameters
    parser.add_argument('--num_steps', type=int, default=300,
                        help='Number of simulation steps')
    parser.add_argument('--optimize_interval', type=int, default=1,
                        help='Number of steps between trajectory optimizations')
    
    # Command parameters
    parser.add_argument('--command', type=str, default='walk_forward',
                        choices=['walk_forward', 'walk_backward', 'strafe_left', 
                                'strafe_right', 'turn_left', 'turn_right', 'stop',
                                'reach_forward', 'reach_backward', 'reach_left', 
                                'reach_right', 'reach_up', 'reach_down', 'home'],
                        help='Command to send to the robot')
    
    # RL warmstart parameters
    parser.add_argument('--policy_checkpoint', type=str, default=None,
                        help='Path to RL policy checkpoint for warmstart')
    parser.add_argument('--disable_rl_warmstart', action='store_true', default=False,
                        help='Disable RL warmstart')
    
    # Visualization options
    parser.add_argument('--debug_viz', action='store_true', default=True,
                        help='Enable debug visualization')
    parser.add_argument('--debug_viz_origins', action='store_true', default=False,
                        help='Enable visualization of environment origins')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Test mode options
    parser.add_argument('--run_comparison', action='store_true', default=False,
                        help='Run comparison between policy execution and trajectory optimization')
    parser.add_argument('--disable_trajectory_opt', action='store_true', default=False,
                        help='Disable trajectory optimization (run policy only)')
    parser.add_argument('--replay', action='store_true', default=True,
                        help='Replay recorded simulation steps')
    args = parser.parse_args()
    return args


def plot_rewards(results, save_path='legged_gym_v2_rewards.png', title_prefix=''):
    """Plot reward curves from results."""
    rewards_history = results.get("rewards_history", [])
    reward_subterms_history = results.get("reward_subterms_history", {})
    optimization_times = results.get("optimization_times", [])
    
    if not rewards_history and not reward_subterms_history:
        print("No reward data to plot")
        return
    
    # Calculate number of subplots needed
    num_plots = 1  # Total reward
    if reward_subterms_history:
        num_plots += 1  # Reward subterms
    if optimization_times:
        num_plots += 1  # Optimization times
    
    # Create figure with subplots arranged in rows
    fig_height = 4 * num_plots  # Height scales with number of plots
    fig_width = 12  # Fixed width
    fig, axes = plt.subplots(num_plots, 1, figsize=(fig_width, fig_height))
    
    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot total rewards
    if rewards_history:
        axes[plot_idx].plot(rewards_history, 'b-', linewidth=2, label='Total Reward')
        axes[plot_idx].set_title(f'{title_prefix}Total Rewards')
        axes[plot_idx].set_xlabel('Step')
        axes[plot_idx].set_ylabel('Reward')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend()
        plot_idx += 1
    
    # Plot reward subterms
    if reward_subterms_history:
        ax = axes[plot_idx]
        colors = plt.cm.tab10(np.linspace(0, 1, len(reward_subterms_history)))
        
        for i, (reward_name, values) in enumerate(reward_subterms_history.items()):
            if values:  # Only plot if there are values
                ax.plot(values, color=colors[i], linewidth=1.5, 
                       label=reward_name.replace('_', ' ').title())
        
        ax.set_title(f'{title_prefix}Reward Subterms')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward Value')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plot_idx += 1
    
    # Plot optimization times
    if optimization_times:
        axes[plot_idx].plot(optimization_times, 'r-', linewidth=2, 
                           label='Optimization Time')
        axes[plot_idx].set_title(f'{title_prefix}Optimization Times')
        axes[plot_idx].set_xlabel('Optimization Step')
        axes[plot_idx].set_ylabel('Time (seconds)')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend()
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Reward plots saved to {save_path}")
    return fig


def plot_comparison_rewards(baseline_results, traj_opt_results, save_path='comparison_rewards.png'):
    """Plot comparison of reward curves between baseline and trajectory optimization."""
    baseline_rewards = baseline_results.get("rewards_history", [])
    traj_opt_rewards = traj_opt_results.get("rewards_history", [])
    
    if not baseline_rewards and not traj_opt_rewards:
        print("No reward data to plot for comparison")
        return
    
    # Calculate number of subplot rows needed
    num_plots = 0
    if baseline_rewards or traj_opt_rewards:
        num_plots += 1  # Total reward comparison
    
    baseline_subterms = baseline_results.get("reward_subterms_history", {})
    traj_opt_subterms = traj_opt_results.get("reward_subterms_history", {})
    if baseline_subterms:
        num_plots += 1  # Baseline subterms
    if traj_opt_subterms:
        num_plots += 1  # Traj opt subterms
    
    optimization_times = traj_opt_results.get("optimization_times", [])
    if optimization_times:
        num_plots += 1  # Optimization times
    
    if baseline_rewards and traj_opt_rewards:
        num_plots += 1  # Moving average comparison
    
    # Create figure with subplots in rows
    fig_height = 4 * num_plots
    fig_width = 12
    fig, axes = plt.subplots(num_plots, 1, figsize=(fig_width, fig_height))
    
    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Total reward comparison
    if baseline_rewards or traj_opt_rewards:
        if baseline_rewards:
            axes[plot_idx].plot(baseline_rewards, 'r-', linewidth=2, label='Baseline', alpha=0.8)
        if traj_opt_rewards:
            axes[plot_idx].plot(traj_opt_rewards, 'b-', linewidth=2, label='Trajectory Optimization', alpha=0.8)
        axes[plot_idx].set_title('Total Reward Comparison')
        axes[plot_idx].set_xlabel('Step')
        axes[plot_idx].set_ylabel('Reward')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 2: Baseline reward subterms
    if baseline_subterms:
        colors = plt.cm.tab10(np.linspace(0, 1, len(baseline_subterms)))
        for i, (reward_name, values) in enumerate(baseline_subterms.items()):
            if values:
                axes[plot_idx].plot(values, color=colors[i], linewidth=1.5, 
                        label=reward_name.replace('_', ' ').title())
        axes[plot_idx].set_title('Baseline Reward Subterms')
        axes[plot_idx].set_xlabel('Step')
        axes[plot_idx].set_ylabel('Reward Value')
        axes[plot_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 3: Trajectory optimization reward subterms
    if traj_opt_subterms:
        colors = plt.cm.tab10(np.linspace(0, 1, len(traj_opt_subterms)))
        for i, (reward_name, values) in enumerate(traj_opt_subterms.items()):
            if values:
                axes[plot_idx].plot(values, color=colors[i], linewidth=1.5, 
                        label=reward_name.replace('_', ' ').title())
        axes[plot_idx].set_title('Trajectory Optimization Reward Subterms')
        axes[plot_idx].set_xlabel('Step')
        axes[plot_idx].set_ylabel('Reward Value')
        axes[plot_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 4: Optimization times (if available)
    if optimization_times:
        axes[plot_idx].plot(optimization_times, 'g-', linewidth=2)
        axes[plot_idx].set_title('Optimization Times')
        axes[plot_idx].set_xlabel('Optimization Step')
        axes[plot_idx].set_ylabel('Time (seconds)')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 5: Moving average comparison
    if baseline_rewards and traj_opt_rewards:
        window_size = max(1, min(50, len(baseline_rewards) // 10))
        
        if len(baseline_rewards) >= window_size:
            baseline_ma = np.convolve(baseline_rewards, np.ones(window_size)/window_size, mode='valid')
            axes[plot_idx].plot(range(window_size-1, len(baseline_rewards)), baseline_ma, 
                    'r-', linewidth=2, label=f'Baseline (MA-{window_size})', alpha=0.8)
        
        if len(traj_opt_rewards) >= window_size:
            traj_opt_ma = np.convolve(traj_opt_rewards, np.ones(window_size)/window_size, mode='valid')
            axes[plot_idx].plot(range(window_size-1, len(traj_opt_rewards)), traj_opt_ma, 
                    'b-', linewidth=2, label=f'Traj Opt (MA-{window_size})', alpha=0.8)
        
        axes[plot_idx].set_title('Moving Average Comparison')
        axes[plot_idx].set_xlabel('Step')
        axes[plot_idx].set_ylabel('Reward')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to {save_path}")
    return fig


def main():
    """Test the trajectory gradient sampling environment v2 for legged gym robots."""
    args = parse_arguments()
    
    # Get the task name for batch rollout environment
    task_name = get_task_name(args.robot)
    
    # Get the command for the robot
    robot_type = 'franka' if args.robot == 'franka' else 'legged'
    command = get_command(args.command, robot_type)
    # Get trajectory optimization configuration
    trajectory_config = get_trajectory_config(args.robot, args.policy_checkpoint)
    
    # Override config with command line arguments if provided
    if args.horizon_nodes is not None:
        trajectory_config.trajectory_opt.horizon_nodes = args.horizon_nodes
    if args.horizon_samples is not None:
        trajectory_config.trajectory_opt.horizon_samples = args.horizon_samples
    if args.num_diffuse_steps is not None:
        trajectory_config.trajectory_opt.num_diffuse_steps = args.num_diffuse_steps
    if args.num_diffuse_steps_init is not None:
        trajectory_config.trajectory_opt.num_diffuse_steps_init = args.num_diffuse_steps_init
    
    # Disable RL warmstart if requested
    if args.disable_rl_warmstart:
        trajectory_config.rl_warmstart.enable = False
    
    print(f"Robot: {args.robot}")
    print(f"Task: {task_name} (batch rollout environment)")
    print(f"Command: {command}")
    print(f"Main environments: {args.num_envs}")
    print(f"Rollout environments per main: {args.rollout_envs}")
    print(f"Trajectory optimization enabled: {not args.disable_trajectory_opt}")
    print(f"RL warmstart enabled: {trajectory_config.rl_warmstart.enable}")
    if trajectory_config.rl_warmstart.enable and trajectory_config.rl_warmstart.policy_checkpoint:
        print(f"RL policy checkpoint: {trajectory_config.rl_warmstart.policy_checkpoint}")
    print(f"Horizon samples: {trajectory_config.trajectory_opt.horizon_samples}")
    print(f"Horizon nodes: {trajectory_config.trajectory_opt.horizon_nodes}")
    print(f"Starting simulation for {args.num_steps} steps...")
    
    # Create the environment runner v2
    env_runner = LeggedGymEnvRunner2(
        task_name=task_name,
        num_main_envs=args.num_envs,
        num_rollout_per_main=args.rollout_envs,
        device="cuda:0",
        max_steps=args.num_steps,
        optimize_interval=args.optimize_interval,
        seed=args.seed,
        headless=args.headless,
        enable_trajectory_optimization=not args.disable_trajectory_opt,
        trajectory_opt_config=trajectory_config,
        command=command,
        debug_viz=args.debug_viz,
        debug_viz_origins=args.debug_viz_origins
    )
    
    # Reset the environment
    env_runner.reset()
    
    # Track performance metrics
    rewards_history = []
    optimization_times = []
    
    # Run the simulation
    if args.replay:
        # Start replay if requested
        env_runner.start_recording()
    start_time = time.time()
    try:
        if args.run_comparison and not args.disable_trajectory_opt:
            # Run comparison between policy execution and trajectory optimization
            results = env_runner.run_comparison(seed=args.seed)
            
            # Print detailed results
            print("\n" + "=" * 60)
            print("DETAILED RESULTS:")
            print("=" * 60)
            
            baseline_results = results.get("zero action_execution", results.get("policy_execution", {}))
            traj_opt_results = results.get("trajectory_optimization_v2", {})
            comparison = results.get("comparison", {})
            
            if baseline_results:
                print(f"Baseline execution:")
                print(f"  Test mean score: {baseline_results.get('test_mean_score', 0):.4f}")
                print(f"  Average reward: {baseline_results.get('average_reward', 0):.4f}")
                print(f"  Total steps: {baseline_results.get('steps', 0)}")
                
            if traj_opt_results:
                print(f"Trajectory optimization v2:")
                print(f"  Test mean score: {traj_opt_results.get('test_mean_score', 0):.4f}")
                print(f"  Average reward: {traj_opt_results.get('average_reward', 0):.4f}")
                print(f"  Total steps: {traj_opt_results.get('steps', 0)}")
                print(f"  Avg optimization time: {traj_opt_results.get('optimization_time', 0):.4f}s")
                print(f"  Num optimizations: {traj_opt_results.get('num_optimizations', 0)}")
                
            if comparison:
                print(f"Comparison:")
                print(f"  Improvement: {comparison.get('improvement', 0):.4f}")
                print(f"  Improvement percentage: {comparison.get('improvement_percentage', 0):.2f}%")
                
            # Plot comparison results
            if baseline_results and traj_opt_results:
                plot_comparison_rewards(baseline_results, traj_opt_results, 
                                      f'{args.robot}_comparison_rewards.png')
        
        elif not args.disable_trajectory_opt:
            # Run trajectory optimization only
            results = env_runner.run_with_trajectory_optimization(seed=args.seed)
            
            # Extract results
            rewards_history = results.get("rewards_history", [])
            optimization_times = results.get("optimization_times", [])
            
            # Print results
            print("\nSimulation completed successfully!")
            print(f"Average reward: {results['average_reward']:.3f}")
            print(f"Test mean score: {results['test_mean_score']:.3f}")
            print(f"Average optimization time: {results.get('optimization_time', 0):.4f} seconds")
            print(f"Number of optimizations: {results.get('num_optimizations', 0)}")
            
            # Plot results
            plot_rewards(results, f'{args.robot}_trajectory_opt_rewards.png', 
                        f'{args.robot.title()} Trajectory Optimization - ')
        
        else:
            # Run policy execution only (with zero actions)
            results = env_runner.run(policy=None, seed=args.seed)
            
            print("\nPolicy execution completed!")
            print(f"Average reward: {results['average_reward']:.3f}")
            print(f"Test mean score: {results['test_mean_score']:.3f}")
            
            # Plot results
            plot_rewards(results, f'{args.robot}_policy_execution_rewards.png', 
                        f'{args.robot.title()} Policy Execution - ')
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        results = {}

    total_time = time.time() - start_time
    print(f"Total simulation time: {total_time:.2f} seconds")
    
    if args.replay:
        env_runner.stop_recording()
        while(1):
            env_runner.replay_blocking()

    # Cleanup
    if hasattr(env_runner.env, 'end'):
        env_runner.env.end()


if __name__ == "__main__":
    main()
