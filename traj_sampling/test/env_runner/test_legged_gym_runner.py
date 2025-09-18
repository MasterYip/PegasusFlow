#!/usr/bin/env python3

"""
Test script for LeggedGym environment with trajectory optimization.

This script demonstrates the use of trajectory optimization with LeggedGym environments.
It uses the LeggedGymEnvRunner to run trajectory optimization for legged robots.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import isaacgym
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from traj_sampling.env_runner.legged_gym.legged_gym_envrunner import LeggedGymEnvRunner


def get_command(command_name):
    """Get command vector based on command name."""
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test LeggedGym with trajectory optimization")
    
    # Robot and task selection
    parser.add_argument('--robot', type=str, default='elspider_air',
                        choices=['anymal_c', 'go2', 'cassie', 'elspider_air'],
                        help='Robot type to use')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Run in headless mode (no GUI)')
                        
    # Environment parameters
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of main environments')
    parser.add_argument('--rollout_envs', type=int, default=128,
                        help='Number of rollout environments per main environment')
    
    # Trajectory optimization parameters
    parser.add_argument('--horizon_nodes', type=int, default=4,
                        help='Number of control nodes in the horizon')
    parser.add_argument('--horizon_samples', type=int, default=16,
                        help='Number of samples in the horizon')
    parser.add_argument('--num_diffuse_steps', type=int, default=1,
                        help='Number of diffusion steps')
    
    # Simulation parameters
    parser.add_argument('--num_steps', type=int, default=300,
                        help='Number of simulation steps')
    parser.add_argument('--optimize_interval', type=int, default=1,
                        help='Number of steps between trajectory optimizations')
    
    # Command parameters
    parser.add_argument('--command', type=str, default='walk_forward',
                        choices=['walk_forward', 'walk_backward', 'strafe_left', 
                                'strafe_right', 'turn_left', 'turn_right', 'stop'],
                        help='Command to send to the robot')
    
    # Visualization options
    parser.add_argument('--debug_viz', action='store_true', default=True,
                        help='Enable debug visualization')
    parser.add_argument('--debug_viz_origins', action='store_true', default=False,
                        help='Enable visualization of environment origins')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    args = parser.parse_args()
    return args


def main():
    """Test the trajectory gradient sampling environment for legged gym robots."""
    args = parse_arguments()
    
    # Set the task based on the selected robot
    if args.robot == 'cassie':
        task_name = "cassie_traj_grad_sampling"
    elif args.robot == 'go2':
        task_name = "go2_traj_grad_sampling"
    elif args.robot == 'elspider_air':
        task_name = "elspider_air_traj_grad_sampling"
    else:
        task_name = "anymal_c_traj_grad_sampling"
    
    # Get the command for the robot
    command = get_command(args.command)
    
    print(f"Robot: {args.robot}")
    print(f"Task: {task_name}")
    print(f"Command: {command}")
    print(f"Main environments: {args.num_envs}")
    print(f"Rollout environments per main: {args.rollout_envs}")
    print(f"Starting simulation for {args.num_steps} steps...")
    
    # Create the environment runner
    env_runner = LeggedGymEnvRunner(
        task_name=task_name,
        num_main_envs=args.num_envs,
        num_rollout_per_main=args.rollout_envs,
        device="cuda:0",
        max_steps=args.num_steps,
        horizon_samples=args.horizon_samples,
        horizon_nodes=args.horizon_nodes,
        optimize_interval=args.optimize_interval,
        seed=args.seed,
        headless=args.headless,
        enable_trajectory_optimization=True,
        command=command,
        debug_viz=args.debug_viz,
        debug_viz_origins=args.debug_viz_origins
    )
    
    # Reset the environment
    env_runner.reset()
    
    # Track rewards and optimization times
    rewards_history = []
    run_time_history = []
    
    # Configure number of diffusion steps
    if hasattr(env_runner.traj_sampler, 'num_diffuse_steps'):
        env_runner.traj_sampler.num_diffuse_steps = args.num_diffuse_steps
    
    # Run the simulation with trajectory optimization
    start_time = time.time()
    try:
        results = env_runner.run_with_trajectory_optimization(seed=args.seed)
        
        # Extract results
        rewards_history = results.get("rewards_history", [])
        run_time_history = results.get("optimization_times", [])
        
        # Print results
        print("\nSimulation completed successfully!")
        print(f"Average reward: {results['average_reward']:.3f}")
        print(f"Test mean score: {results['test_mean_score']:.3f}")
        print(f"Average optimization time: {results.get('optimization_time', 0):.4f} seconds")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    total_time = time.time() - start_time
    print(f"Total simulation time: {total_time:.2f} seconds")
    
    # Plot rewards if available
    if rewards_history:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(rewards_history)
        plt.title('Rewards During Trajectory Optimization')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        
        if run_time_history:
            plt.subplot(1, 2, 2)
            plt.plot(run_time_history)
            plt.title('Optimization Times')
            plt.xlabel('Optimization Step')
            plt.ylabel('Time (seconds)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('legged_gym_rewards.png')
        print("Reward plot saved to legged_gym_rewards.png")
    
    # Cleanup
    if hasattr(env_runner.env, 'end'):
        env_runner.env.end()


if __name__ == "__main__":
    main() 