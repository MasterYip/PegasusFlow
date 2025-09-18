"""
LeggedGym Environment Runner for trajectory gradient sampling.

This module implements the environment runner for legged gym environments
that supports both regular policy execution and trajectory optimization.
"""

import torch
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import time

from ..env_runner_base import BatchEnvRunnerBase
from ...traj_grad_sampling import TrajGradSampling, TrajGradSamplingCfg

# Import legged gym task registry to create environments
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot  # Import this first to avoid circular import
from isaacgym import gymapi
from isaacgym import gymutil


class LeggedGymEnvRunner(BatchEnvRunnerBase):
    """Environment runner for LeggedGym environments with trajectory optimization support.

    This runner can operate in two modes:
    1. Regular policy execution
    2. Trajectory optimization using TrajGradSampling
    """

    def __init__(self,
                 output_dir: str = "./output",
                 task_name: str = "elspider_air_traj_grad_sampling",
                 num_main_envs: int = 4,
                 num_rollout_per_main: int = 16,
                 device: str = "cuda:0",
                 max_steps: int = 500,
                 horizon_samples: int = 20,
                 horizon_nodes: int = 5,
                 optimize_interval: int = 1,
                 seed: int = 0,
                 headless: bool = False,
                 enable_trajectory_optimization: bool = True,
                 command: Optional[List[float]] = None,
                 debug_viz: bool = False,
                 debug_viz_origins: bool = False):
        """Initialize the LeggedGym environment runner.

        Args:
            output_dir: Output directory for results
            task_name: Name of the legged gym task to run
            num_main_envs: Number of main environments
            num_rollout_per_main: Number of rollout environments per main environment
            device: Device for computations
            max_steps: Maximum number of steps per episode
            horizon_samples: Number of samples in trajectory horizon
            horizon_nodes: Number of control nodes in trajectory
            optimize_interval: Steps between trajectory optimizations
            seed: Random seed
            headless: Whether to run in headless mode
            enable_trajectory_optimization: Whether to enable trajectory optimization
            command: Command to send to the robot (e.g., [1.0, 0.0, 0.0, 0.0] for forward)
            debug_viz: Whether to enable debug visualization
            debug_viz_origins: Whether to visualize environment origins
        """
        self.task_name = task_name
        self.headless = headless
        self.debug_viz = debug_viz
        self.debug_viz_origins = debug_viz_origins
        self.seed = seed
        self.device_id = device.split(':')[-1] if ':' in device else '0'
        self.device_str = device
        
        # Create the legged gym environment
        self.env, _ = self._create_legged_gym_env(
            task_name=task_name,
            num_main_envs=num_main_envs,
            num_rollout_per_main=num_rollout_per_main,
            device=device,
            headless=headless,
            seed=seed
        )
        
        # Store environment parameters
        self.dt = self.env.dt
        self.env_step = self.dt
        
        # Set environment visualization flags
        if hasattr(self.env, "debug_viz"):
            self.env.debug_viz = debug_viz
        if hasattr(self.env, "debug_viz_origins"):
            self.env.debug_viz_origins = debug_viz_origins

        super().__init__(
            env=self.env,
            device=device,
            max_steps=max_steps,
            horizon_samples=horizon_samples,
            optimize_interval=optimize_interval
        )

        self.horizon_nodes = horizon_nodes
        self.enable_trajectory_optimization = enable_trajectory_optimization
        
        # Set default command if provided
        self.command = command
        if command is not None:
            self._set_command(command)

        # Initialize trajectory gradient sampling if enabled
        if self.enable_trajectory_optimization:
            self._init_trajectory_optimization()

        # Placeholder for environment-specific data
        self.env_data = {}

    def _create_legged_gym_env(self, task_name, num_main_envs, num_rollout_per_main, device, headless, seed):
        """Create a legged gym environment with batch rollout capability.

        Args:
            task_name: Name of the task to run
            num_main_envs: Number of main environments
            num_rollout_per_main: Number of rollout environments per main environment
            device: Device for computations
            headless: Whether to run in headless mode
            seed: Random seed

        Returns:
            Tuple of (env, env_cfg)
        """
        # Import necessary legged gym modules
        try:
            import isaacgym
            import legged_gym
        except ImportError:
            raise ImportError("IsaacGym and legged_gym must be installed to use LeggedGymEnvRunner")
            
        # Create command line arguments
        args = gymutil.parse_arguments(
            description="LeggedGymEnvRunner",
            headless=headless,
            custom_parameters=[
                {"name": "--seed", "type": int, "default": seed, "help": "Random seed"},
                {"name": "--task", "type": str, "default": task_name, "help": "Task name"},
            ]
        )
        args.headless = headless
        args.seed = seed
        args.device = device
        args.device_id = int(self.device_id)
        args.task = task_name
        args.num_envs = num_main_envs
        
        # Get the environment config
        env_cfg, _ = task_registry.get_cfgs(args.task)
        
        # Configure environment for batch rollout
        env_cfg.env.num_envs = num_main_envs
        env_cfg.env.rollout_envs = num_rollout_per_main
        env_cfg.trajectory_opt.enable_traj_opt = False
        self.enable_rl_warmstart = env_cfg.rl_warmstart.enable
        env_cfg.rl_warmstart.enable = False
        self.env_cfg = env_cfg

        # Enable debug visualization if requested
        if hasattr(env_cfg.viewer, "debug_viz"):
            env_cfg.viewer.debug_viz = self.debug_viz
        if hasattr(env_cfg.viewer, "debug_viz_origins"):
            env_cfg.viewer.debug_viz_origins = self.debug_viz_origins
        
        # Create the environment
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        
        print(f"Created LeggedGym environment for task {task_name} with {num_main_envs} main environments "
              f"and {num_rollout_per_main} rollout environments per main environment")
        
        return env, env_cfg

    def _init_trajectory_optimization(self):
        """Initialize trajectory gradient sampling for trajectory optimization."""
        # Create configuration for trajectory optimization
        cfg = TrajGradSamplingCfg()

        # Set trajectory optimization parameters
        cfg.trajectory_opt.enable_traj_opt = True
        cfg.trajectory_opt.horizon_samples = self.horizon_samples
        cfg.trajectory_opt.horizon_nodes = self.horizon_nodes
        cfg.trajectory_opt.num_samples = self.env.num_rollout_per_main - 1  # NOTE: One for mean traj
        cfg.trajectory_opt.num_diffuse_steps = 1
        cfg.trajectory_opt.num_diffuse_steps_init = 6
        cfg.trajectory_opt.temp_sample = 0.1
        cfg.trajectory_opt.update_method = "avwbfo"
        cfg.trajectory_opt.interp_method = "spline"
        cfg.trajectory_opt.gamma = 1.0
        cfg.trajectory_opt.noise_scaling = 1.5
        cfg.rl_warmstart = self.env_cfg.rl_warmstart
        cfg.rl_warmstart.enable = self.enable_rl_warmstart

        # Set environment parameters
        cfg.env.num_actions = self.env.num_actions
        cfg.sim_device = self.device_str

        # Initialize trajectory gradient sampler
        self.traj_sampler = TrajGradSampling(
            cfg=cfg,
            device=self.device_str,
            num_envs=self.env.num_main_envs,
            num_actions=self.env.num_actions,
            dt=self.env.dt,
            main_env_indices=self.env.main_env_indices
        )

        # Set the trajectory sampler in the base class
        self.set_traj_sampler(self.traj_sampler)

        # Initialize RL policy if enabled
        if cfg.rl_warmstart.enable:
            self.traj_sampler.init_rl_policy(
                num_obs=self.env_cfg.env.num_observations,
                num_privileged_obs=self.num_privileged_obs if hasattr(self, 'num_privileged_obs') else None
            )

        print(f"Trajectory optimization initialized with {self.horizon_samples} horizon samples, "
              f"{self.horizon_nodes} nodes, and {self.env.num_rollout_per_main} rollout environments")

    def _set_command(self, command):
        """Set command for the robot.
        
        Args:
            command: Command to send to the robot (e.g., [1.0, 0.0, 0.0, 0.0] for forward)
        """
        if not hasattr(self.env, "commands"):
            print("Warning: Environment does not have commands attribute")
            return
            
        if isinstance(command, list):
            command_tensor = torch.tensor(command, device=self.device_str)
        else:
            command_tensor = command
            
        # Expand command to all environments
        commands = command_tensor.unsqueeze(0).repeat(self.env.total_num_envs, 1)
        self.env.set_all_commands(commands)
        print(f"Set command: {command}")

    def rollout_callback(self, action_trajectories: torch.Tensor) -> torch.Tensor:
        """Callback function for trajectory rollout evaluation.

        This function takes a batch of action trajectories and evaluates them
        using the rollout environments, returning the cumulative rewards.

        Args:
            action_trajectories: Batch of action trajectories 
                                [batch_size, horizon_length, action_dim]

        Returns:
            Cumulative rewards for each trajectory [batch_size, horizon_length]
        """
        batch_size = action_trajectories.shape[0]
        horizon_length = action_trajectories.shape[1]

        # Initialize rewards tensor
        rewards = torch.zeros((batch_size, horizon_length), device=self.device_str)

        # Sync rollout environments to main environments
        self.env._sync_main_to_rollout()

        # Roll out each trajectory
        for t in range(horizon_length):
            if t < horizon_length - 1:  # Don't step on the last timestep
                # Get actions for this timestep
                actions = action_trajectories[:, t, :]

                # Step rollout environments
                obs, priv_obs, step_rewards, dones, info = self.env.step_rollout(actions)
                rewards[:, t] = step_rewards

        # Restore main environment states
        self.env._restore_main_env_states()

        return rewards

    def _init_trajectories_from_rl(self):
        """Initialize trajectories by rolling out the RL policy."""
        if self.traj_sampler is None:
            return

        def rollout_callback(rl_policy, obs_mean, obs_var):
            """Callback function to perform RL policy rollout."""
            # Sync the rollout environments with the main environments first
            self.env._sync_main_to_rollout()

            # Create a tensor to hold the action trajectories for all main environments
            batch_size = self.env.num_envs
            horizon = self.traj_sampler.horizon_samples
            traj_actions = torch.zeros((batch_size, horizon+1, self.traj_sampler.action_size), device=self.device)

            # Get current observations from rollout environments
            self.env.mean_traj_env_indices = torch.range(0, self.env.num_envs - 1, device=self.device).long()*self.env.num_rollout_per_main
            obs_batch = self.env.obs_buf[self.env.main_env_indices + 1].clone()
            # Roll out the policy for each step in the horizon
            for i in range(horizon+1):
                # Prepare observations for policy (using privileged or non-privileged based on config)
                if self.env.cfg.rl_warmstart.obs_type == "privileged" and hasattr(self, "rollout_privileged_obs"):
                    policy_obs = self.env.rollout_privileged_obs.clone()
                else:
                    policy_obs = obs_batch.clone()

                # Standardize observations if needed
                if obs_mean is not None and obs_var is not None:
                    policy_obs = (policy_obs - obs_mean) / torch.sqrt(obs_var + 1e-8)

                with torch.no_grad():
                    # Get actions from policy using act_inference method
                    actions = rl_policy.act_inference(policy_obs)

                # Store actions in trajectory
                traj_actions[:, i, :] = actions

                # Step rollout environments to get next observations
                obs, privileged_obs, _, _, _ = self.env.step_rollout(actions)
                obs_batch = obs[self.env.mean_traj_env_indices].clone()
            # Reset rollout environments back to main state
            self.env._sync_main_to_rollout()

            return traj_actions

        self.traj_sampler.init_trajectories_from_rl(rollout_callback)

    def run_with_trajectory_optimization(self, seed: int = 0, **kwargs) -> Dict[str, Any]:
        """Run the environment with trajectory optimization.

        Args:
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            Dictionary containing run results and metrics
        """
        if not self.enable_trajectory_optimization:
            raise ValueError("Trajectory optimization is not enabled. Set enable_trajectory_optimization=True.")

        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Reset environment
        obs = self.reset()

        # Set default command if provided
        if self.command is not None:
            self._set_command(self.command)

        # Track metrics
        total_rewards = torch.zeros(self.env.num_main_envs, device=self.device_str)
        ema_rewards = torch.zeros(self.env.num_main_envs, device=self.device_str)
        ema_coeff = 0.99
        optimization_times = []
        
        # Command tracking metrics
        if hasattr(self.env, "commands"):
            command_tracking = []

        print(f"Starting trajectory optimization for {self.max_steps} steps")
        print(f"Main environments: {self.env.num_main_envs}, "
              f"Rollout environments per main: {self.env.num_rollout_per_main}")

        self._init_trajectories_from_rl()

        for step in range(self.max_steps):
            # Optimize trajectories at specified intervals
            if step % self.optimize_interval == 0:
                start_time = time.time()

                print(f"\nStep {step}: Optimizing trajectories...")
                self.optimize_trajectories(initial=(step == 0), obs=self.env.get_observations())

                opt_time = time.time() - start_time
                optimization_times.append(opt_time)
                print(f"Optimization completed in {opt_time:.4f} seconds")

            # Get actions from optimized trajectories
            actions = self.get_next_actions()
            
            # Step environment
            obs, priv_obs, rewards, dones, info = self.env.step(actions)

            # Update trajectory sampler (shift trajectories)
            if self.traj_sampler is not None:
                self.traj_sampler.shift_trajectory_batch(policy_obs=obs)

            # Update metrics
            total_rewards += rewards
            ema_rewards = rewards * (1 - ema_coeff) + ema_rewards * ema_coeff
            
            # Track command tracking if available
            if hasattr(self.env, "commands") and hasattr(self.env, "base_lin_vel"):
                # For first environment, track command vs actual velocity
                cmd = self.env.commands[0, :3].cpu().numpy()  # First 3 values are velocity commands
                actual = self.env.base_lin_vel[0].cpu().numpy()
                command_tracking.append((cmd, actual))

            # Check termination for any environment
            if dones.any():
                terminated_envs = torch.nonzero(dones).flatten()
                print(f"Environments {terminated_envs.tolist()} terminated at step {step}")
                
                # Reset terminated environments
                self.env.reset_idx(terminated_envs)

            # Print progress
            if step % 10 == 0:
                avg_reward = rewards.mean().item()
                avg_ema_reward = ema_rewards.mean().item()
                print(f"Step {step}: Avg Reward = {avg_reward:.3f}, "
                      f"Avg EMA Reward = {avg_ema_reward:.3f}")

        # Calculate final metrics
        avg_total_reward = total_rewards.mean().item()
        avg_ema_reward = ema_rewards.mean().item()
        avg_optimization_time = np.mean(optimization_times) if optimization_times else 0.0

        print(f"\nTrajectory optimization completed.")
        print(f"Average total reward: {avg_total_reward:.3f}")
        print(f"Average EMA reward: {avg_ema_reward:.3f}")
        print(f"Average optimization time: {avg_optimization_time:.4f} seconds")
        
        results = {
            "test_mean_score": avg_ema_reward,
            "total_reward": avg_total_reward,
            "average_reward": avg_total_reward / self.max_steps,
            "steps": self.max_steps,
            "optimization_time": avg_optimization_time,
            "num_optimizations": len(optimization_times)
        }
        
        # Add command tracking if available
        if command_tracking:
            results["command_tracking"] = command_tracking

        return results
        
    def run(self, policy=None, seed: int = 0, **kwargs) -> Dict[str, Any]:
        """Run the environment with a given policy.

        Args:
            policy: Policy to use for action selection (optional)
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            Dictionary containing run results and metrics
        """
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Reset environment
        obs = self.reset()

        # Set default command if provided
        if self.command is not None:
            self._set_command(self.command)

        # Track metrics
        total_reward = 0.0
        ema_reward = 0.0
        ema_coeff = 0.99
        step_count = 0
        
        # Command tracking metrics
        if hasattr(self.env, "commands"):
            command_tracking = []

        if policy is not None:
            print(f"Starting policy execution for {self.max_steps} steps")
        else:
            print("No policy provided. Running zero actions.")

        for step in range(self.max_steps):
            if policy is not None:
                # Policy-based action selection
                current_obs = self.env.get_observations()  # Get observations for main environments
                
                # Predict action using policy
                with torch.no_grad():
                    if hasattr(policy, "predict_action"):
                        # For traj_opt_policy format
                        obs_dict = {'obs': current_obs}
                        action_dict = policy.predict_action(obs_dict)
                        actions = action_dict["action"].squeeze(1)  # Remove time dimension
                    elif hasattr(policy, "act_inference"):
                        # For RL policy format
                        actions = policy.act_inference(current_obs)
                    else:
                        # Default case
                        actions = policy(current_obs)
            else:
                # Zero actions
                actions = torch.zeros((self.env.num_main_envs, self.env.num_actions), 
                                     device=self.device_str)

            # Step environment
            obs, rewards, dones, info = self.env.step(actions)

            # Update metrics
            reward = rewards.mean().item()
            total_reward += reward
            ema_reward = reward * (1 - ema_coeff) + ema_reward * ema_coeff
            step_count += 1
            
            # Track command tracking if available
            if hasattr(self.env, "commands") and hasattr(self.env, "base_lin_vel"):
                # For first environment, track command vs actual velocity
                cmd = self.env.commands[0, :3].cpu().numpy()  # First 3 values are velocity commands
                actual = self.env.base_lin_vel[0].cpu().numpy()
                command_tracking.append((cmd, actual))

            # Check termination
            if dones.any():
                terminated_envs = torch.nonzero(dones).flatten()
                print(f"Environments {terminated_envs.tolist()} terminated at step {step}")
                
                # Reset terminated environments
                self.env.reset_idx(terminated_envs)

            # Print progress
            if step % 50 == 0:
                print(f"Step {step}: Reward = {reward:.3f}, EMA Reward = {ema_reward:.3f}")

        controller_type = "Policy" if policy is not None else "Zero action"
        print(f"{controller_type} execution completed. Total steps: {step_count}, "
              f"Average reward: {total_reward/step_count:.3f}, Final EMA reward: {ema_reward:.3f}")

        results = {
            "test_mean_score": ema_reward,
            "total_reward": total_reward,
            "average_reward": total_reward / step_count,
            "steps": step_count
        }
        
        # Add command tracking if available
        if command_tracking:
            results["command_tracking"] = command_tracking
            
        return results

    def run_comparison(self, policy=None, seed: int = 0, **kwargs) -> Dict[str, Any]:
        """Run both policy execution and trajectory optimization for comparison.

        Args:
            policy: Policy to use for comparison
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            Dictionary containing comparison results
        """
        print("=" * 60)
        controller_type = "Policy" if policy is not None else "Zero action"
        print(f"RUNNING COMPARISON: {controller_type} vs Trajectory Optimization")
        print("=" * 60)

        # Run policy execution
        print(f"\n1. Running with {controller_type.lower()} execution...")
        baseline_results = self.run(policy, seed=seed, **kwargs)

        # Reset environment state
        self.reset()

        # Run trajectory optimization (if enabled)
        if self.enable_trajectory_optimization:
            print("\n2. Running with trajectory optimization...")
            traj_opt_results = self.run_with_trajectory_optimization(seed=seed, **kwargs)
        else:
            print("\n2. Trajectory optimization disabled.")
            traj_opt_results = {}

        # Compile comparison results
        comparison_results = {
            f"{controller_type.lower()}_execution": baseline_results,
            "trajectory_optimization": traj_opt_results,
            "comparison": {}
        }

        if traj_opt_results:
            baseline_score = baseline_results.get("test_mean_score", 0.0)
            traj_opt_score = traj_opt_results.get("test_mean_score", 0.0)
            improvement = traj_opt_score - baseline_score
            improvement_pct = (improvement / abs(baseline_score)) * 100 if baseline_score != 0 else 0

            comparison_results["comparison"] = {
                f"{controller_type.lower()}_score": baseline_score,
                "trajectory_optimization_score": traj_opt_score,
                "improvement": improvement,
                "improvement_percentage": improvement_pct
            }

            print("\n" + "=" * 60)
            print("COMPARISON RESULTS:")
            print("=" * 60)
            print(f"{controller_type} execution score: {baseline_score:.4f}")
            print(f"Trajectory optimization score: {traj_opt_score:.4f}")
            print(f"Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
            print("=" * 60)

        return comparison_results 