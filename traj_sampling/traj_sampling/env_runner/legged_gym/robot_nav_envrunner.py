"""
RobotNavEnvRunner for robot navigation trajectory optimization experiments.

This module implements an environment runner specifically for robot navigation
experiments that tracks detailed metrics for each environment including:
- Per-environment reward tracking
- Goal completion detection and metrics
- Distance to goal tracking
- Navigation performance metrics
- Enhanced data collection for navigation experiments
"""

import os
import torch
import numpy as np
import time
from typing import Dict, List, Any, Optional
from copy import deepcopy

from .legged_gym_envrunner2 import LeggedGymEnvRunner2
from ...config.trajectory_optimization_config import (
    TrajectoryOptimizationCfg
)


class RobotNavEnvRunner(LeggedGymEnvRunner2):
    """Environment runner for robot navigation experiments with enhanced tracking.

    This runner extends LeggedGymEnvRunner2 with additional tracking capabilities:
    1. Per-environment reward and episode tracking
    2. Goal completion detection and metrics
    3. Distance to goal tracking
    4. Navigation performance metrics
    5. Enhanced data collection for navigation experiments
    """

    def __init__(self,
                 output_dir: str = "./output",
                 task_name: str = "robot_nav_batch_rollout",
                 num_main_envs: int = 4,
                 num_rollout_per_main: int = 128,
                 device: str = "cuda:0",
                 max_steps: int = 500,
                 optimize_interval: int = 1,
                 seed: int = 0,
                 headless: bool = False,
                 enable_trajectory_optimization: bool = True,
                 trajectory_opt_config: Optional[TrajectoryOptimizationCfg] = None,
                 command: Optional[List[float]] = None,
                 debug_viz: bool = False,
                 debug_viz_origins: bool = False,
                 experiment_name: str = "robot_nav_experiment",
                 save_step_snapshots: bool = False,
                 snapshot_interval: int = 1,
                 snapshot_output_dir: Optional[str] = None):
        """Initialize the RobotNavEnvRunner.

        Args:
            experiment_name: Name of the experiment for data organization
            Other args: Same as LeggedGymEnvRunner2
        """
        # Set default trajectory config for robot navigation if not provided
        if trajectory_opt_config is None:
            trajectory_opt_config = TrajectoryOptimizationCfg()

        super().__init__(
            output_dir=output_dir,
            task_name=task_name,
            num_main_envs=num_main_envs,
            num_rollout_per_main=num_rollout_per_main,
            device=device,
            max_steps=max_steps,
            optimize_interval=optimize_interval,
            seed=seed,
            headless=headless,
            enable_trajectory_optimization=enable_trajectory_optimization,
            trajectory_opt_config=trajectory_opt_config,
            command=command,
            debug_viz=debug_viz,
            debug_viz_origins=debug_viz_origins
        )

        self.experiment_name = experiment_name

        # Snapshot configuration
        self.save_step_snapshots = save_step_snapshots
        self.snapshot_interval = snapshot_interval
        self.snapshot_output_dir = snapshot_output_dir or os.path.join(output_dir, "step_snapshots")
        self.snapshot_counter = 0

        # Enhanced tracking for robot navigation experiments
        self.per_env_rewards = []  # List of [num_envs] rewards per step
        self.per_env_episode_rewards = []  # List of [num_envs] episode rewards
        self.per_env_completion_steps = []  # Steps when each env completed task
        self.per_env_dones = []  # List of [num_envs] done flags per step
        self.episode_starts = []  # Track when episodes start for each env
        self.current_episode_rewards = torch.zeros(num_main_envs, device=device)
        self.completion_detected = torch.zeros(num_main_envs, dtype=torch.bool, device=device)

        # Navigation-specific tracking
        self.per_env_distances = []  # List of [num_envs] distances to goal per step
        self.per_env_goal_reached = []  # List of [num_envs] goal reached flags per step
        self.per_env_navigation_commands = []  # List of [num_envs, 3] navigation commands per step
        self.per_env_robot_positions = []  # List of [num_envs, 3] robot positions per step
        self.per_env_goal_positions = []  # List of [num_envs, 3] goal positions per step

        # Track noise scheduling parameters if using adaptive scheduling
        self.noise_schedule_history = []

        print(f"RobotNavEnvRunner initialized for experiment: {experiment_name}")
        print(f"Enhanced tracking for {num_main_envs} main environments")
        if self.save_step_snapshots:
            print(f"Step snapshots enabled: saving every {self.snapshot_interval} steps to {self.snapshot_output_dir}")

    def reset(self) -> torch.Tensor:
        """Reset environment and tracking variables."""
        obs = super().reset()

        # Reset tracking variables
        self.per_env_rewards = []
        self.per_env_episode_rewards = []
        self.per_env_completion_steps = []
        self.per_env_dones = []
        self.episode_starts = []
        self.current_episode_rewards = torch.zeros(self.env.num_main_envs, device=self.device_str)
        self.completion_detected = torch.zeros(self.env.num_main_envs, dtype=torch.bool, device=self.device_str)
        self.noise_schedule_history = []

        # Reset navigation-specific tracking
        self.per_env_distances = []
        self.per_env_goal_reached = []
        self.per_env_navigation_commands = []
        self.per_env_robot_positions = []
        self.per_env_goal_positions = []

        return obs

    def _ensure_viewer_camera_ready(self):
        """Ensure the viewer is ready for snapshots."""
        if not self.save_step_snapshots:
            return
        
        # Check if environment supports viewer snapshots
        if hasattr(self.env, 'save_viewer_snapshot'):
            if not getattr(self.env, 'headless', True) and hasattr(self.env, 'viewer'):
                print("Viewer available for snapshots")
            else:
                print("Warning: Running in headless mode, snapshots disabled")
                self.save_step_snapshots = False
        else:
            print("Warning: Environment does not support viewer snapshots")
            self.save_step_snapshots = False

    def _save_step_snapshot(self, step: int):
        """Save a snapshot at the current step if snapshots are enabled."""
        if not self.save_step_snapshots:
            return

        if step % self.snapshot_interval != 0:
            return

        try:
            # Ensure the environment has the snapshot capability
            if hasattr(self.env, 'save_viewer_snapshot'):
                # Save snapshot with step information
                filename_prefix = self.experiment_name
                saved_file = self.env.save_viewer_snapshot(
                    output_dir=self.snapshot_output_dir,
                    filename_prefix=filename_prefix
                )

                if saved_file:
                    print(f"Step {step}: Snapshot saved to {saved_file}")
                else:
                    print(f"Step {step}: Failed to save snapshot")

        except Exception as e:
            print(f"Warning: Could not save step snapshot at step {step}: {e}")

    def _track_episode_metrics(self, rewards: torch.Tensor, dones: torch.Tensor, step: int):
        """Track per-environment episode metrics.

        Args:
            rewards: Rewards for each environment [num_envs]
            dones: Done flags for each environment [num_envs]
            step: Current step number
        """
        # Store per-environment rewards and dones
        self.per_env_rewards.append(rewards.cpu().numpy().copy())
        self.per_env_dones.append(dones.cpu().numpy().copy())

        # Update current episode rewards
        self.current_episode_rewards += rewards

        # Check for completed episodes
        completed_envs = dones.nonzero(as_tuple=True)[0]
        for env_idx in completed_envs:
            env_idx_item = env_idx.item()

            # Record episode completion
            if not self.completion_detected[env_idx_item]:
                self.per_env_completion_steps.append((env_idx_item, step))
                self.completion_detected[env_idx_item] = True

            # Record episode reward
            episode_reward = self.current_episode_rewards[env_idx_item].item()
            self.per_env_episode_rewards.append((env_idx_item, episode_reward, step))

            # Reset episode reward for this environment
            self.current_episode_rewards[env_idx_item] = 0.0

    def _track_navigation_metrics(self, step: int):
        """Track navigation-specific metrics.

        Args:
            step: Current step number
        """
        if not hasattr(self.env, 'get_goal_reached_status'):
            return

        try:
            # Get goal reached status for all environments
            goal_reached = self.env.get_goal_reached_status()
            self.per_env_goal_reached.append(goal_reached.cpu().numpy().copy())

            # Get distance to goal for all environments
            distances = self.env.get_distance_to_goal()
            self.per_env_distances.append(distances.cpu().numpy().copy())

            # Get robot positions (only for main environments)
            if hasattr(self.env, 'root_states'):
                robot_positions = []
                for i in range(self.env.num_main_envs):
                    main_env_idx = i * (1 + self.env.num_rollout_per_main)
                    pos = self.env.root_states[main_env_idx, 0:3].cpu().numpy()
                    robot_positions.append(pos)
                self.per_env_robot_positions.append(np.array(robot_positions))

            # Get goal positions (only for main environments)
            if hasattr(self.env, 'goal_positions'):
                goal_positions = []
                for i in range(self.env.num_main_envs):
                    goal_pos = self.env.goal_positions[i].cpu().numpy()
                    goal_positions.append(goal_pos)
                self.per_env_goal_positions.append(np.array(goal_positions))

            # Get navigation commands (only for main environments)
            if hasattr(self.env, 'commands'):
                nav_commands = []
                for i in range(self.env.num_main_envs):
                    main_env_idx = i * (1 + self.env.num_rollout_per_main)
                    cmd = self.env.commands[main_env_idx, 0:3].cpu().numpy()
                    nav_commands.append(cmd)
                self.per_env_navigation_commands.append(np.array(nav_commands))

        except Exception as e:
            # If any navigation methods don't exist, skip navigation tracking
            print(f"Warning: Could not track navigation metrics: {e}")

    def _track_noise_schedule(self, step: int):
        """Track noise scheduling parameters if using adaptive scheduling.

        Args:
            step: Current step number
        """
        if (self.traj_sampler is not None and
                hasattr(self.traj_sampler, 'get_current_noise_scale')):
            try:
                noise_scale = self.traj_sampler.get_current_noise_scale()
                self.noise_schedule_history.append((step, noise_scale))
            except Exception:
                # If method doesn't exist, skip noise tracking
                pass

    def _calculate_completion_stats(self, total_steps: int) -> Dict[str, Any]:
        """Calculate task completion statistics.

        Args:
            total_steps: Total number of steps run

        Returns:
            Dictionary with completion statistics
        """
        completion_steps = [step for _, step in self.per_env_completion_steps]
        completed_count = len(completion_steps)
        total_envs = self.env.num_main_envs

        stats = {
            "completion_rate": completed_count / total_envs,
            "completed_count": completed_count,
            "total_envs": total_envs,
            "mean_completion_steps": np.mean(completion_steps) if completion_steps else total_steps,
            "min_completion_steps": np.min(completion_steps) if completion_steps else total_steps,
            "max_completion_steps": np.max(completion_steps) if completion_steps else total_steps,
            "std_completion_steps": np.std(completion_steps) if len(completion_steps) > 1 else 0.0,
            "completion_steps": completion_steps
        }

        return stats

    def _calculate_navigation_stats(self, total_steps: int) -> Dict[str, Any]:
        """Calculate navigation-specific statistics.

        Args:
            total_steps: Total number of steps run

        Returns:
            Dictionary with navigation statistics
        """
        if not self.per_env_distances:
            return {
                "final_avg_distance": None,
                "min_distance_achieved": None,
                "distance_trajectory": None,
                "goal_reached_rate": None
            }

        # Get final distances
        final_distances = self.per_env_distances[-1]
        final_avg_distance = np.mean(final_distances)

        # Get minimum distances achieved throughout the episode
        distances_array = np.array(self.per_env_distances)
        min_distances = np.min(distances_array, axis=0)
        min_distance_achieved = np.mean(min_distances)

        # Get average distance trajectory over time
        avg_distance_trajectory = np.mean(distances_array, axis=1)

        # Calculate goal reached rate
        if self.per_env_goal_reached:
            final_goal_reached = self.per_env_goal_reached[-1]
            goal_reached_rate = np.mean(final_goal_reached)
        else:
            goal_reached_rate = None

        stats = {
            "final_avg_distance": final_avg_distance,
            "min_distance_achieved": min_distance_achieved,
            "distance_trajectory": avg_distance_trajectory.tolist(),
            "goal_reached_rate": goal_reached_rate,
            "final_distances": final_distances.tolist(),
            "min_distances": min_distances.tolist()
        }

        return stats

    def run_with_trajectory_optimization(self, seed: int = 0, **kwargs) -> Dict[str, Any]:
        """Run with trajectory optimization and enhanced navigation tracking."""
        if not self.enable_trajectory_optimization:
            raise ValueError("Trajectory optimization is not enabled.")

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

        # Track reward history for plotting
        rewards_history = []
        reward_subterms_history = {}

        # Track last episode sums for step reward calculation
        if hasattr(self.env, 'episode_sums'):
            last_episode_sums = deepcopy(self.env.episode_sums)

        print(f"Starting robot navigation experiment: {self.experiment_name}")
        print(f"Trajectory optimization for {self.max_steps} steps")
        print(f"Main environments: {self.env.num_main_envs}, "
              f"Rollout environments per main: {self.env.num_rollout_per_main}")

        # Initialize trajectories from RL if enabled
        if (self.traj_sampler.use_rl_warmstart and
            hasattr(self.traj_sampler, 'node_trajectories') and
                not self.traj_sampler.rl_traj_initialized):
            self._init_trajectories_from_rl()

        completed_steps = 0
        interrupted = False

        try:
            for step in range(self.max_steps):
                completed_steps = step + 1

                # Optimize trajectories at specified intervals
                if step % self.optimize_interval == 0:
                    start_time = time.time()

                    print(f"\nStep {step}: Optimizing trajectories...")

                    # Use the exact same optimization call as original implementation
                    if step == 0:
                        n_diffuse = self.trajectory_opt_config.trajectory_opt.num_diffuse_steps_init
                    else:
                        n_diffuse = self.trajectory_opt_config.trajectory_opt.num_diffuse_steps
                    self.traj_sampler.optimize_all_trajectories(
                        rollout_callback=self.rollout_callback,
                        n_diffuse=n_diffuse,
                        initial=(step == 0),
                        obs=self.env.get_observations()
                    )

                    opt_time = time.time() - start_time
                    optimization_times.append(opt_time)
                    print(f"Optimization completed in {opt_time:.4f} seconds")

                # Track noise scheduling
                self._track_noise_schedule(step)

                # Save step snapshot if enabled
                self._save_step_snapshot(step)

                # Get actions from optimized trajectories
                actions = self.get_next_actions()

                # Step environment
                obs, priv_obs, rewards, dones, info = self.env.step(actions)

                # Enhanced tracking for robot navigation experiments
                self._track_episode_metrics(rewards, self.env.get_goal_reached_status(), step)
                self._track_navigation_metrics(step)

                # Store reward history
                rewards_history.append(rewards.cpu().numpy())

                # Track reward subterms using episode_sums for step rewards
                if hasattr(self.env, 'episode_sums'):
                    for reward_name, reward_sum in self.env.episode_sums.items():
                        if reward_name not in reward_subterms_history:
                            reward_subterms_history[reward_name] = []

                        # Calculate step reward as difference from last episode sum
                        current_sum = reward_sum[0] if isinstance(reward_sum, torch.Tensor) else reward_sum
                        last_sum = last_episode_sums[reward_name][0] if isinstance(
                            last_episode_sums[reward_name], torch.Tensor) else last_episode_sums[reward_name]
                        step_reward = (current_sum - last_sum)

                        if isinstance(step_reward, torch.Tensor):
                            reward_subterms_history[reward_name].append(step_reward.cpu().item())
                        else:
                            reward_subterms_history[reward_name].append(float(step_reward))

                    # Update last episode sums for next step
                    last_episode_sums = deepcopy(self.env.episode_sums)

                # Update trajectory sampler (shift trajectories)
                if self.traj_sampler is not None:
                    # ...existing code for trajectory shifting...
                    policy_obs = None
                    if (self.traj_sampler.use_rl_warmstart and
                        self.traj_sampler.cfg.rl_warmstart.use_for_append and
                        self.traj_sampler.rl_policy is not None and
                            self.traj_sampler.rl_traj_initialized):

                        if (self.traj_sampler.cfg.rl_warmstart.obs_type == "privileged" and
                                hasattr(self, "last_mean_traj_privileged_obs")):
                            policy_obs = self.last_mean_traj_privileged_obs.clone()
                        elif hasattr(self, "last_mean_traj_obs"):
                            policy_obs = self.last_mean_traj_obs.clone()

                    self.traj_sampler.shift_trajectory_batch(policy_obs=policy_obs)

                # Update metrics
                total_rewards += rewards
                ema_rewards = rewards * (1 - ema_coeff) + ema_rewards * ema_coeff

                # Check termination for any environment
                if dones.any():
                    terminated_envs = torch.nonzero(dones).flatten()
                    print(f"Environments {terminated_envs.tolist()} completed task at step {step}")

                    # Reset terminated environments
                    # self.env.reset_idx(terminated_envs)

                # Print progress
                if step % 10 == 0:
                    avg_reward = rewards.mean().item()
                    avg_ema_reward = ema_rewards.mean().item()
                    completed_count = self.completion_detected.sum().item()

                    # Print navigation-specific progress
                    if self.per_env_distances:
                        avg_distance = np.mean(self.per_env_distances[-1])
                        print(f"Step {step}: Avg Reward = {avg_reward:.3f}, "
                              f"EMA Reward = {avg_ema_reward:.3f}, "
                              f"Completed: {completed_count}/{self.env.num_main_envs}, "
                              f"Avg Distance: {avg_distance:.3f}")
                    else:
                        print(f"Step {step}: Avg Reward = {avg_reward:.3f}, "
                              f"EMA Reward = {avg_ema_reward:.3f}, "
                              f"Completed: {completed_count}/{self.env.num_main_envs}")

                # Early termination if all environments completed
                if self.completion_detected.all():
                    print(f"All environments completed task at step {step}")
                    break

        except KeyboardInterrupt:
            print(f"\n\nTrajectory optimization interrupted by user at step {completed_steps}")
            interrupted = True

        # Calculate final metrics
        avg_total_reward = total_rewards.mean().item()
        avg_ema_reward = ema_rewards.mean().item()
        avg_optimization_time = np.mean(optimization_times) if optimization_times else 0.0

        # Calculate completion statistics
        completion_stats = self._calculate_completion_stats(completed_steps)

        # Calculate navigation statistics
        navigation_stats = self._calculate_navigation_stats(completed_steps)

        print(f"\nRobot navigation experiment completed: {self.experiment_name}")
        print(f"Average total reward: {avg_total_reward:.3f}")
        print(f"Average EMA reward: {avg_ema_reward:.3f}")
        print(f"Average optimization time: {avg_optimization_time:.4f} seconds")
        print(f"Completion rate: {completion_stats['completion_rate']:.2%}")
        if navigation_stats['final_avg_distance'] is not None:
            print(f"Final average distance to goal: {navigation_stats['final_avg_distance']:.3f}")

        results = {
            "num_main_envs": self.env.num_main_envs,
            "num_rollout_per_main": self.env.num_rollout_per_main,
            "test_mean_score": avg_ema_reward,
            "total_reward": avg_total_reward,
            "average_reward": avg_total_reward / completed_steps if completed_steps > 0 else 0.0,
            "steps": completed_steps,
            "optimization_time": avg_optimization_time,
            "num_optimizations": len(optimization_times),
            "rewards_history": rewards_history,
            "reward_subterms_history": reward_subterms_history,
            "optimization_times": optimization_times,
            "interrupted": interrupted,

            # Enhanced robot navigation-specific metrics
            "per_env_rewards": self.per_env_rewards,
            "per_env_episode_rewards": self.per_env_episode_rewards,
            "per_env_completion_steps": self.per_env_completion_steps,
            "per_env_dones": self.per_env_dones,
            "completion_stats": completion_stats,
            "noise_schedule_history": self.noise_schedule_history,
            "experiment_name": self.experiment_name,

            # Navigation-specific metrics
            "per_env_distances": self.per_env_distances,
            "per_env_goal_reached": self.per_env_goal_reached,
            "per_env_navigation_commands": self.per_env_navigation_commands,
            "per_env_robot_positions": self.per_env_robot_positions,
            "per_env_goal_positions": self.per_env_goal_positions,
            "navigation_stats": navigation_stats
        }

        return results

    def run(self, policy=None, seed: int = 0, **kwargs) -> Dict[str, Any]:
        """Run with enhanced tracking for baseline comparison."""
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

        # Track reward history for plotting
        rewards_history = []
        reward_subterms_history = {}

        # Track last episode sums for step reward calculation
        if hasattr(self.env, 'episode_sums'):
            last_episode_sums = deepcopy(self.env.episode_sums)

        if policy is not None:
            print(f"Starting policy execution for {self.max_steps} steps")
        else:
            policy = self.traj_sampler.rl_policy if (
                self.enable_trajectory_optimization and self.traj_sampler and self.traj_sampler.use_rl_warmstart) else None
            if policy is None:
                print("No policy provided. Running zero actions.")

        controller_type = "Policy" if policy is not None else "Zero action"
        print(f"Starting robot navigation baseline experiment: {controller_type}")

        interrupted = False

        try:
            for step in range(self.max_steps):
                step_count = step + 1

                if policy is not None:
                    # ...existing policy execution code...
                    current_obs = self.env.get_observations()

                    with torch.no_grad():
                        if hasattr(policy, "predict_action"):
                            obs_dict = {'obs': current_obs}
                            action_dict = policy.predict_action(obs_dict)
                            actions = action_dict["action"].squeeze(1)
                        elif hasattr(policy, "act_inference"):
                            actions = policy.act_inference(current_obs)
                        else:
                            actions = policy(current_obs)
                else:
                    # Zero actions
                    actions = torch.zeros((self.env.num_main_envs, self.env.num_actions),
                                          device=self.device_str)

                # Step environment
                obs, priv_obs, rewards, dones, info = self.env.step(actions)

                # Save step snapshot if enabled
                self._save_step_snapshot(step)

                # Enhanced tracking for robot navigation experiments
                self._track_episode_metrics(rewards, self.env.get_goal_reached_status(), step)
                self._track_navigation_metrics(step)

                # Update metrics
                reward = rewards.mean().item()
                total_reward += reward
                ema_reward = reward * (1 - ema_coeff) + ema_reward * ema_coeff

                # Store reward history
                rewards_history.append(reward)

                # ...existing reward subterms tracking code...
                if hasattr(self.env, 'episode_sums'):
                    for reward_name, reward_sum in self.env.episode_sums.items():
                        if reward_name not in reward_subterms_history:
                            reward_subterms_history[reward_name] = []

                        current_sum = reward_sum[0] if isinstance(reward_sum, torch.Tensor) else reward_sum
                        last_sum = last_episode_sums[reward_name][0] if isinstance(
                            last_episode_sums[reward_name], torch.Tensor) else last_episode_sums[reward_name]
                        step_reward = (current_sum - last_sum)

                        if isinstance(step_reward, torch.Tensor):
                            reward_subterms_history[reward_name].append(step_reward.cpu().item())
                        else:
                            reward_subterms_history[reward_name].append(float(step_reward))

                    last_episode_sums = deepcopy(self.env.episode_sums)

                # Check termination
                if dones.any():
                    terminated_envs = torch.nonzero(dones).flatten()
                    print(f"Environments {terminated_envs.tolist()} completed task at step {step}")

                    # Reset terminated environments
                    # self.env.reset_idx(terminated_envs)

                # Print progress
                if step % 50 == 0:
                    completed_count = self.completion_detected.sum().item()

                    # Print navigation-specific progress
                    if self.per_env_distances:
                        avg_distance = np.mean(self.per_env_distances[-1])
                        print(f"Step {step}: Reward = {reward:.3f}, EMA = {ema_reward:.3f}, "
                              f"Completed: {completed_count}/{self.env.num_main_envs}, "
                              f"Avg Distance: {avg_distance:.3f}")
                    else:
                        print(f"Step {step}: Reward = {reward:.3f}, EMA = {ema_reward:.3f}, "
                              f"Completed: {completed_count}/{self.env.num_main_envs}")

                # Early termination if all environments completed
                if self.completion_detected.all():
                    print(f"All environments completed task at step {step}")
                    break

        except KeyboardInterrupt:
            print(f"\n\n{controller_type} execution interrupted by user at step {step_count}")
            interrupted = True

        # Calculate completion statistics
        completion_stats = self._calculate_completion_stats(step_count)

        # Calculate navigation statistics
        navigation_stats = self._calculate_navigation_stats(step_count)

        print(f"{controller_type} execution completed. Steps: {step_count}, "
              f"Average reward: {total_reward/step_count:.3f}, "
              f"Completion rate: {completion_stats['completion_rate']:.2%}")
        if navigation_stats['final_avg_distance'] is not None:
            print(f"Final average distance to goal: {navigation_stats['final_avg_distance']:.3f}")

        results = {
            "num_main_envs": self.env.num_main_envs,
            "num_rollout_per_main": self.env.num_rollout_per_main,
            "test_mean_score": ema_reward,
            "total_reward": total_reward,
            "average_reward": total_reward / step_count if step_count > 0 else 0.0,
            "steps": step_count,
            "rewards_history": rewards_history,
            "reward_subterms_history": reward_subterms_history,
            "interrupted": interrupted,

            # Enhanced robot navigation-specific metrics
            "per_env_rewards": self.per_env_rewards,
            "per_env_episode_rewards": self.per_env_episode_rewards,
            "per_env_completion_steps": self.per_env_completion_steps,
            "per_env_dones": self.per_env_dones,
            "completion_stats": completion_stats,
            "experiment_name": f"{controller_type.lower()}_baseline",

            # Navigation-specific metrics
            "per_env_distances": self.per_env_distances,
            "per_env_goal_reached": self.per_env_goal_reached,
            "per_env_navigation_commands": self.per_env_navigation_commands,
            "per_env_robot_positions": self.per_env_robot_positions,
            "per_env_goal_positions": self.per_env_goal_positions,
            "navigation_stats": navigation_stats
        }

        return results
