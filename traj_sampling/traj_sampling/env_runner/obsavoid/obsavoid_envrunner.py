"""
ObsAvoid Environment Runner for trajectory gradient sampling.

This module implements the environment runner for the obsavoid environment
that supports both regular policy execution and trajectory optimization.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from ..env_runner_base import BatchEnvRunnerBase
from .obsavoid_batch_env import ObsAvoidBatchEnv, create_randpath_bound_batch_env, create_sine_bound_batch_env

from ...traj_grad_sampling import TrajGradSampling, TrajGradSamplingCfg


class ObsAvoidEnvRunner(BatchEnvRunnerBase):
    """Environment runner for ObsAvoid environments with trajectory optimization support.

    This runner can operate in two modes:
    1. Regular policy execution (similar to original ObsAvoidRunner)
    2. Trajectory optimization using TrajGradSampling
    """

    def __init__(self,
                 output_dir: str = "./output",
                 num_main_envs: int = 4,
                 num_rollout_per_main: int = 16,
                 device: str = "cuda:0",
                 max_steps: int = 500,
                 horizon_samples: int = 20,
                 horizon_nodes: int = 5,
                 optimize_interval: int = 1,
                 env_type: str = "randpath",  # "randpath", "sine", "increase"
                 env_step: float = 0.01,
                 n_obs_steps: int = 4,
                 n_action_steps: int = 8,
                 enable_trajectory_optimization: bool = True,
                 enable_vis: bool = False,
                 vis_time_window: float = 1.0):
        """Initialize the ObsAvoid environment runner.

        Args:
            output_dir: Output directory for results
            num_main_envs: Number of main environments
            num_rollout_per_main: Number of rollout environments per main environment
            device: Device for computations
            max_steps: Maximum number of steps per episode
            horizon_samples: Number of samples in trajectory horizon
            horizon_nodes: Number of control nodes in trajectory
            optimize_interval: Steps between trajectory optimizations
            env_type: Type of environment ("randpath", "sine", "increase")
            env_step: Environment timestep
            n_obs_steps: Number of observation steps for policy
            n_action_steps: Number of action steps for policy
            enable_trajectory_optimization: Whether to enable trajectory optimization
            enable_vis: Whether to enable visualization
            vis_time_window: Time window for visualization
        """
        # Create the batch environment
        if env_type == "sine":
            env = create_sine_bound_batch_env(
                num_main_envs=num_main_envs,
                num_rollout_per_main=num_rollout_per_main,
                device=device,
                env_step=env_step,
                enable_vis=enable_vis,
                vis_time_window=vis_time_window
            )
        else:  # Default to randpath
            env = create_randpath_bound_batch_env(
                num_main_envs=num_main_envs,
                num_rollout_per_main=num_rollout_per_main,
                device=device,
                env_step=env_step,
                enable_vis=enable_vis,
                vis_time_window=vis_time_window
            )

        super().__init__(
            env=env,
            device=device,
            max_steps=max_steps,
            horizon_samples=horizon_samples,
            optimize_interval=optimize_interval
        )

        self.horizon_nodes = horizon_nodes
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.enable_trajectory_optimization = enable_trajectory_optimization

        # Initialize trajectory gradient sampling if enabled
        if self.enable_trajectory_optimization:
            self._init_trajectory_optimization()

        # For policy-based execution
        self.obs_history = None

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
        cfg.trajectory_opt.noise_scaling = 200

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

        print(f"Trajectory optimization initialized with {self.horizon_samples} horizon samples, "
              f"{self.horizon_nodes} nodes, and {self.env.num_rollout_per_main} rollout environments")

    def run(self, policy=None, ctrl_mode: str = 'acc', seed: int = 20, **kwargs) -> Dict[str, Any]:
        """Run the environment with a given policy (without trajectory optimization).

        This method provides compatibility with the original ObsAvoidRunner interface.
        If no policy is provided, uses a simple PID controller.

        Args:
            policy: Policy to use for action selection (optional)
            ctrl_mode: Control mode ('acc', 'y', 'dy')
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

        # Track metrics
        total_reward = 0.0
        ema_reward = 0.0
        ema_coeff = 0.99
        step_count = 0

        if policy is not None:
            print(f"Starting policy execution for {self.max_steps} steps with control mode: {ctrl_mode}")
            # Initialize observation history for policy
            self.obs_history = [obs[0].cpu().numpy() for _ in range(self.n_obs_steps)]
        else:
            print(f"Starting PID controller execution for {self.max_steps} steps")

        for step in range(self.max_steps):
            if policy is not None:
                # Policy-based action selection
                current_obs = self.env.get_observation()[0].cpu().numpy()  # Use first environment

                # Update observation history
                self.obs_history = self.obs_history[1:] + [current_obs]
                np_obs_array = np.array(self.obs_history, dtype=np.float32)

                # Convert to tensor and add batch dimension
                obs_tensor = torch.from_numpy(np_obs_array).to(device=self.device).unsqueeze(0)
                obs_dict = {'obs': obs_tensor}

                # Predict action using policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # Extract action
                action = action_dict["action"][0, 0, 0].detach().cpu().numpy()
            else:
                # Batch PID controller for all main environments
                main_indices = self.env.main_env_indices
                num_main_envs = len(main_indices)
                
                # Get current states for all main environments
                y_current = self.env.y[main_indices]
                v_current = self.env.v[main_indices]
                t_current = self.env.t[main_indices]
                
                # Compute target positions (center of corridor) for all environments
                target_y = torch.zeros(num_main_envs, device=self.device)
                
                for i, env_idx in enumerate(main_indices):
                    params = self.env.bound_params[env_idx.item()]
                    
                    # Compute target position for this environment
                    res = params['slope'] * t_current[i].item()
                    for j in range(len(params['coef']) // 2):
                        res += params['coef'][j * 2] * np.sin((j + 1) * t_current[i].item()) + \
                            params['coef'][j * 2 + 1] * np.cos((j + 1) * t_current[i].item())
                    
                    target_y[i] = res
                
                # Batch PID control for all main environments
                errors = target_y - y_current
                actions = torch.zeros(num_main_envs, device=self.device)
                
                for i, env_idx in enumerate(main_indices):
                    params = self.env.bound_params[env_idx.item()]
                    action = params['p'] * errors[i].item() + params['d'] * (-v_current[i].item())
                    action = action / self.env.acc_scale  # Scale to action space
                    action = np.clip(action, -500, 500)  # Clip to bounds
                    actions[i] = action

            # Convert action to tensor format for environment
            # actions is already a tensor with shape (num_main_envs,) from the batch PID controller
            if policy is not None:
                # FIXME
                # For policy mode, we only have one action, so expand it to all main environments
                action_tensor = torch.tensor([[action] for _ in range(self.env.num_main_envs)], 
                                           device=self.device, dtype=torch.float32)
            else:
                # For PID mode, actions is already computed for all main environments
                action_tensor = actions.unsqueeze(-1)  # Add action dimension: (num_main_envs, 1)

            # Step environment
            obs, rewards, dones, info = self.env.step(action_tensor)

            # Update metrics
            reward = rewards[0].item()
            total_reward += reward
            ema_reward = reward * (1 - ema_coeff) + ema_reward * ema_coeff
            step_count += 1

            # Check termination
            if dones[0] or abs(self.env.y[0].item()) > 10:
                print(f"Episode terminated at step {step}")
                break

            # Print progress
            if step % 50 == 0:
                print(f"Step {step}: Action = {action:.3f}, Reward = {reward:.3f}, "
                      f"EMA Reward = {ema_reward:.3f}, Position = {self.env.y[0].item():.3f}")

        controller_type = "Policy" if policy is not None else "PID"
        print(f"{controller_type} execution completed. Total steps: {step_count}, "
              f"Average reward: {total_reward/step_count:.3f}, Final EMA reward: {ema_reward:.3f}")

        return {
            "test_mean_score": ema_reward,
            "total_reward": total_reward,
            "average_reward": total_reward / step_count,
            "steps": step_count
        }

    def run_with_trajectory_optimization(self, seed: int = 20, **kwargs) -> Dict[str, Any]:
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

        # Track metrics
        total_rewards = torch.zeros(self.env.num_main_envs, device=self.device)
        ema_rewards = torch.zeros(self.env.num_main_envs, device=self.device)
        ema_coeff = 0.99
        optimization_times = []

        print(f"Starting trajectory optimization for {self.max_steps} steps")
        print(f"Main environments: {self.env.num_main_envs}, "
              f"Rollout environments per main: {self.env.num_rollout_per_main}")

        for step in range(self.max_steps):
            # Optimize trajectories at specified intervals
            if step % self.optimize_interval == 0:
                import time
                start_time = time.time()

                print(f"\nStep {step}: Optimizing trajectories...")
                self.optimize_trajectories(initial=(step == 0), obs=self.env.get_observation())

                opt_time = time.time() - start_time
                optimization_times.append(opt_time)
                print(f"Optimization completed in {opt_time:.4f} seconds")

            # Get actions from optimized trajectories
            actions = self.get_next_actions()
            actions = torch.clamp(actions, -500, 500)  # Clip actions to [-500, 500] (Default action bounds)
            print(f"Step {step}: Actions selected: {actions.cpu().numpy().T}")
            # Step environment
            obs, rewards, dones, info = self.env.step(actions)

            # Update trajectory sampler (shift trajectories)
            if self.traj_sampler is not None:
                self.traj_sampler.shift_trajectory_batch()

            # Update metrics
            total_rewards += rewards
            ema_rewards = rewards * (1 - ema_coeff) + ema_rewards * ema_coeff

            # Check termination for any environment
            if dones.any():
                terminated_envs = torch.nonzero(dones).flatten()
                print(f"Environments {terminated_envs.tolist()} terminated at step {step}")

                # Reset terminated environments (simple reset)
                for env_idx in terminated_envs:
                    self.env.y[env_idx] = self.env.bound_params[env_idx]['initial_y']
                    self.env.v[env_idx] = self.env.bound_params[env_idx]['initial_v']
                    self.env.t[env_idx] = 0.0

            # Print progress
            if step % 10 == 0:
                avg_reward = rewards.mean().item()
                avg_ema_reward = ema_rewards.mean().item()
                avg_position = self.env.y[self.env.main_env_indices].mean().item()
                print(f"Step {step}: Avg Reward = {avg_reward:.3f}, "
                      f"Avg EMA Reward = {avg_ema_reward:.3f}, Avg Position = {avg_position:.3f}")

        # Calculate final metrics
        avg_total_reward = total_rewards.mean().item()
        avg_ema_reward = ema_rewards.mean().item()
        avg_optimization_time = np.mean(optimization_times) if optimization_times else 0.0

        print(f"\nTrajectory optimization completed.")
        print(f"Average total reward: {avg_total_reward:.3f}")
        print(f"Average EMA reward: {avg_ema_reward:.3f}")
        print(f"Average optimization time: {avg_optimization_time:.4f} seconds")

        return {
            "test_mean_score": avg_ema_reward,
            "total_reward": avg_total_reward,
            "average_reward": avg_total_reward / self.max_steps,
            "steps": self.max_steps,
            "optimization_time": avg_optimization_time,
            "num_optimizations": len(optimization_times)
        }

    def run_comparison(self, policy=None, ctrl_mode: str = 'acc', seed: int = 20, **kwargs) -> Dict[str, Any]:
        """Run both policy/PID execution and trajectory optimization for comparison.

        Args:
            policy: Policy to use for comparison (optional, uses PID if None)
            ctrl_mode: Control mode for policy execution
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            Dictionary containing comparison results
        """
        print("=" * 60)
        controller_type = "Policy" if policy is not None else "PID"
        print(f"RUNNING COMPARISON: {controller_type} vs Trajectory Optimization")
        print("=" * 60)

        # Run policy/PID execution
        print(f"\n1. Running with {controller_type.lower()} execution...")
        baseline_results = self.run(policy, ctrl_mode=ctrl_mode, seed=seed, **kwargs)

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
