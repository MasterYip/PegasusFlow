"""
Base classes for environment runners and environments.

This module provides the foundation for environment management and interaction
with trajectory gradient sampling. It includes base classes for environments
and environment runners that can be extended for specific use cases.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union


class EnvBase(ABC):
    """Base class for environments that can be used with trajectory gradient sampling.

    This class defines the interface that environments must implement to work
    with the trajectory optimization system.
    """

    def __init__(self, device: str = "cuda:0"):
        """Initialize the base environment.

        Args:
            device: Device to run computations on
        """
        self.device = device
        self.num_envs = 1
        self.num_actions = 1
        self.obs_dim = 1
        self.dt = 0.01

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset the environment and return initial observations.

        Returns:
            Initial observations tensor of shape (num_envs, obs_dim)
        """
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step the environment with given actions.

        Args:
            actions: Action tensor of shape (num_envs, num_actions)

        Returns:
            Tuple of (observations, rewards, dones, info_dict)
        """
        pass

    @abstractmethod
    def get_observation(self) -> torch.Tensor:
        """Get current observations.

        Returns:
            Observation tensor of shape (num_envs, obs_dim)
        """
        pass

    @abstractmethod
    def get_reward(self) -> torch.Tensor:
        """Get current rewards.

        Returns:
            Reward tensor of shape (num_envs,)
        """
        pass

    def set_commands(self, commands: torch.Tensor):
        """Set commands for the environment (optional).

        Args:
            commands: Command tensor
        """
        pass


class BatchEnvBase(EnvBase):
    """Base class for batch environments that support rollout operations.

    This extends EnvBase to support batch rollout operations needed for
    trajectory gradient sampling.
    """

    def __init__(self, num_main_envs: int, num_rollout_per_main: int = 1, device: str = "cuda:0"):
        """Initialize the batch environment.

        Args:
            num_main_envs: Number of main environments
            num_rollout_per_main: Number of rollout environments per main environment
            device: Device to run computations on
        """
        super().__init__(device)
        self.num_main_envs = num_main_envs
        self.num_rollout_per_main = num_rollout_per_main
        self.total_num_envs = num_main_envs * (1 + num_rollout_per_main)
        self.num_envs = num_main_envs  # For compatibility

        # Initialize environment indices
        self._init_env_indices()

    def _init_env_indices(self):
        """Initialize indices for main environments and rollout environments."""
        # Indices for main environments
        self.main_env_indices = torch.arange(
            0, self.total_num_envs, 1 + self.num_rollout_per_main, device=self.device
        )

        # Create mapping and masks
        self.rollout_to_main_map = torch.zeros(self.total_num_envs, dtype=torch.long, device=self.device)
        self.is_main_env = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        self.is_rollout_env = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)

        # Mark which environments are main vs rollout
        self.is_main_env[self.main_env_indices] = True
        self.is_rollout_env = ~self.is_main_env

        # For each environment, compute its main env index
        for i in range(self.num_main_envs):
            start_idx = i * (1 + self.num_rollout_per_main)
            end_idx = (i + 1) * (1 + self.num_rollout_per_main)
            self.rollout_to_main_map[start_idx:end_idx] = i

        # Get rollout env indices
        self.rollout_env_indices = torch.nonzero(self.is_rollout_env).flatten()

        # For each main env, get its rollout env indices
        self.main_to_rollout_indices = []
        for i in range(self.num_main_envs):
            start_idx = i * (1 + self.num_rollout_per_main) + 1
            end_idx = (i + 1) * (1 + self.num_rollout_per_main)
            rollout_indices = torch.arange(start_idx, end_idx, device=self.device)
            self.main_to_rollout_indices.append(rollout_indices)

    @abstractmethod
    def step_rollout(self, rollout_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step rollout environments with given actions.

        Args:
            rollout_actions: Action tensor for rollout environments

        Returns:
            Tuple of (observations, rewards, dones, info_dict) for rollout environments
        """
        pass

    @abstractmethod
    def sync_main_to_rollout(self):
        """Synchronize rollout environments to match their main environment states."""
        pass

    @abstractmethod
    def cache_main_env_states(self):
        """Cache the current state of main environments."""
        pass

    @abstractmethod
    def restore_main_env_states(self):
        """Restore main environments to their cached states."""
        pass


class EnvRunnerBase(ABC):
    """Base class for environment runners.

    Environment runners manage the interaction between policies and environments,
    providing a clean interface for running experiments and collecting data.
    """

    def __init__(self,
                 env: EnvBase,
                 device: str = "cuda:0",
                 max_steps: int = 1000):
        """Initialize the environment runner.

        Args:
            env: Environment to run
            device: Device for computations
            max_steps: Maximum number of steps per episode
        """
        self.env = env
        self.device = device
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize trajectory gradient sampling if needed
        self.traj_sampler = None

    @abstractmethod
    def run(self, policy, **kwargs) -> Dict[str, Any]:
        """Run the environment with a given policy.

        Args:
            policy: Policy to use for action selection
            **kwargs: Additional arguments

        Returns:
            Dictionary containing run results and metrics
        """
        pass

    def reset(self):
        """Reset the environment runner."""
        self.current_step = 0
        return self.env.reset()

    def set_traj_sampler(self, traj_sampler):
        """Set trajectory gradient sampler for trajectory optimization.

        Args:
            traj_sampler: TrajGradSampling instance
        """
        self.traj_sampler = traj_sampler


class BatchEnvRunnerBase(EnvRunnerBase):
    """Base class for batch environment runners that support trajectory optimization.

    This extends EnvRunnerBase to work with batch environments and trajectory
    gradient sampling for model predictive control.
    """

    def __init__(self,
                 env: BatchEnvBase,
                 device: str = "cuda:0",
                 max_steps: int = 1000,
                 horizon_samples: int = 20,
                 optimize_interval: int = 1):
        """Initialize the batch environment runner.

        Args:
            env: Batch environment to run
            device: Device for computations
            max_steps: Maximum number of steps per episode
            horizon_samples: Number of samples in trajectory horizon
            optimize_interval: Steps between trajectory optimizations
        """
        super().__init__(env, device, max_steps)
        self.horizon_samples = horizon_samples
        self.optimize_interval = optimize_interval

        # Initialize trajectory storage
        self.trajectory_history = []
        self.reward_history = []

    def rollout_callback(self, action_trajectories: torch.Tensor) -> torch.Tensor:
        """Callback function for trajectory rollout evaluation.

        This function takes a batch of action trajectories and evaluates them
        using the rollout environments, returning the cumulative rewards.

        Args:
            action_trajectories: Batch of action trajectories 
                                [batch_size, horizon_samples+1, action_dim]

        Returns:
            Cumulative rewards for each trajectory [batch_size, horizon_samples+1]
        """
        batch_size = action_trajectories.shape[0]
        horizon_length = action_trajectories.shape[1]

        # Initialize rewards tensor
        rewards = torch.zeros((batch_size, horizon_length), device=self.device)

        # Cache main environment states
        self.env.cache_main_env_states()

        # Sync rollout environments to main environments
        self.env.sync_main_to_rollout()

        # Roll out each trajectory
        for t in range(horizon_length):
            if t < horizon_length - 1:  # Don't step on the last timestep
                # Get actions for this timestep
                actions = action_trajectories[:, t, :]

                # Step rollout environments
                _, step_rewards, _, _ = self.env.step_rollout(actions)
                rewards[:, t] = step_rewards

        # Restore main environment states
        self.env.restore_main_env_states()

        return rewards

    @abstractmethod
    def run_with_trajectory_optimization(self, **kwargs) -> Dict[str, Any]:
        """Run the environment with trajectory optimization.

        Args:
            **kwargs: Additional arguments

        Returns:
            Dictionary containing run results and metrics
        """
        pass

    def optimize_trajectories(self, initial: bool = False, obs: Optional[torch.Tensor] = None):
        """Optimize trajectories using the trajectory gradient sampler.

        Args:
            initial: Whether this is the initial optimization
            obs: Optional observations for transformer policy
        """
        if self.traj_sampler is not None:
            self.traj_sampler.optimize_all_trajectories(
                self.rollout_callback,
                initial=initial,
                obs=obs
            )

    def get_next_actions(self) -> torch.Tensor:
        """Get the next actions from optimized trajectories.

        Returns:
            Action tensor for main environments
        """
        if self.traj_sampler is not None:
            # Get first action from optimized trajectory for each environment
            actions = self.traj_sampler.action_trajectories[:, 0, :]
            return actions
        else:
            # Return zero actions as fallback
            return torch.zeros((self.env.num_main_envs, self.env.num_actions), device=self.device)
