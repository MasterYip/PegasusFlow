"""
Weighted Basis Function Optimization (WBFO) implementation.

This module provides utilities for trajectory optimization using weighted basis functions.
It implements the approach described in the paper which extends conventional trajectory
optimization methods by introducing a more nuanced weighting scheme that captures both 
temporal and spatial relationships.
"""

import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from .spline import SplineBase, UniBSpline, InterpolatedSpline, CatmullRomSpline

# Try to import JAX implementation if available
try:
    from .spline import InterpolatedSplineJAX, JAX_AVAILABLE
except ImportError:
    InterpolatedSplineJAX = None
    JAX_AVAILABLE = False


class OptimizerBase(ABC):
    """Abstract base class for trajectory optimizers.
    
    This class defines the common interface that all trajectory optimizers should implement.
    It provides basic functionality for initialization and abstract methods that subclasses
    must implement.
    """

    def __init__(self,
                 horizon_nodes: int,
                 horizon_samples: int,
                 action_dim: int,
                 temp_tau: float = 0.1,
                 dt: float = 0.02,
                 device: torch.device = None):
        """Initialize the base optimizer.

        Args:
            horizon_nodes: Number of knot points in the trajectory
            horizon_samples: Number of sample points in the dense trajectory
            action_dim: Dimension of the action space
            temp_tau: Temperature parameter for softmax weighting
            dt: Timestep for trajectory planning (default: 0.02)
            device: Device to use for tensor operations
        """
        self.horizon_nodes = horizon_nodes
        self.horizon_samples = horizon_samples
        self.action_dim = action_dim
        self.temp_tau = temp_tau
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize spline for node2dense and dense2node operations

        # self.spline = UniBSpline(
        #     horizon_nodes=horizon_nodes,
        #     horizon_samples=horizon_samples,
        #     dt=dt,
        #     device=self.device
        # )
        
        # self.spline = InterpolatedSplineJAX(
        #     horizon_nodes=horizon_nodes,
        #     horizon_samples=horizon_samples,
        #     dt=dt,
        #     device=self.device
        # )

        self.spline = CatmullRomSpline(
            horizon_nodes=horizon_nodes,
            horizon_samples=horizon_samples,
            dt=dt,
            device=self.device
        )

    @abstractmethod
    def optimize(self,
                 mean_traj: torch.Tensor,
                 sampled_trajs: torch.Tensor,
                 rewards: torch.Tensor) -> torch.Tensor:
        """Optimize a trajectory using the specific algorithm.

        Args:
            mean_traj: Mean trajectory [horizon_nodes, action_dim]
            sampled_trajs: Sampled trajectories [num_samples, horizon_nodes, action_dim]
            rewards: Rewards for each trajectory [num_samples, horizon_samples]

        Returns:
            Optimized trajectory [horizon_nodes, action_dim]
        """
        pass

    def optimize_batch(self,
                       mean_trajs: torch.Tensor,
                       sampled_trajs_batch: torch.Tensor,
                       rewards_batch: torch.Tensor,
                       sample_indices: torch.Tensor = None) -> torch.Tensor:
        """Optimize multiple trajectories using batch processing.

        Args:
            mean_trajs: Mean trajectories [num_envs, horizon_nodes, action_dim]
            sampled_trajs_batch: Batch of sampled trajectories [total_samples, horizon_nodes, action_dim]
            rewards_batch: Rewards for each trajectory [total_samples, horizon_samples]
            sample_indices: Which environment each sample belongs to [total_samples]

        Returns:
            Optimized trajectories [num_envs, horizon_nodes, action_dim]
        """
        num_envs = mean_trajs.shape[0]
        total_samples = sampled_trajs_batch.shape[0]

        if sample_indices is None:
            samples_per_env = total_samples // num_envs
            sample_indices = torch.arange(total_samples, device=self.device) // samples_per_env

        # Initialize result tensor
        optimized_trajs = torch.zeros_like(mean_trajs)

        # Optimize each environment's trajectory
        for env_idx in range(num_envs):
            # Get indices for this environment's samples
            env_mask = (sample_indices == env_idx)
            if not torch.any(env_mask):
                # If no samples for this env, keep the mean trajectory
                optimized_trajs[env_idx] = mean_trajs[env_idx]
                continue

            env_indices = env_mask.nonzero(as_tuple=True)[0]
            env_trajs = sampled_trajs_batch[env_indices]
            env_rewards = rewards_batch[env_indices]

            # Optimize this environment's trajectory
            optimized_trajs[env_idx] = self.optimize(
                mean_trajs[env_idx],
                env_trajs,
                env_rewards
            )

        return optimized_trajs

    def node2dense(self, nodes: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert control nodes to dense control sequence using the spline implementation.

        Args:
            nodes: Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]

        Returns:
            Dense control sequence [horizon_samples, action_dim] or batch [batch_size, horizon_samples, action_dim]
        """
        return self.spline.node2dense(nodes)

    def dense2node(self, dense: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert dense control sequence to control nodes using the spline implementation.

        Args:
            dense: Dense control sequence [horizon_samples, action_dim] or batch [batch_size, horizon_samples, action_dim]

        Returns:
            Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]
        """
        return self.spline.dense2node(dense)


class WeightedBasisFunctionOptimizer(OptimizerBase):
    """Implements Weighted Basis Function Optimization for trajectory optimization."""

    def __init__(self,
                 horizon_nodes: int,
                 horizon_samples: int,
                 action_dim: int,
                 temp_tau: float = 0.1,
                 temp_node: float = 1.0,
                 dt: float = 0.02,
                 device: torch.device = None):
        """Initialize the optimizer.

        Args:
            horizon_nodes: Number of knot points in the trajectory
            horizon_samples: Number of sample points in the dense trajectory
            action_dim: Dimension of the action space
            temp_tau: Temperature parameter for softmax weighting
            temp_node: Temperature parameter for node weighting
            dt: Timestep for trajectory planning (default: 0.02)
            device: Device to use for tensor operations
        """
        super().__init__(horizon_nodes, horizon_samples, action_dim, temp_tau, dt, device)
        self.temp_node = temp_node

        # Precompute the basis mask matrix
        self.phi = self.spline.compute_basis_mask_matrix(self.horizon_nodes, self.horizon_samples)

    def optimize(self,
                 mean_traj: torch.Tensor,
                 sampled_trajs: torch.Tensor,
                 rewards: torch.Tensor) -> torch.Tensor:
        """Optimize a trajectory using weighted basis function optimization.

        Args:
            mean_traj: Mean trajectory [horizon_nodes, action_dim]
            sampled_trajs: Sampled trajectories [num_samples, horizon_nodes, action_dim]
            rewards: Rewards for each trajectory [num_samples, horizon_samples]

        Returns:
            Optimized trajectory [horizon_nodes, action_dim]
        """
        num_samples = sampled_trajs.shape[0]

        # Compute W = S * Φ
        W = torch.mm(rewards, self.phi)  # [num_samples, horizon_nodes]

        # Normalize columns of W
        W = (W - W.mean(dim=0, keepdim=True)) / (W.std(dim=0, keepdim=True) + 1e-8)  # [num_samples, horizon_nodes]
        # Node level softmax weighting
        W = F.softmax(W / self.temp_tau, dim=0)  # [num_samples, horizon_nodes]

        # Optimized vectorized computation
        # Method 1: Using torch.einsum for efficient weighted sum
        # W: [num_samples, horizon_nodes], sampled_trajs: [num_samples, horizon_nodes, action_dim]
        # Result: [horizon_nodes, action_dim]
        updated_traj = torch.einsum('sn,sna->na', W, sampled_trajs)

        return updated_traj

    def debug_draw(self, Mat):
        """Debug visualization of matrices."""
        import matplotlib.pyplot as plt
        # Visualize the basis function matrix (self.phi)
        plt.figure(figsize=(10, 6))
        plt.imshow(Mat.cpu().numpy(), aspect='auto', cmap='viridis')
        plt.colorbar(label='clr')
        plt.show()


class ActionValueWBFO(WeightedBasisFunctionOptimizer):
    """Action-Value Weighted Basis Function Optimization with discount factor."""

    def __init__(self,
                 horizon_nodes: int,
                 horizon_samples: int,
                 action_dim: int,
                 gamma: float = 0.99,
                 temp_tau: float = 0.1,
                 temp_node: float = 1.0,
                 dt: float = 0.02,
                 device: torch.device = None):
        """Initialize the ActionValueWBFO optimizer.

        Args:
            horizon_nodes: Number of knot points in the trajectory
            horizon_samples: Number of sample points in the dense trajectory
            action_dim: Dimension of the action space
            gamma: Discount factor for future rewards
            temp_tau: Temperature parameter for softmax weighting
            temp_node: Temperature parameter for node weighting
            dt: Timestep for trajectory planning (default: 0.02)
            device: Device to use for tensor operations
        """
        super().__init__(horizon_nodes, horizon_samples, action_dim, temp_tau, temp_node, dt, device)
        self.gamma = gamma
        self.discount_factor = torch.tensor([self.gamma ** i for i in range(horizon_samples)], device=self.device)

    def optimize(self,
                 mean_traj: torch.Tensor,
                 sampled_trajs: torch.Tensor,
                 rewards: torch.Tensor) -> torch.Tensor:
        """Optimize a trajectory using weighted basis function optimization.

        Args:
            mean_traj: Mean trajectory [horizon_nodes, action_dim]
            sampled_trajs: Sampled trajectories [num_samples, horizon_nodes, action_dim]
            rewards: Rewards for each trajectory [num_samples, horizon_samples]

        Returns:
            Optimized trajectory [horizon_nodes, action_dim]
        """
        num_samples = sampled_trajs.shape[0]

        # Compute W = Σ * S * Φ
        W = torch.mm(rewards, self.phi)  # [num_samples, horizon_nodes]

        # Then apply discounted cumulative calculation to W (node-wise instead of step-wise)
        discounted_cum_W = torch.zeros_like(W)

        if self.gamma == 0.0:
            # When gamma == 0, discounted cumulative W are just the immediate W values
            discounted_cum_W = W
        elif self.gamma < 1.0:
            # For each node k, calculate: value(i,k) = W(i,k) + gamma*W(i,k+1) + gamma^2*W(i,k+2) + ...
            # Start from the last node and work backwards
            discounted_cum_W[:, -1] = W[:, -1]  # Last node has no future nodes
            # Work backwards through nodes
            for k in range(self.horizon_nodes - 2, -1, -1):
                discounted_cum_W[:, k] = W[:, k] + self.gamma * discounted_cum_W[:, k + 1]
        elif self.gamma == 1.0:
            # When gamma == 1.0, discounted cumulative W are just cumulative sums from right to left
            # Use torch.cumsum with flipped tensor for efficient computation
            flipped_W = torch.flip(W, dims=[1])  # Flip along node dimension
            flipped_cumsum = torch.cumsum(flipped_W, dim=1)  # Cumulative sum
            discounted_cum_W = torch.flip(flipped_cumsum, dims=[1])  # Flip back to original order

        # Use the discounted cumulative W for final processing
        W = discounted_cum_W
        # Normalize columns of W
        W = (W - W.mean(dim=0, keepdim=True)) / (W.std(dim=0, keepdim=True) + 1e-8)  # [num_samples, horizon_nodes]
        # Node level softmax weighting
        W = F.softmax(W / self.temp_tau, dim=0)  # [num_samples, horizon_nodes]

        # Optimized vectorized computation instead of loop
        # Method 1: Using torch.einsum for efficient weighted sum
        # W: [num_samples, horizon_nodes], sampled_trajs: [num_samples, horizon_nodes, action_dim]
        # Result: [horizon_nodes, action_dim]
        updated_traj = torch.einsum('sn,sna->na', W, sampled_trajs)

        return updated_traj


class MPPIOptimizer(OptimizerBase):
    """Model Predictive Path Integral (MPPI) optimizer for trajectory optimization."""

    def __init__(self,
                 horizon_nodes: int,
                 horizon_samples: int,
                 action_dim: int,
                 temp_tau: float = 0.1,
                 dt: float = 0.02,
                 device: torch.device = None):
        """Initialize the MPPI optimizer.

        Args:
            horizon_nodes: Number of knot points in the trajectory
            horizon_samples: Number of sample points in the dense trajectory
            action_dim: Dimension of the action space
            temp_tau: Temperature parameter for softmax weighting
            dt: Timestep for trajectory planning (default: 0.02)
            device: Device to use for tensor operations
        """
        super().__init__(horizon_nodes, horizon_samples, action_dim, temp_tau, dt, device)

    def optimize(self,
                 mean_traj: torch.Tensor,
                 sampled_trajs: torch.Tensor,
                 rewards: torch.Tensor) -> torch.Tensor:
        """Optimize a trajectory using MPPI.

        Args:
            mean_traj: Mean trajectory [horizon_nodes, action_dim]
            sampled_trajs: Sampled trajectories [num_samples, horizon_nodes, action_dim]
            rewards: Rewards for each trajectory [num_samples, horizon_samples]

        Returns:
            Optimized trajectory [horizon_nodes, action_dim]
        """
        # Calculate mean reward from the mean trajectory (first sample)
        mean_reward = rewards[0].mean(dim=-1)
        
        # Calculate total rewards for each sample
        rews = rewards.mean(dim=-1)  # [num_samples]

        # Calculate weights using MPPI formula
        reward_std = torch.std(rews) + 1e-8
        logp = (rews - mean_reward) / reward_std / self.temp_tau
        weights = F.softmax(logp, dim=0)  # [num_samples]

        # Update trajectory using weighted average
        updated_traj = torch.einsum("n,nij->ij", weights, sampled_trajs)

        return updated_traj


def create_wbfo_optimizer(cfg):
    """Create a WeightedBasisFunctionOptimizer from configuration.

    Args:
        cfg: Configuration object with trajectory_opt settings

    Returns:
        WeightedBasisFunctionOptimizer instance
    """
    # Extract relevant parameters from config
    horizon_nodes = cfg.trajectory_opt.horizon_nodes
    horizon_samples = cfg.trajectory_opt.horizon_samples
    action_dim = getattr(cfg, 'num_actions', getattr(cfg.env, 'num_actions', 12))  # Default to 12 if not specified
    temp_tau = cfg.trajectory_opt.temp_sample
    dt = getattr(cfg.trajectory_opt, 'dt', 0.02)  # Default timestep

    # Set device if specified in config
    device = torch.device(cfg.sim_device) if hasattr(cfg, "sim_device") else None

    # Create and return optimizer
    return WeightedBasisFunctionOptimizer(
        horizon_nodes=horizon_nodes+1,
        horizon_samples=horizon_samples+1,
        action_dim=action_dim,
        temp_tau=temp_tau,
        dt=dt,
        device=device
    )


def create_avwbfo_optimizer(cfg):
    """Create an ActionValueWBFO optimizer from configuration.

    Args:
        cfg: Configuration object with trajectory_opt settings

    Returns:
        ActionValueWBFO instance
    """
    # Extract relevant parameters from config
    horizon_nodes = cfg.trajectory_opt.horizon_nodes
    horizon_samples = cfg.trajectory_opt.horizon_samples
    action_dim = getattr(cfg, 'num_actions', getattr(cfg.env, 'num_actions', 12))  # Default to 12 if not specified
    temp_tau = cfg.trajectory_opt.temp_sample
    gamma = getattr(cfg.trajectory_opt, 'gamma', 0.99)  # Discount factor, default to 0.99
    dt = getattr(cfg.trajectory_opt, 'dt', 0.02)  # Default timestep

    # Set device if specified in config
    device = torch.device(cfg.sim_device) if hasattr(cfg, "sim_device") else None

    # Create and return optimizer
    return ActionValueWBFO(
        horizon_nodes=horizon_nodes+1,
        horizon_samples=horizon_samples+1,
        action_dim=action_dim,
        temp_tau=temp_tau,
        gamma=gamma,
        dt=dt,
        device=device
    )


def create_mppi_optimizer(cfg):
    """Create an MPPI optimizer from configuration.

    Args:
        cfg: Configuration object with trajectory_opt settings

    Returns:
        MPPIOptimizer instance
    """
    # Extract relevant parameters from config
    horizon_nodes = cfg.trajectory_opt.horizon_nodes
    horizon_samples = cfg.trajectory_opt.horizon_samples
    action_dim = getattr(cfg, 'num_actions', getattr(cfg.env, 'num_actions', 12))  # Default to 12 if not specified
    temp_tau = cfg.trajectory_opt.temp_sample
    dt = getattr(cfg.trajectory_opt, 'dt', 0.02)  # Default timestep

    # Set device if specified in config
    device = torch.device(cfg.sim_device) if hasattr(cfg, "sim_device") else None

    # Create and return optimizer
    return MPPIOptimizer(
        horizon_nodes=horizon_nodes+1,
        horizon_samples=horizon_samples+1,
        action_dim=action_dim,
        temp_tau=temp_tau,
        dt=dt,
        device=device
    )
