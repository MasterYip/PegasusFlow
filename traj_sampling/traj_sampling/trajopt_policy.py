"""
Trajectory optimization policy module.

This module provides abstract base classes and concrete implementations for 
trajectory optimization policies that can be easily replaced and tested.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, Union, List, Tuple
from enum import Enum
import pickle
import os
from collections import defaultdict


from .noise_scheduler import (
    NoiseSchedulerBase, S2NoiseScheduler, S3NoiseScheduler, HierarchicalNoiseScheduler,
    create_s2_scheduler, create_s3_scheduler, create_noise_scheduler, create_hierarchical_scheduler,
    constant_decay, linear_decay, exponential_decay, cosine_decay
)
from .noise_sampler import (
    sample_noise, sample_normal_noise, sample_uniform_noise,
    NoiseSamplerFactory, NoiseDistribution, MonteCarloSampler, LatinHypercubeSampler
)


class TrajOptMode(Enum):
    """Trajectory optimization modes."""
    TRAJ = "traj"  # Direct trajectory output: traj_new = policy(traj, obs)
    DELTA_TRAJ = "delta_traj"  # Delta trajectory output: traj_new = traj + policy(traj, obs)


class TrajOptPolicyBase(ABC):
    """Base class for trajectory optimization policies.

    This abstract class defines the interface for trajectory optimization policies
    that can operate in different modes (direct trajectory or delta trajectory).
    """

    def __init__(self,
                 mode: TrajOptMode = TrajOptMode.TRAJ,
                 device: Optional[torch.device] = None,
                 noise_scheduler = None):
        """Initialize trajectory optimization policy.

        Args:
            mode: Trajectory optimization mode (TRAJ or DELTA_TRAJ)
            device: Device for computations
            noise_scheduler: Noise scheduler for trajectory optimization
        """
        self.mode = mode
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_scheduler = noise_scheduler

    @abstractmethod
    def forward(self,
                traj: torch.Tensor,
                obs: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass of the trajectory optimization policy.

        Args:
            traj: Current trajectory [batch_size, horizon, action_dim] or [horizon, action_dim]
            obs: Optional observations [batch_size, obs_dim] or [obs_dim]
            **kwargs: Additional arguments specific to the policy

        Returns:
            Optimized trajectory based on the mode:
            - TRAJ mode: traj_new = policy(traj, obs)
            - DELTA_TRAJ mode: traj_new = traj + policy(traj, obs)
        """
        pass

    @abstractmethod
    def optimize_trajectories(self,
                              trajectories: torch.Tensor,
                              rollout_callback: Optional[Callable] = None,
                              n_diffuse: Optional[int] = None,
                              obs: Optional[torch.Tensor] = None,
                              **kwargs) -> torch.Tensor:
        """Optimize multiple trajectories using the policy.

        This method provides a unified interface for trajectory optimization
        across different policy implementations (sampling, transformer, etc.).

        Args:
            trajectories: Input trajectories [batch_size, horizon_nodes, action_dim]
            rollout_callback: Function to perform batch rollout (may not be used by all policies)
            n_diffuse: Number of diffusion/optimization steps (may not be used by all policies)
            obs: Optional observations [batch_size, obs_dim] or [obs_dim]
            **kwargs: Additional arguments for policy-specific optimization

        Returns:
            Optimized trajectories [batch_size, horizon_nodes, action_dim]
        """
        pass

    def __call__(self,
                 traj: torch.Tensor,
                 obs: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """Call the policy."""
        return self.forward(traj, obs, **kwargs)

    def set_mode(self, mode: TrajOptMode):
        """Set the trajectory optimization mode."""
        self.mode = mode

    def set_rollout_callback(self, rollout_callback: Callable):
        """Set the rollout callback function.
        
        Args:
            rollout_callback: Function to perform batch rollout
        """
        # Default implementation - subclasses can override if needed
        pass

    def to(self, device: torch.device):
        """Move policy to device."""
        self.device = device
        if self.noise_scheduler is not None:
            self.noise_scheduler.to(device)
        return self

    def _create_noise_scheduler_from_config(self, config, horizon_nodes: int, action_dim: int):
        """Create noise scheduler from configuration.
        
        Args:
            config: Configuration object with noise scheduler settings
            horizon_nodes: Number of trajectory nodes
            action_dim: Action dimension
            
        Returns:
            Noise scheduler instance or None if not available
        """
        if not hasattr(config, 'noise_scheduler_type') or NoiseSchedulerBase is None:
            return None
            
        scheduler_type = config.noise_scheduler_type
        
        if scheduler_type == "s2":
            return create_s2_scheduler(
                horizon_nodes=horizon_nodes,
                action_dim=action_dim,
                shape=getattr(config, 'noise_shape_fn', 'sine'),
                decay=getattr(config, 'noise_decay_fn', 'exponential'),
                base_scale=getattr(config, 'noise_base_scale', 1.0),
                device=self.device,
                **getattr(config, 'noise_decay_kwargs', {})
            )
        elif scheduler_type == "s3":
            return create_s3_scheduler(
                horizon_nodes=horizon_nodes,
                action_dim=action_dim,
                dim_scale=getattr(config, 'noise_dim_scale', None),
                shape=getattr(config, 'noise_shape_fn', 'sine'),
                decay=getattr(config, 'noise_decay_fn', 'exponential'),
                base_scale=getattr(config, 'noise_base_scale', 1.0),
                device=self.device,
                **getattr(config, 'noise_decay_kwargs', {})
            )
        elif scheduler_type == "hierarchical":
            # Check if we should use predefined activation pattern or custom parameters
            if config.noise_hierarchical_activation_pattern is not None:
                return create_hierarchical_scheduler(
                    horizon_nodes=horizon_nodes,
                    action_dim=action_dim,
                    activation_pattern=getattr(config, 'noise_hierarchical_activation_pattern', 'staggered'),
                    dim_scale=getattr(config, 'noise_dim_scale', None),
                    decay=getattr(config, 'noise_decay_fn', 'exponential'),
                    base_scale=getattr(config, 'noise_base_scale', 1.0),
                    device=self.device,
                    **getattr(config, 'noise_decay_kwargs', {})
                )
            elif HierarchicalNoiseScheduler is not None:
                # Use custom activation parameters if available
                decay_fns = {
                    "constant": constant_decay,
                    "linear": linear_decay,
                    "exponential": exponential_decay,
                    "cosine": cosine_decay
                }
                return HierarchicalNoiseScheduler(
                    horizon_nodes=horizon_nodes,
                    action_dim=action_dim,
                    dim_scale=getattr(config, 'noise_dim_scale', None),
                    activate_start=getattr(config, 'noise_hierarchical_activate_start', None),
                    activate_len=getattr(config, 'noise_hierarchical_activate_len', None),
                    decay_fn=decay_fns.get(getattr(config, 'noise_decay_fn', 'exponential'), exponential_decay),
                    base_scale=getattr(config, 'noise_base_scale', 1.0),
                    device=self.device,
                    **getattr(config, 'noise_decay_kwargs', {})
                )
            else:
                return None
        else:
            # Use the general factory function for other types
            return create_noise_scheduler(
                schedule_type=scheduler_type,
                horizon_nodes=horizon_nodes,
                action_dim=action_dim,
                device=self.device,
                **getattr(config, 'noise_decay_kwargs', {})
            )


class TrajOptPolicyTF(TrajOptPolicyBase, nn.Module):
    """Transformer-based trajectory optimization policy.

    This policy uses a transformer architecture to optimize trajectories based on
    observations only, without requiring rollout callbacks for sampling.
    """

    def __init__(self,
                 horizon_nodes: int,
                 action_dim: int,
                 obs_dim: Optional[int] = None,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 mode: TrajOptMode = TrajOptMode.DELTA_TRAJ,
                 use_obs: bool = True,
                 use_noise_conditioning: bool = False,
                 device: Optional[torch.device] = None,
                 noise_scheduler: Optional['NoiseSchedulerBase'] = None,
                 config=None):
        """Initialize transformer trajectory optimization policy.

        Args:
            horizon_nodes: Number of trajectory nodes
            action_dim: Action dimension
            obs_dim: Observation dimension (required if use_obs=True)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            mode: Trajectory optimization mode
            use_obs: Whether to use observations as input
            use_noise_conditioning: Whether to use noise scheduling as conditioning
            device: Device for computations
            noise_scheduler: Noise scheduler for trajectory optimization
            config: Configuration object (can be used to create noise scheduler)
        """
        # Create noise scheduler from config if provided and not explicitly passed
        if noise_scheduler is None and config is not None:
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            noise_scheduler = self._create_noise_scheduler_from_config(config, horizon_nodes, action_dim)
            
        TrajOptPolicyBase.__init__(self, mode, device, noise_scheduler)
        nn.Module.__init__(self)

        self.horizon_nodes = horizon_nodes
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.use_obs = use_obs
        self.use_noise_conditioning = use_noise_conditioning

        if use_obs and obs_dim is None:
            raise ValueError("obs_dim must be specified when use_obs=True")

        # Input embeddings
        self.traj_embed = nn.Linear(action_dim, d_model).to(device)
        if use_obs:
            self.obs_embed = nn.Linear(obs_dim, d_model).to(device)
        if use_noise_conditioning:
            self.noise_embed = nn.Linear(action_dim, d_model).to(device)  # Noise conditioning embedding

        # Positional encoding for trajectory
        self.pos_encoding = nn.Parameter(torch.randn(horizon_nodes, d_model, device=device))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        ).to(device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

        # Output projection
        self.output_proj = nn.Linear(d_model, action_dim).to(device)

        # Initialize weights
        self._init_weights()

        # Ensure the entire model is on the correct device
        self.to(self.device)

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                traj: torch.Tensor,
                obs: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                iteration: Optional[int] = None,
                max_iterations: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass of transformer policy.

        Args:
            traj: Current trajectory [batch_size, horizon_nodes, action_dim] or [horizon_nodes, action_dim]
            obs: Observations [batch_size, obs_dim] or [obs_dim] (required if use_obs=True)
            mask: Optional attention mask [batch_size, seq_len] or [seq_len]
            iteration: Current optimization iteration (for noise conditioning)
            max_iterations: Maximum number of iterations (for noise conditioning)

        Returns:
            Optimized trajectory
        """
        # # Ensure input is on correct device
        # traj = traj.to(self.device)
        # if obs is not None:
        #     obs = obs.to(self.device)
        # if mask is not None:
        #     mask = mask.to(self.device)
        
        # Handle single trajectory input
        if traj.dim() == 2:
            traj = traj.unsqueeze(0)  # Add batch dimension
            single_batch = True
        else:
            single_batch = False

        batch_size, seq_len, _ = traj.shape

        # Check if sequence length matches expected horizon_nodes
        if seq_len != self.horizon_nodes:
            raise ValueError(f"Expected trajectory length {self.horizon_nodes}, got {seq_len}")

        # Check observation requirements
        if self.use_obs:
            if obs is None:
                raise ValueError("Observations required when use_obs=True")
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)  # Add batch dimension

        # Embed trajectory
        traj_embedded = self.traj_embed(traj)  # [batch_size, horizon_nodes, d_model]

        # Add noise conditioning if enabled
        if self.use_noise_conditioning and self.noise_scheduler is not None:
            if iteration is not None and max_iterations is not None:
                # Get noise scale from scheduler
                noise_scale = self.noise_scheduler.get_noise_scale(iteration, max_iterations)  # [horizon_nodes, action_dim]
                
                # Embed noise scale and add to trajectory embedding
                noise_embedded = self.noise_embed(noise_scale.unsqueeze(0))  # [1, horizon_nodes, d_model]
                traj_embedded = traj_embedded + noise_embedded
        
        # Add positional encoding
        traj_embedded = traj_embedded + self.pos_encoding.unsqueeze(0)

        # Prepare transformer input
        if self.use_obs:
            # Embed observations
            obs_embedded = self.obs_embed(obs)  # [batch_size, d_model]
            obs_embedded = obs_embedded.unsqueeze(1)  # [batch_size, 1, d_model]

            # Concatenate obs and trajectory
            transformer_input = torch.cat([obs_embedded, traj_embedded], dim=1)  # [batch_size, 1+horizon_nodes, d_model]

            # Adjust mask if provided
            if mask is not None:
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                # Add mask for observation token
                obs_mask = torch.zeros(batch_size, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([obs_mask, mask], dim=1)
        else:
            transformer_input = traj_embedded

        # Apply transformer
        transformer_output = self.transformer(transformer_input, src_key_padding_mask=mask)

        # Extract trajectory part (skip observation token if present)
        if self.use_obs:
            traj_output = transformer_output[:, 1:, :]  # Skip first token (observation)
        else:
            traj_output = transformer_output

        # Project to action space
        delta_traj = self.output_proj(traj_output)  # [batch_size, horizon_nodes, action_dim]

        # Apply mode-specific output
        if self.mode == TrajOptMode.TRAJ:
            result = delta_traj
        else:  # DELTA_TRAJ mode
            result = traj + delta_traj

        # Remove batch dimension if input was single trajectory
        if single_batch:
            result = result.squeeze(0)

        return result

    def optimize_trajectories(self,
                              trajectories: torch.Tensor,
                              rollout_callback: Optional[Callable] = None,
                              n_diffuse: Optional[int] = None,
                              obs: Optional[torch.Tensor] = None,
                              **kwargs) -> torch.Tensor:
        """Optimize multiple trajectories using the transformer policy.

        This method provides the same interface as TrajOptPolicySampling.optimize_trajectories
        but uses the transformer to predict trajectory improvements instead of sampling.

        Args:
            trajectories: Input trajectories [batch_size, horizon_nodes, action_dim]
            rollout_callback: Function to perform batch rollout (not used by transformer)
            n_diffuse: Number of diffusion steps (not used by transformer, kept for compatibility)
            obs: Optional observations [batch_size, obs_dim] or [obs_dim]
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Optimized trajectories [batch_size, horizon_nodes, action_dim]
        """
        # Ensure input is on correct device
        trajectories = trajectories.to(self.device)
        if obs is not None:
            obs = obs.to(self.device)

        # Handle single trajectory input
        if trajectories.dim() == 2:
            trajectories = trajectories.unsqueeze(0)  # Add batch dimension
            single_batch = True
        else:
            single_batch = False

        batch_size, horizon, action_dim = trajectories.shape

        # Validate input dimensions
        if horizon != self.horizon_nodes or action_dim != self.action_dim:
            raise ValueError(f"Expected trajectory shape [{self.horizon_nodes}, {self.action_dim}], "
                             f"got [{horizon}, {action_dim}]")

        # Handle observations for batch
        if self.use_obs:
            if obs is None:
                raise ValueError("Observations required when transformer uses observations")
            
            # If obs is 1D, expand to match batch size
            if obs.dim() == 1:
                obs = obs.unsqueeze(0).expand(batch_size, -1)
            elif obs.dim() == 2 and obs.shape[0] == 1 and batch_size > 1:
                obs = obs.expand(batch_size, -1)
        
        # Use transformer to predict trajectory improvements
        with torch.no_grad():
            if self.use_obs:
                prediction = self.forward(trajectories, obs)
            else:
                prediction = self.forward(trajectories)
            
            # Apply mode-specific logic for optimization
            if self.mode == TrajOptMode.TRAJ:
                # In TRAJ mode, the transformer directly predicts the optimized trajectory
                optimized = prediction
            else:  # DELTA_TRAJ mode
                # In DELTA_TRAJ mode, the transformer predicts deltas to add to the input
                optimized = trajectories + prediction

        # Remove batch dimension if input was single trajectory
        if single_batch:
            optimized = optimized.squeeze(0)

        return optimized

    def set_rollout_callback(self, rollout_callback: Callable):
        """Set the rollout callback function.
        
        Note: Transformer policy doesn't use rollout callbacks, but this method
        is provided for compatibility with the sampling policy interface.
        
        Args:
            rollout_callback: Function to perform batch rollout (ignored)
        """
        # Transformer doesn't use rollout callbacks, but store for compatibility
        self.rollout_callback = rollout_callback


class TrajOptPolicySampling(TrajOptPolicyBase):
    """Sampling-based trajectory optimization policy.

    This policy uses batch rollout and sampling methods (MPPI, WBFO, AVWBFO) 
    for trajectory optimization. It migrates the existing sampling logic.
    """

    def __init__(self,
                 horizon_nodes: int,
                 action_dim: int,
                 num_samples: int = 64,
                 num_diffuse_steps: int = 5,
                 temp_sample: float = 0.1,
                 noise_scaling: float = 0.4,
                 traj_diffuse_factor: float = 0.5,
                 horizon_diffuse_factor: float = 0.9,
                 update_method: str = "avwbfo",  # "mppi", "wbfo", "avwbfo"
                 horizon_samples: int = 100,  # Added horizon_samples parameter
                 gamma: float = 0.99,  # Added gamma parameter for AVWBFO
                 dt: float = 0.02,  # Added dt parameter
                 mode: TrajOptMode = TrajOptMode.TRAJ,
                 device: Optional[torch.device] = None,
                 noise_scheduler: Optional['NoiseSchedulerBase'] = None,
                 noise_sampler_type: str = 'lhs',  # Added noise sampler type
                 noise_distribution: str = 'normal',  # Added noise distribution
                 noise_sampler_seed: Optional[int] = None,  # Added noise sampler seed
                 config=None):
        """Initialize sampling-based trajectory optimization policy.

        Args:
            horizon_nodes: Number of trajectory nodes
            action_dim: Action dimension
            num_samples: Number of samples for optimization
            num_diffuse_steps: Number of diffusion steps
            temp_sample: Temperature for sampling
            noise_scaling: Scaling factor for noise (used for fallback)
            traj_diffuse_factor: Trajectory diffusion factor (used for fallback)
            horizon_diffuse_factor: Horizon diffusion factor (used for fallback)
            update_method: Update method ("mppi", "wbfo", "avwbfo")
            horizon_samples: Number of samples in dense trajectory (for WBFO/AVWBFO)
            gamma: Discount factor for AVWBFO
            dt: Timestep for trajectory planning (default: 0.02)
            mode: Trajectory optimization mode
            device: Device for computations
            noise_scheduler: Noise scheduler for trajectory optimization
            noise_sampler_type: Type of noise sampler ('mc', 'lhs', 'halton', None)
            noise_distribution: Distribution for noise sampling ('normal', 'uniform')
            noise_sampler_seed: Random seed for noise sampler
            config: Configuration object (can be used to create noise scheduler)
        """
        # Create noise scheduler from config if provided and not explicitly passed
        if noise_scheduler is None and config is not None and config.noise_scheduler_type is not None:
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            noise_scheduler = self._create_noise_scheduler_from_config(config, horizon_nodes, action_dim)
        
        super().__init__(mode, device, noise_scheduler)    

        self.horizon_nodes = horizon_nodes
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.num_diffuse_steps = num_diffuse_steps
        self.temp_sample = temp_sample
        self.noise_scaling = noise_scaling
        self.traj_diffuse_factor = traj_diffuse_factor
        self.horizon_diffuse_factor = horizon_diffuse_factor
        self.update_method = update_method
        self.horizon_samples = horizon_samples
        self.gamma = gamma
        self.dt = dt

        # Initialize noise sampler (handle None case)
        self.noise_sampler_type = noise_sampler_type
        self.noise_distribution = NoiseDistribution(noise_distribution) if noise_distribution else None
        self.noise_sampler_seed = noise_sampler_seed
        
        if noise_sampler_type is not None:
            self.noise_sampler = NoiseSamplerFactory.create_sampler(
                sampler_type=noise_sampler_type,
                distribution=self.noise_distribution,
                device=self.device,
                seed=noise_sampler_seed
            )
        else:
            self.noise_sampler = None  # Use fallback torch.randn

        # Initialize noise schedule (fallback if no noise scheduler provided)
        self._init_fallback_noise_schedule()

        # Initialize optimizer if available
        self.optimizer = None
        self._init_optimizer()

    def _init_fallback_noise_schedule(self):
        """Initialize fallback noise schedule for diffusion (backward compatibility)."""
        # Create sigma control schedule for backward compatibility when no noise scheduler is provided
        self.sigma_control = torch.flip(
            self.horizon_diffuse_factor ** torch.arange(self.horizon_nodes, device=self.device),
            dims=[0]
        ) * self.noise_scaling

    def _init_optimizer(self):
        """Initialize the specific optimizer (WBFO, AVWBFO, etc.)."""
        if self.update_method == "mppi":
            # Create MPPI optimizer
            self.optimizer = self._create_mppi_optimizer()
        elif self.update_method == "wbfo":
            # Create WBFO optimizer
            self.optimizer = self._create_wbfo_optimizer()
        elif self.update_method == "avwbfo":
            # Create AVWBFO optimizer
            self.optimizer = self._create_avwbfo_optimizer()
        else:
            raise ValueError(f"Unknown update method: {self.update_method}")

    def _create_wbfo_optimizer(self):
        """Create a WBFO optimizer with current parameters."""
        from.optimizer import WeightedBasisFunctionOptimizer

        return WeightedBasisFunctionOptimizer(
            horizon_nodes=self.horizon_nodes,
            horizon_samples=self.horizon_samples,
            action_dim=self.action_dim,
            temp_tau=self.temp_sample,
            dt=self.dt,
            device=self.device
        )

    def _create_mppi_optimizer(self):
        """Create an MPPI optimizer with current parameters."""
        from.optimizer import MPPIOptimizer

        return MPPIOptimizer(
            horizon_nodes=self.horizon_nodes,
            horizon_samples=self.horizon_samples,
            action_dim=self.action_dim,
            temp_tau=self.temp_sample,
            dt=self.dt,
            device=self.device
        )

    def _create_avwbfo_optimizer(self):
        """Create an AVWBFO optimizer with current parameters."""
        from.optimizer import ActionValueWBFO

        return ActionValueWBFO(
            horizon_nodes=self.horizon_nodes,
            horizon_samples=self.horizon_samples,
            action_dim=self.action_dim,
            temp_tau=self.temp_sample,
            gamma=self.gamma,
            dt=self.dt,
            device=self.device
        )

    def forward(self,
                traj: torch.Tensor,
                obs: Optional[torch.Tensor] = None,
                rollout_callback: Optional[Callable] = None,
                noise_scale: Optional[torch.Tensor] = None,
                n_samples: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass of sampling policy.

        Args:
            traj: Current trajectory [batch_size, horizon_nodes, action_dim] or [horizon_nodes, action_dim]
            obs: Optional observations (not used in sampling policy)
            rollout_callback: Function to perform batch rollout (required)
            noise_scale: Optional noise scale
            n_samples: Optional number of samples

        Returns:
            Optimized trajectory
        """
        if rollout_callback is None:
            raise ValueError("rollout_callback is required for sampling-based policy")

        # Handle single trajectory input
        if traj.dim() == 2:
            traj = traj.unsqueeze(0)  # Add batch dimension
            single_batch = True
        else:
            single_batch = False

        batch_size, horizon, action_dim = traj.shape

        if horizon != self.horizon_nodes or action_dim != self.action_dim:
            raise ValueError(f"Expected trajectory shape [{self.horizon_nodes}, {self.action_dim}], "
                             f"got [{horizon}, {action_dim}]")

        # Use default parameters if not specified
        if noise_scale is None:
            if self.noise_scheduler is not None:
                # Use noise scheduler for first iteration (we'll update per diffusion step)
                noise_scale = self.noise_scheduler.get_noise_scale(0, self.num_diffuse_steps)
            else:
                # Fallback to old sigma control
                noise_scale = self.sigma_control
        if n_samples is None:
            n_samples = self.num_samples

        # Perform diffusion-based optimization
        curr_trajs = traj.clone()

        # BUG: Delta traj logic not correct
        for i in range(self.num_diffuse_steps):
            # Calculate noise scale for this diffusion step
            if self.noise_scheduler is not None:
                # Use noise scheduler
                step_noise_scale = self.noise_scheduler.get_noise_scale(i, self.num_diffuse_steps)
                # from matplotlib import pyplot as plt
                # plt.plot(step_noise_scale.cpu().numpy(), label=f"Step {i}")
                # plt.legend()
                # plt.show()
            else:
                # Fallback to old behavior
                step_noise_scale = noise_scale * (self.traj_diffuse_factor ** i)

            # Perform batch gradient evaluation and update
            curr_trajs = self._eval_traj_grad_batch(
                curr_trajs, rollout_callback, step_noise_scale, n_samples
            )

        # Apply mode-specific output
        if self.mode == TrajOptMode.TRAJ:
            result = curr_trajs
        else:  # DELTA_TRAJ mode
            result = curr_trajs - traj

        # Remove batch dimension if input was single trajectory
        if single_batch:
            result = result.squeeze(0)

        return result

    def _eval_traj_grad_batch(self,
                              mean_trajs: torch.Tensor,
                              rollout_callback: Callable,
                              noise_scale: torch.Tensor,
                              n_samples: int) -> torch.Tensor:
        """Evaluate trajectory gradients for batch of trajectories.

        Args:
            mean_trajs: Mean trajectories [batch_size, horizon_nodes, action_dim]
            rollout_callback: Function to perform batch rollout
            noise_scale: Noise scale tensor
            n_samples: Number of samples per trajectory

        Returns:
            Updated trajectories [batch_size, horizon_nodes, action_dim]
        """
        batch_size = mean_trajs.shape[0]
        total_samples = (n_samples + 1) * batch_size

        # Initialize batch of sampled trajectories
        all_samples = torch.zeros(
            (total_samples, self.horizon_nodes, self.action_dim),
            device=self.device
        )

        # Generate noise samples using the configured noise sampler or fallback
        eps_shape = (total_samples, self.horizon_nodes, self.action_dim)
        
        if self.noise_sampler is not None:
            # Use configured noise sampler
            if self.noise_distribution == NoiseDistribution.NORMAL:
                eps = self.noise_sampler.sample(eps_shape, mean=0.0, std=1.0)
            elif self.noise_distribution == NoiseDistribution.UNIFORM:
                eps = self.noise_sampler.sample(eps_shape, low=-1.0, high=1.0)
            else:
                # Fallback to standard random sampling
                eps = torch.randn(eps_shape, device=self.device)
        else:
            # Use fallback torch.randn method
            eps = torch.randn(eps_shape, device=self.device)

        # Create batch data tracking
        env_sample_indices = torch.zeros(total_samples, dtype=torch.long, device=self.device)

        # Fill the batch with samples for each trajectory
        for i in range(batch_size):
            start_idx = i * (n_samples + 1)
            end_idx = (i + 1) * (n_samples + 1)
            mean_traj = mean_trajs[i]

            # Generate perturbed trajectories with hierarchical noise support
            if noise_scale.dim() == 2:
                # Hierarchical noise: [horizon_nodes, action_dim]
                samples_i = eps[start_idx:end_idx] * noise_scale[None, :, :] + mean_traj
            elif noise_scale.dim() == 1:
                # Backward compatibility: [horizon_nodes] - broadcast to action dimensions
                samples_i = eps[start_idx:end_idx] * noise_scale[None, :, None] + mean_traj
            else:
                # Scalar noise
                samples_i = eps[start_idx:end_idx] * noise_scale + mean_traj

            # Keep first control fixed (current action)
            samples_i[:, 0] = mean_traj[0]
            samples_i[0] = mean_traj  # First sample is the mean

            # Store in batch
            all_samples[start_idx:end_idx] = samples_i
            env_sample_indices[start_idx:end_idx] = i

        u_samples = self.optimizer.node2dense(all_samples)

        # Roll out all trajectories
        rewards_batch = rollout_callback(u_samples)

        # Process results for each trajectory
        updated_trajs = torch.zeros_like(mean_trajs)

        # Use specialized optimizer if available
        updated_trajs = self.optimizer.optimize_batch(
            mean_trajs, all_samples, rewards_batch, env_sample_indices
        )

        return updated_trajs

    def set_rollout_callback(self, rollout_callback: Callable):
        """Set the rollout callback function."""
        self.rollout_callback = rollout_callback

    def optimize_trajectories(self,
                              trajectories: torch.Tensor,
                              rollout_callback: Optional[Callable] = None,
                              n_diffuse: Optional[int] = None,
                              obs: Optional[torch.Tensor] = None,
                              **kwargs) -> torch.Tensor:
        """Optimize multiple trajectories with specified number of diffusion steps.

        Args:
            trajectories: Input trajectories [batch_size, horizon_nodes, action_dim]
            rollout_callback: Function to perform batch rollout
            n_diffuse: Number of diffusion steps (if None, uses default)

        Returns:
            Optimized trajectories [batch_size, horizon_nodes, action_dim]
        """
        if n_diffuse is None:
            n_diffuse = self.num_diffuse_steps

        curr_trajs = trajectories.clone()

        # Store original diffusion steps
        original_steps = self.num_diffuse_steps
        self.num_diffuse_steps = n_diffuse

        try:
            # Perform optimization
            result = self.forward(curr_trajs, rollout_callback=rollout_callback)
        finally:
            # Restore original diffusion steps
            self.num_diffuse_steps = original_steps

        return result


class TrajOptDataCollector:
    """Data collector for trajectory optimization imitation learning.
    
    This class collects input-output pairs from a teacher policy (e.g., TrajOptPolicySampling)
    to create a dataset for training a student policy (e.g., TrajOptPolicyTF).
    """
    
    def __init__(self, 
                 mode: TrajOptMode = TrajOptMode.DELTA_TRAJ,
                 max_samples: int = 10000,
                 device: Optional[torch.device] = None):
        """Initialize trajectory optimization data collector.
        
        Args:
            mode: Target mode for data collection (TRAJ or DELTA_TRAJ)
            max_samples: Maximum number of samples to collect
            device: Device for tensor operations
        """
        self.mode = mode
        self.max_samples = max_samples
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Storage for collected data
        self.trajectories_input = []  # Input trajectories
        self.observations_input = []  # Input observations (optional)
        self.trajectories_output = []  # Target trajectories (based on mode)
        self.meta_data = []  # Additional metadata
        
        self.num_samples = 0
        
    def collect_sample(self,
                       input_traj: torch.Tensor,
                       output_traj: torch.Tensor,
                       obs: Optional[torch.Tensor] = None,
                       meta: Optional[Dict[str, Any]] = None) -> None:
        """Collect a single trajectory optimization sample.
        
        Args:
            input_traj: Input trajectory [horizon_nodes, action_dim] or [batch_size, horizon_nodes, action_dim]
            output_traj: Output trajectory from teacher policy
            obs: Optional observations [obs_dim] or [batch_size, obs_dim]
            meta: Optional metadata dictionary
        """
        if self.num_samples >= self.max_samples:
            return
            
        # Handle batch inputs
        if input_traj.dim() == 3:
            batch_size = input_traj.shape[0]
            for i in range(batch_size):
                if self.num_samples >= self.max_samples:
                    break
                self._collect_single_sample(
                    input_traj[i], 
                    output_traj[i], 
                    obs[i] if obs is not None else None,
                    meta
                )
        else:
            self._collect_single_sample(input_traj, output_traj, obs, meta)
    
    def _collect_single_sample(self,
                              input_traj: torch.Tensor,
                              output_traj: torch.Tensor,
                              obs: Optional[torch.Tensor] = None,
                              meta: Optional[Dict[str, Any]] = None) -> None:
        """Collect a single sample (internal method)."""
        # Compute target based on mode
        if self.mode == TrajOptMode.TRAJ:
            target = output_traj.clone()
        else:  # DELTA_TRAJ mode
            target = output_traj - input_traj
            
        # Store data
        self.trajectories_input.append(input_traj.cpu())
        self.trajectories_output.append(target.cpu())
        
        if obs is not None:
            self.observations_input.append(obs.cpu())
        else:
            self.observations_input.append(None)
            
        self.meta_data.append(meta)
        self.num_samples += 1
    
    def get_dataset(self) -> Dict[str, Any]:
        """Get the collected dataset.
        
        Returns:
            Dictionary containing the dataset
        """
        # Filter out None observations
        has_obs = any(obs is not None for obs in self.observations_input)
        
        dataset = {
            'trajectories_input': torch.stack(self.trajectories_input) if self.trajectories_input else None,
            'trajectories_output': torch.stack(self.trajectories_output) if self.trajectories_output else None,
            'observations_input': torch.stack([obs for obs in self.observations_input if obs is not None]) if has_obs else None,
            'meta_data': self.meta_data,
            'mode': self.mode,
            'num_samples': self.num_samples,
            'has_observations': has_obs
        }
        
        return dataset
    
    def save_dataset(self, filepath: str) -> None:
        """Save the collected dataset to file.
        
        Args:
            filepath: Path to save the dataset
        """
        dataset = self.get_dataset()
        torch.save(dataset, filepath)
        print(f"Saved trajectory optimization dataset with {self.num_samples} samples to {filepath}")
    
    def load_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load a dataset from file.
        
        Args:
            filepath: Path to load the dataset from
            
        Returns:
            Loaded dataset dictionary
        """
        dataset = torch.load(filepath, map_location=self.device)
        
        # Move tensor data to the correct device
        if dataset['trajectories_input'] is not None:
            dataset['trajectories_input'] = dataset['trajectories_input'].to(self.device)
        if dataset['trajectories_output'] is not None:
            dataset['trajectories_output'] = dataset['trajectories_output'].to(self.device)
        if dataset['observations_input'] is not None:
            dataset['observations_input'] = dataset['observations_input'].to(self.device)
        
        print(f"Loaded trajectory optimization dataset with {dataset['num_samples']} samples from {filepath}")
        return dataset
    
    def clear(self) -> None:
        """Clear all collected data."""
        self.trajectories_input.clear()
        self.observations_input.clear()
        self.trajectories_output.clear()
        self.meta_data.clear()
        self.num_samples = 0
    
    def is_full(self) -> bool:
        """Check if the collector has reached maximum capacity."""
        return self.num_samples >= self.max_samples


class TrajOptTrainer:
    """Trainer for trajectory optimization policies using imitation learning.
    
    This class handles training of TrajOptPolicyTF using data collected from
    TrajOptPolicySampling or other teacher policies.
    """
    
    def __init__(self,
                 student_policy: TrajOptPolicyTF,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: Optional[torch.device] = None):
        """Initialize trajectory optimization trainer.
        
        Args:
            student_policy: The transformer policy to train
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            device: Device for computations
        """
        self.student_policy = student_policy
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move policy to device
        self.student_policy.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training metrics
        self.training_losses = []
        self.validation_losses = []
        
    def train_on_dataset(self,
                        dataset: Dict[str, Any],
                        num_epochs: int = 100,
                        batch_size: int = 32,
                        validation_split: float = 0.2,
                        print_interval: int = 10) -> Dict[str, List[float]]:
        """Train the student policy on a collected dataset.
        
        Args:
            dataset: Dataset dictionary from TrajOptDataCollector
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            print_interval: Interval for printing training progress
            
        Returns:
            Dictionary containing training and validation losses
        """
        # Extract data from dataset and move to correct device
        traj_input = dataset['trajectories_input'].to(self.device)
        traj_target = dataset['trajectories_output'].to(self.device)
        obs_input = dataset['observations_input']
        has_obs = dataset['has_observations']
        
        if has_obs and obs_input is not None:
            obs_input = obs_input.to(self.device)
        
        num_samples = dataset['num_samples']
        
        print(f"Training data moved to device: {self.device}")
        print(f"Model on device: {next(self.student_policy.parameters()).device}")
        
        # Split data into training and validation
        val_size = int(num_samples * validation_split)
        train_size = num_samples - val_size
        
        indices = torch.randperm(num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Training data
        train_traj_input = traj_input[train_indices]
        train_traj_target = traj_target[train_indices]
        train_obs_input = obs_input[train_indices] if has_obs else None
        
        # Validation data
        val_traj_input = val_traj_input[val_indices]
        val_traj_target = val_traj_target[val_indices]
        val_obs_input = val_obs_input[val_indices] if has_obs else None
        
        print(f"Training on {train_size} samples, validating on {val_size} samples")
        print(f"Input trajectory shape: {train_traj_input.shape}")
        print(f"Target trajectory shape: {train_traj_target.shape}")
        if has_obs:
            print(f"Observation shape: {train_obs_input.shape}")
        
        # Training loop
        self.student_policy.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle training data
            perm = torch.randperm(train_size)
            
            # Training batches
            for i in range(0, train_size, batch_size):
                batch_indices = perm[i:i+batch_size]
                
                batch_traj_input = train_traj_input[batch_indices]
                batch_traj_target = train_traj_target[batch_indices]
                batch_obs_input = train_obs_input[batch_indices] if has_obs else None
                
                # Ensure batch data is on correct device
                batch_traj_input = batch_traj_input.to(self.device)
                batch_traj_target = batch_traj_target.to(self.device)
                if batch_obs_input is not None:
                    batch_obs_input = batch_obs_input.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if has_obs:
                    predicted = self.student_policy(batch_traj_input, batch_obs_input)
                else:
                    predicted = self.student_policy(batch_traj_input)
                
                # Compute loss
                loss = F.mse_loss(predicted, batch_traj_target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            self.training_losses.append(avg_train_loss)
            
            # Validation
            if val_size > 0:
                val_loss = self._validate(val_traj_input, val_traj_target, val_obs_input, batch_size)
                self.validation_losses.append(val_loss)
            
            # Print progress
            if epoch % print_interval == 0:
                if val_size > 0:
                    print(f"Epoch {epoch}/{num_epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch}/{num_epochs}: Train Loss = {avg_train_loss:.6f}")
        
        print("Training completed!")
        
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
    
    def _validate(self,
                  val_traj_input: torch.Tensor,
                  val_traj_target: torch.Tensor,
                  val_obs_input: Optional[torch.Tensor],
                  batch_size: int) -> float:
        """Perform validation on validation set."""
        self.student_policy.eval()
        
        total_loss = 0.0
        num_batches = 0
        val_size = val_traj_input.shape[0]
        
        with torch.no_grad():
            for i in range(0, val_size, batch_size):
                batch_traj_input = val_traj_input[i:i+batch_size]
                batch_traj_target = val_traj_target[i:i+batch_size]
                batch_obs_input = val_obs_input[i:i+batch_size] if val_obs_input is not None else None
                
                if val_obs_input is not None:
                    predicted = self.student_policy(batch_traj_input, batch_obs_input)
                else:
                    predicted = self.student_policy(batch_traj_input)
                
                loss = F.mse_loss(predicted, batch_traj_target)
                total_loss += loss.item()
                num_batches += 1
        
        self.student_policy.train()
        return total_loss / num_batches
    
    def save_checkpoint(self, filepath: str, epoch: int, metadata: Optional[Dict] = None) -> None:
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            metadata: Optional metadata to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'metadata': metadata
        }
        
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint at epoch {epoch} to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.student_policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        self.validation_losses = checkpoint.get('validation_losses', [])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint.get('metadata', {})


class TrajOptPolicyWrapper:
    """Wrapper class to easily switch between different trajectory optimization policies."""

    def __init__(self, policy: TrajOptPolicyBase):
        """Initialize wrapper with a policy.

        Args:
            policy: The trajectory optimization policy to wrap
        """
        self.policy = policy

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Call the wrapped policy."""
        return self.policy(*args, **kwargs)

    def set_policy(self, policy: TrajOptPolicyBase):
        """Set a new policy."""
        self.policy = policy

    def set_mode(self, mode: TrajOptMode):
        """Set the trajectory optimization mode."""
        self.policy.set_mode(mode)

    def to(self, device: torch.device):
        """Move policy to device."""
        self.policy.to(device)
        return self


# Factory functions for easy policy creation
def create_transformer_policy(horizon_nodes: int,
                              action_dim: int,
                              obs_dim: Optional[int] = None,
                              config=None,
                              **kwargs) -> TrajOptPolicyTF:
    """Create a transformer-based trajectory optimization policy."""
    return TrajOptPolicyTF(horizon_nodes, action_dim, obs_dim, config=config, **kwargs)


def create_sampling_policy(horizon_nodes: int,
                           action_dim: int,
                           config=None,
                           **kwargs) -> TrajOptPolicySampling:
    """Create a sampling-based trajectory optimization policy."""
    return TrajOptPolicySampling(horizon_nodes, action_dim, config=config, **kwargs)


def create_policy_wrapper(policy: TrajOptPolicyBase) -> TrajOptPolicyWrapper:
    """Create a policy wrapper for easy switching between policies."""
    return TrajOptPolicyWrapper(policy)
