"""
Noise scheduling for trajectory optimization using variable separation.

This module provides noise scheduling strategies based on the factorized form:
f(t, dim, it) = DimScale(dim) * Shape(t) * Decay(it)

Where:
- DimScale: Per-dimension scaling factors
- Shape: Temporal shape function along trajectory
- Decay: Iteration-based decay function

Two main schedulers:
- S3NoiseScheduler: Full 3-factor separation f(t, dim, it) = DimScale(dim) * Shape(t) * Decay(it)
- S2NoiseScheduler: Simplified 2-factor separation f(t, it) = Shape(t) * Decay(it)
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum

#= Shape functions =#

@torch.jit.script
def sine_shape(x):
    """Sine shape function for fix-end-point optimization."""
    x = torch.clamp(x, 0, 1)
    return torch.sin(x * np.pi)

@torch.jit.script
def linear_shape(x):
    """Linear shape function."""
    x = torch.clamp(x, 0, 1)
    return x

@torch.jit.script
def quadratic_shape(x):
    """Quadratic shape function."""
    x = torch.clamp(x, 0, 1)
    return x ** 2

@torch.jit.script
def exponential_shape(x):
    """Exponential shape function."""
    x = torch.clamp(x, 0, 1)
    return torch.exp(x)

@torch.jit.script
def linear_activate_shape(x, activate_start: float = 0.3, activate_len: float = 0.4):
    """Linear activation shape function with configurable start and length.
    
    Maps [0,1] to [0,1] where:
    - Before activate_start: output is 0
    - From activate_start to activate_start+activate_len: linear ramp from 0 to 1
    - After activate_start+activate_len: output is 1
    
    Args:
        x: Input tensor in [0,1]
        activate_start: Start of activation region
        activate_len: Length of activation ramp
    """
    x = torch.clamp(x, 0, 1)
    activate_end = activate_start + activate_len
    
    # Before activation: 0
    result = torch.zeros_like(x)
    
    # During activation: linear ramp
    in_ramp = (x >= activate_start) & (x <= activate_end)
    if activate_len > 0:
        ramp_progress = (x[in_ramp] - activate_start) / activate_len
        result[in_ramp] = ramp_progress
    
    # After activation: 1
    after_ramp = x > activate_end
    result[after_ramp] = 1.0
    
    return result

#= Decay functions =#

@torch.jit.script
def constant_decay(iteration: int, max_iterations: int):
    """Constant decay (no decay)."""
    return 1.0

@torch.jit.script
def linear_decay(iteration: int, max_iterations: int, final_ratio: float = 0.1):
    """Linear decay from 1.0 to final_ratio."""
    if max_iterations <= 1:
        return 1.0
    progress = float(iteration) / float(max_iterations - 1)
    return 1.0 + progress * (final_ratio - 1.0)

@torch.jit.script
def exponential_decay(iteration: int, max_iterations: int, decay_rate: float = 0.9):
    """Exponential decay."""
    return decay_rate ** float(iteration)

@torch.jit.script
def cosine_decay(iteration: int, max_iterations: int, final_ratio: float = 0.01):
    """Cosine annealing decay."""
    if max_iterations <= 1:
        return 1.0
    progress = float(iteration) / float(max_iterations - 1)
    cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress)))
    return final_ratio + (1.0 - final_ratio) * float(cosine_factor)



class NoiseScheduleType(Enum):
    """Types of noise scheduling strategies."""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_DECAY = "cosine_decay"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class NoiseSchedulerBase(ABC):
    """Abstract base class for noise schedulers."""

    def __init__(self,
                 horizon_nodes: int,
                 action_dim: int,
                 device: Optional[torch.device] = None):
        """Initialize the noise scheduler.

        Args:
            horizon_nodes: Number of trajectory nodes
            action_dim: Dimension of the action space
            device: Device for tensor operations
        """
        self.horizon_nodes = horizon_nodes
        self.action_dim = action_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def get_noise_scale(self,
                        iteration: int,
                        max_iterations: int,
                        **kwargs) -> torch.Tensor:
        """Get noise scale for the current iteration.

        Args:
            iteration: Current optimization iteration
            max_iterations: Maximum number of iterations
            **kwargs: Additional arguments for specific schedulers

        Returns:
            Noise scale tensor [horizon_nodes, action_dim] or [horizon_nodes] or scalar
        """
        pass

    def to(self, device: torch.device):
        """Move scheduler to device."""
        self.device = device
        return self


class S3NoiseScheduler(NoiseSchedulerBase):
    """3-factor noise scheduler: f(t, dim, it) = DimScale(dim) * Shape(t) * Decay(it)
    
    This scheduler uses variable separation with three factors:
    - DimScale: Per-dimension scaling factors
    - Shape: Temporal shape function along trajectory  
    - Decay: Iteration-based decay function
    """

    def __init__(self,
                 horizon_nodes: int,
                 action_dim: int,
                 dim_scale: Optional[Union[float, List[float], torch.Tensor]] = None,
                 shape_fn: Callable[[torch.Tensor], torch.Tensor] = sine_shape,
                 decay_fn: Callable[[int, int], float] = exponential_decay,
                 base_scale: float = 1.0,
                 device: Optional[torch.device] = None,
                 **decay_kwargs):
        """Initialize S3 noise scheduler.

        Args:
            horizon_nodes: Number of trajectory nodes
            action_dim: Dimension of the action space
            dim_scale: Per-dimension scaling factors [action_dim] or scalar
            shape_fn: Temporal shape function
            decay_fn: Iteration decay function
            base_scale: Base scaling factor
            device: Device for tensor operations
            **decay_kwargs: Additional arguments for decay function
        """
        super().__init__(horizon_nodes, action_dim, device)
        self.shape_fn = shape_fn
        self.decay_fn = decay_fn
        self.base_scale = base_scale
        self.decay_kwargs = decay_kwargs
        
        # Set up dimension scaling
        if dim_scale is None:
            self.dim_scale = torch.ones(action_dim, device=self.device)
        elif isinstance(dim_scale, (int, float)):
            self.dim_scale = torch.full((action_dim,), float(dim_scale), device=self.device)
        elif isinstance(dim_scale, list):
            self.dim_scale = torch.tensor(dim_scale, device=self.device)
        else:
            self.dim_scale = dim_scale.to(self.device)
        
        # Create time indices for shape function
        self.time_indices = torch.linspace(0, 1, horizon_nodes, device=self.device)

    def get_noise_scale(self,
                        iteration: int,
                        max_iterations: int,
                        **kwargs) -> torch.Tensor:
        """Get noise scale using 3-factor decomposition."""
        # Shape factor [horizon_nodes]
        shape_factors = self.shape_fn(self.time_indices)
        
        # Decay factor [scalar]
        decay_factor = self.decay_fn(iteration, max_iterations, **self.decay_kwargs)
        
        # Combine: [horizon_nodes, action_dim]
        noise_scale = (shape_factors.unsqueeze(1) * 
                      self.dim_scale.unsqueeze(0) * 
                      decay_factor * 
                      self.base_scale)
        
        return noise_scale


class S2NoiseScheduler(NoiseSchedulerBase):
    """2-factor noise scheduler: f(t, it) = Shape(t) * Decay(it)
    
    This scheduler uses simplified variable separation with two factors:
    - Shape: Temporal shape function along trajectory
    - Decay: Iteration-based decay function
    """

    def __init__(self,
                 horizon_nodes: int,
                 action_dim: int,
                 shape_fn: Callable[[torch.Tensor], torch.Tensor] = sine_shape,
                 decay_fn: Callable[[int, int], float] = exponential_decay,
                 base_scale: float = 1.0,
                 device: Optional[torch.device] = None,
                 **decay_kwargs):
        """Initialize S2 noise scheduler.

        Args:
            horizon_nodes: Number of trajectory nodes
            action_dim: Dimension of the action space
            shape_fn: Temporal shape function
            decay_fn: Iteration decay function
            base_scale: Base scaling factor
            device: Device for tensor operations
            **decay_kwargs: Additional arguments for decay function
        """
        super().__init__(horizon_nodes, action_dim, device)
        self.shape_fn = shape_fn
        self.decay_fn = decay_fn
        self.base_scale = base_scale
        self.decay_kwargs = decay_kwargs
        
        # Create time indices for shape function
        self.time_indices = torch.linspace(0, 1, horizon_nodes, device=self.device)

    def get_noise_scale(self,
                        iteration: int,
                        max_iterations: int,
                        **kwargs) -> torch.Tensor:
        """Get noise scale using 2-factor decomposition."""
        # Shape factor [horizon_nodes]
        shape_factors = self.shape_fn(self.time_indices)
        
        # Decay factor [scalar]
        decay_factor = self.decay_fn(iteration, max_iterations, **self.decay_kwargs)
        
        # Combine and broadcast to [horizon_nodes, action_dim]
        noise_scale = (shape_factors * decay_factor * self.base_scale).unsqueeze(1).expand(-1, self.action_dim)
        
        return noise_scale

class AdaptiveNoiseScheduler(NoiseSchedulerBase):
    """Adaptive noise scheduler that adjusts noise based on optimization progress.
    
    This scheduler monitors the improvement in trajectory quality and adapts
    the noise scale accordingly.
    """

    def __init__(self,
                 horizon_nodes: int,
                 action_dim: int,
                 initial_scale: float = 1.0,
                 adaptation_rate: float = 0.1,
                 improvement_threshold: float = 0.01,
                 device: Optional[torch.device] = None):
        """Initialize adaptive noise scheduler.

        Args:
            horizon_nodes: Number of trajectory nodes
            action_dim: Dimension of the action space
            initial_scale: Initial noise scale
            adaptation_rate: Rate of adaptation (0 < rate < 1)
            improvement_threshold: Threshold for detecting improvement
            device: Device for tensor operations
        """
        super().__init__(horizon_nodes, action_dim, device)
        self.initial_scale = initial_scale
        self.adaptation_rate = adaptation_rate
        self.improvement_threshold = improvement_threshold
        
        # History tracking
        self.current_scale = initial_scale
        self.cost_history = []
        self.last_improvement = 0

    def get_noise_scale(self,
                        iteration: int,
                        max_iterations: int,
                        current_cost: Optional[float] = None,
                        **kwargs) -> torch.Tensor:
        """Get adaptive noise scale based on current performance."""
        if current_cost is not None:
            self.cost_history.append(current_cost)
            
            # Check for improvement
            if len(self.cost_history) >= 2:
                improvement = self.cost_history[-2] - self.cost_history[-1]
                
                if improvement > self.improvement_threshold:
                    # Good improvement - reduce noise slightly
                    self.current_scale *= (1 - self.adaptation_rate * 0.1)
                    self.last_improvement = iteration
                elif iteration - self.last_improvement > 3:
                    # No improvement for several iterations - increase noise
                    self.current_scale *= (1 + self.adaptation_rate * 0.2)
                    
        # Ensure scale doesn't go too low or too high
        self.current_scale = max(0.01, min(self.current_scale, self.initial_scale * 2))
        
        return torch.full(
            (self.horizon_nodes, self.action_dim),
            self.current_scale,
            device=self.device
        )

    def reset(self):
        """Reset the adaptive scheduler."""
        self.current_scale = self.initial_scale
        self.cost_history.clear()
        self.last_improvement = 0

class HierarchicalNoiseScheduler(NoiseSchedulerBase):
    """Hierarchical noise scheduler with per-dimension activation phases.
    
    This scheduler uses linear_activate_shape with different activation start and length
    for each action dimension, creating a hierarchical activation pattern where
    different dimensions become active at different phases of the optimization.
    """

    def __init__(self,
                 horizon_nodes: int,
                 action_dim: int,
                 dim_scale: Optional[Union[float, List[float], torch.Tensor]] = None,
                 activate_start: Optional[Union[float, List[float], torch.Tensor]] = None,
                 activate_len: Optional[Union[float, List[float], torch.Tensor]] = None,
                 decay_fn: Callable[[int, int], float] = exponential_decay,
                 base_scale: float = 1.0,
                 device: Optional[torch.device] = None,
                 **decay_kwargs):
        """Initialize hierarchical noise scheduler.

        Args:
            horizon_nodes: Number of trajectory nodes
            action_dim: Dimension of the action space
            dim_scale: Per-dimension scaling factors [action_dim] or scalar
            activate_start: Per-dimension activation start points [action_dim] or scalar
            activate_len: Per-dimension activation lengths [action_dim] or scalar
            decay_fn: Iteration decay function
            base_scale: Base scaling factor
            device: Device for tensor operations
            **decay_kwargs: Additional arguments for decay function
        """
        super().__init__(horizon_nodes, action_dim, device)
        self.decay_fn = decay_fn
        self.base_scale = base_scale
        self.decay_kwargs = decay_kwargs
        
        # Set up dimension scaling
        if dim_scale is None:
            self.dim_scale = torch.ones(action_dim, device=self.device)
        elif isinstance(dim_scale, (int, float)):
            self.dim_scale = torch.full((action_dim,), float(dim_scale), device=self.device)
        elif isinstance(dim_scale, list):
            self.dim_scale = torch.tensor(dim_scale, device=self.device)
        else:
            self.dim_scale = dim_scale.to(self.device)
        
        # Set up activation start points - default to staggered activation
        if activate_start is None:
            # Default: stagger activation starts evenly across dimensions
            self.activate_start = torch.linspace(0.0, 0.5, action_dim, device=self.device)
        elif isinstance(activate_start, (int, float)):
            self.activate_start = torch.full((action_dim,), float(activate_start), device=self.device)
        elif isinstance(activate_start, list):
            self.activate_start = torch.tensor(activate_start, device=self.device)
        else:
            self.activate_start = activate_start.to(self.device)
        
        # Set up activation lengths - default to uniform length
        if activate_len is None:
            # Default: uniform activation length for all dimensions
            self.activate_len = torch.full((action_dim,), 0.3, device=self.device)
        elif isinstance(activate_len, (int, float)):
            self.activate_len = torch.full((action_dim,), float(activate_len), device=self.device)
        elif isinstance(activate_len, list):
            self.activate_len = torch.tensor(activate_len, device=self.device)
        else:
            self.activate_len = activate_len.to(self.device)
        
        # Create time indices for shape function
        self.time_indices = torch.linspace(0, 1, horizon_nodes, device=self.device)

    def get_noise_scale(self,
                        iteration: int,
                        max_iterations: int,
                        **kwargs) -> torch.Tensor:
        """Get hierarchical noise scale with per-dimension activation phases."""
        # Decay factor [scalar]
        decay_factor = self.decay_fn(iteration, max_iterations, **self.decay_kwargs)
        
        # Shape factors for each dimension [horizon_nodes, action_dim]
        shape_factors = torch.zeros(self.horizon_nodes, self.action_dim, device=self.device)
        
        for dim in range(self.action_dim):
            # Apply linear_activate_shape with dimension-specific parameters
            dim_shape = linear_activate_shape(
                self.time_indices, 
                self.activate_start[dim].item(), 
                self.activate_len[dim].item()
            )
            shape_factors[:, dim] = dim_shape
        
        # Combine all factors: [horizon_nodes, action_dim]
        noise_scale = (shape_factors * 
                      self.dim_scale.unsqueeze(0) * 
                      decay_factor * 
                      self.base_scale)
        
        return noise_scale

# Backward compatibility - Simple scheduler classes using S2/S3
class ConstantNoiseScheduler(S2NoiseScheduler):
    """Constant noise scheduler for backward compatibility."""
    def __init__(self, horizon_nodes: int, action_dim: int, noise_scale: float = 1.0, device: Optional[torch.device] = None):
        super().__init__(horizon_nodes, action_dim, 
                        shape_fn=sine_shape,
                        decay_fn=constant_decay,
                        base_scale=noise_scale, device=device)

class LinearDecayScheduler(S2NoiseScheduler):
    """Linear decay scheduler for backward compatibility."""
    def __init__(self, horizon_nodes: int, action_dim: int, initial_scale: float = 1.0, final_scale: float = 0.1, device: Optional[torch.device] = None):
        super().__init__(horizon_nodes, action_dim,
                        shape_fn=lambda x: torch.ones_like(x),
                        decay_fn=linear_decay,
                        base_scale=initial_scale, device=device,
                        final_ratio=final_scale/initial_scale)

class ExponentialDecayScheduler(S2NoiseScheduler):
    """Exponential decay scheduler for backward compatibility."""
    def __init__(self, horizon_nodes: int, action_dim: int, initial_scale: float = 3.0, decay_rate: float = 0.6, min_scale: float = 0.005, device: Optional[torch.device] = None):
        super().__init__(horizon_nodes, action_dim,
                        shape_fn=lambda x: torch.ones_like(x),
                        decay_fn=exponential_decay,
                        base_scale=initial_scale, device=device,
                        decay_rate=decay_rate)




def create_noise_scheduler(schedule_type: Union[str, NoiseScheduleType],
                          horizon_nodes: int,
                          action_dim: int,
                          device: Optional[torch.device] = None,
                          **kwargs) -> NoiseSchedulerBase:
    """Factory function to create noise schedulers.

    Args:
        schedule_type: Type of noise scheduler
        horizon_nodes: Number of trajectory nodes
        action_dim: Dimension of the action space
        device: Device for tensor operations
        **kwargs: Additional arguments for specific schedulers

    Returns:
        Noise scheduler instance
    """
    if isinstance(schedule_type, str):
        schedule_type = NoiseScheduleType(schedule_type)

    if schedule_type == NoiseScheduleType.CONSTANT:
        return ConstantNoiseScheduler(horizon_nodes, action_dim, device=device, **kwargs)
    elif schedule_type == NoiseScheduleType.LINEAR_DECAY:
        return LinearDecayScheduler(horizon_nodes, action_dim, device=device, **kwargs)
    elif schedule_type == NoiseScheduleType.EXPONENTIAL_DECAY:
        return ExponentialDecayScheduler(horizon_nodes, action_dim, device=device, **kwargs)
    elif schedule_type == NoiseScheduleType.COSINE_DECAY:
        return S2NoiseScheduler(horizon_nodes, action_dim, 
                               shape_fn=lambda x: torch.ones_like(x),
                               decay_fn=cosine_decay, device=device, **kwargs)
    elif schedule_type == NoiseScheduleType.HIERARCHICAL:
        return HierarchicalNoiseScheduler(horizon_nodes, action_dim, device=device, **kwargs)
    elif schedule_type == NoiseScheduleType.ADAPTIVE:
        return AdaptiveNoiseScheduler(horizon_nodes, action_dim, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown noise schedule type: {schedule_type}")


# Predefined scheduler configurations for common use cases
def create_2d_navigation_scheduler(horizon_nodes: int,
                                   device: Optional[torch.device] = None) -> S3NoiseScheduler:
    """Create a noise scheduler optimized for 2D navigation tasks."""
    dimension_priorities = [1.5, 1.5]  # x, y position - higher priority
    
    return S3NoiseScheduler(
        horizon_nodes=horizon_nodes,
        action_dim=2,
        dim_scale=dimension_priorities,
        shape_fn=sine_shape,
        decay_fn=exponential_decay,
        device=device
    )


def create_inverted_pendulum_scheduler(horizon_nodes: int,
                                      device: Optional[torch.device] = None) -> S3NoiseScheduler:
    """Create a noise scheduler optimized for inverted pendulum control."""
    dimension_priorities = [1.8, 1.0]  # Force higher priority than angle
    
    # Custom temporal shape - focus on earlier time steps
    def temporal_shape(x):
        return 1.5 - 0.5 * x  # Decreases from 1.5 to 1.0 over time
    
    return S3NoiseScheduler(
        horizon_nodes=horizon_nodes,
        action_dim=1,  # Single force input
        dim_scale=1.0,
        shape_fn=temporal_shape,
        decay_fn=exponential_decay,
        device=device
    )


# Convenience functions for creating S2/S3 schedulers with common configurations
def create_s2_scheduler(horizon_nodes: int,
                       action_dim: int,
                       shape: str = "sine",
                       decay: str = "exponential",
                       base_scale: float = 1.0,
                       device: Optional[torch.device] = None,
                       **decay_kwargs) -> S2NoiseScheduler:
    """Create S2 noise scheduler with specified shape and decay functions."""
    
    # Shape function mapping
    shape_fns = {
        "sine": sine_shape,
        "linear": linear_shape, 
        "quadratic": quadratic_shape,
        "exponential": exponential_shape,
        "constant": lambda x: torch.ones_like(x)
    }
    
    # Decay function mapping
    decay_fns = {
        "constant": constant_decay,
        "linear": linear_decay,
        "exponential": exponential_decay,
        "cosine": cosine_decay
    }
    
    return S2NoiseScheduler(
        horizon_nodes=horizon_nodes,
        action_dim=action_dim,
        shape_fn=shape_fns[shape],
        decay_fn=decay_fns[decay],
        base_scale=base_scale,
        device=device,
        **decay_kwargs
    )


def create_s3_scheduler(horizon_nodes: int,
                       action_dim: int,
                       dim_scale: Optional[Union[float, List[float]]] = None,
                       shape: str = "sine",
                       decay: str = "exponential", 
                       base_scale: float = 1.0,
                       device: Optional[torch.device] = None,
                       **decay_kwargs) -> S3NoiseScheduler:
    """Create S3 noise scheduler with specified dimension scale, shape and decay functions."""
    
    # Shape function mapping
    shape_fns = {
        "sine": sine_shape,
        "linear": linear_shape,
        "quadratic": quadratic_shape, 
        "exponential": exponential_shape,
        "constant": lambda x: torch.ones_like(x)
    }
    
    # Decay function mapping
    decay_fns = {
        "constant": constant_decay,
        "linear": linear_decay,
        "exponential": exponential_decay,
        "cosine": cosine_decay
    }
    
    return S3NoiseScheduler(
        horizon_nodes=horizon_nodes,
        action_dim=action_dim,
        dim_scale=dim_scale,
        shape_fn=shape_fns[shape],
        decay_fn=decay_fns[decay],
        base_scale=base_scale,
        device=device,
        **decay_kwargs
    )


def create_hierarchical_scheduler(horizon_nodes: int,
                                action_dim: int,
                                activation_pattern: str = "staggered",
                                dim_scale: Optional[Union[float, List[float]]] = None,
                                decay: str = "exponential",
                                base_scale: float = 1.0,
                                device: Optional[torch.device] = None,
                                **decay_kwargs) -> HierarchicalNoiseScheduler:
    """Create hierarchical noise scheduler with predefined activation patterns.
    
    Args:
        horizon_nodes: Number of trajectory nodes
        action_dim: Dimension of the action space
        activation_pattern: Predefined activation pattern ("staggered", "early_late", "overlapping")
        dim_scale: Per-dimension scaling factors
        decay: Decay function type
        base_scale: Base scaling factor
        device: Device for tensor operations
        **decay_kwargs: Additional arguments for decay function
        
    Returns:
        Configured hierarchical noise scheduler
    """
    # Decay function mapping
    decay_fns = {
        "constant": constant_decay,
        "linear": linear_decay,
        "exponential": exponential_decay,
        "cosine": cosine_decay
    }
    
    # Get the actual decay function
    if decay not in decay_fns:
        raise ValueError(f"Unknown decay function: {decay}")
    decay_fn = decay_fns[decay]
    
    # Predefined activation patterns
    if activation_pattern == "staggered":
        # Evenly stagger activation starts across dimensions
        activate_start = torch.linspace(0.0, 0.5, action_dim)
        activate_len = torch.full((action_dim,), 0.3)
    elif activation_pattern == "early_late":
        # First half of dims activate early, second half activate late
        half_dim = action_dim // 2
        activate_start = torch.cat([
            torch.zeros(half_dim),  # Early activation
            torch.full((action_dim - half_dim,), 0.6)  # Late activation
        ])
        activate_len = torch.full((action_dim,), 0.4)
    elif activation_pattern == "overlapping":
        # Create overlapping activation windows
        activate_start = torch.linspace(0.0, 0.4, action_dim)
        activate_len = torch.full((action_dim,), 0.5)  # Longer overlap
    else:
        raise ValueError(f"Unknown activation pattern: {activation_pattern}")
    
    return HierarchicalNoiseScheduler(
        horizon_nodes=horizon_nodes,
        action_dim=action_dim,
        dim_scale=dim_scale,
        activate_start=activate_start,
        activate_len=activate_len,
        decay_fn=decay_fn,  # Pass the actual function, not the string
        base_scale=base_scale,
        device=device,
        **decay_kwargs
    )

