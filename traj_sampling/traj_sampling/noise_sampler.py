"""
Noise sampling module for trajectory optimization.

This module provides various noise sampling methods including Monte Carlo (MC) 
and Latin Hypercube Sampling (LHS) for generating well-distributed noise samples
in trajectory optimization.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Callable
from enum import Enum


class NoiseDistribution(Enum):
    """Supported noise distributions."""
    NORMAL = "normal"
    UNIFORM = "uniform"


class BaseSampler(ABC):
    """Base class for noise sampling methods."""
    
    def __init__(self,
                 distribution: NoiseDistribution = NoiseDistribution.NORMAL,
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None):
        """Initialize base sampler.
        
        Args:
            distribution: Type of distribution to sample from
            device: Device for tensor operations
            seed: Random seed for reproducibility
        """
        self.distribution = distribution
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        
        # Set up random number generator
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    @abstractmethod
    def sample(self, 
               shape: Union[Tuple[int, ...], torch.Size],
               **kwargs) -> torch.Tensor:
        """Generate noise samples.
        
        Args:
            shape: Shape of the samples to generate
            **kwargs: Additional distribution parameters
            
        Returns:
            Tensor of noise samples
        """
        pass
    
    def _apply_distribution(self, 
                           uniform_samples: torch.Tensor,
                           **kwargs) -> torch.Tensor:
        """Apply the specified distribution to uniform samples.
        
        Args:
            uniform_samples: Uniform samples in [0, 1]
            **kwargs: Distribution parameters
            
        Returns:
            Samples from the specified distribution
        """
        if self.distribution == NoiseDistribution.NORMAL:
            # Convert uniform to normal using inverse transform sampling
            mean = kwargs.get('mean', 0.0)
            std = kwargs.get('std', 1.0)
            
            # Use Box-Muller transform for better numerical stability
            # than direct inverse normal CDF
            if uniform_samples.numel() % 2 == 1:
                # Add one element if odd number of samples
                uniform_samples = torch.cat([uniform_samples, torch.rand(1, device=self.device)])
            
            u1 = uniform_samples[::2]
            u2 = uniform_samples[1::2]
            
            # Box-Muller transform
            r = torch.sqrt(-2 * torch.log(u1))
            theta = 2 * np.pi * u2
            
            z1 = r * torch.cos(theta)
            z2 = r * torch.sin(theta)
            
            # Combine and reshape
            normal_samples = torch.stack([z1, z2], dim=-1).view(-1)
            
            # Trim to original size if we added an element
            if normal_samples.size(0) > uniform_samples.size(0):
                normal_samples = normal_samples[:uniform_samples.size(0)]
            
            # Apply mean and std
            return normal_samples * std + mean
            
        elif self.distribution == NoiseDistribution.UNIFORM:
            # Convert uniform [0,1] to uniform [a,b]
            low = kwargs.get('low', -1.0)
            high = kwargs.get('high', 1.0)
            return uniform_samples * (high - low) + low
        
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")


class MonteCarloSampler(BaseSampler):
    """Monte Carlo noise sampler."""
    
    def __init__(self,
                 distribution: NoiseDistribution = NoiseDistribution.NORMAL,
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None):
        """Initialize Monte Carlo sampler.
        
        Args:
            distribution: Type of distribution to sample from
            device: Device for tensor operations
            seed: Random seed for reproducibility
        """
        super().__init__(distribution, device, seed)
    
    def sample(self, 
               shape: Union[Tuple[int, ...], torch.Size],
               **kwargs) -> torch.Tensor:
        """Generate noise samples using Monte Carlo sampling.
        
        Args:
            shape: Shape of the samples to generate
            **kwargs: Distribution parameters
            
        Returns:
            Tensor of noise samples
        """
        # Apply distribution
        if self.distribution == NoiseDistribution.NORMAL:
            mean = kwargs.get('mean', 0.0)
            std = kwargs.get('std', 1.0)
            noise_samples = torch.normal(mean=mean, std=std, size=shape, device=self.device)
        elif self.distribution == NoiseDistribution.UNIFORM:
            low = kwargs.get('low', -1.0)
            high = kwargs.get('high', 1.0)
            noise_samples = torch.rand(shape, device=self.device) * (high - low) + low

        return noise_samples.view(shape)


class LatinHypercubeSampler(BaseSampler):
    """Latin Hypercube noise sampler."""
    
    def __init__(self,
                 distribution: NoiseDistribution = NoiseDistribution.NORMAL,
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None):
        """Initialize Latin Hypercube sampler.
        
        Args:
            distribution: Type of distribution to sample from
            device: Device for tensor operations
            seed: Random seed for reproducibility
        """
        super().__init__(distribution, device, seed)
    
    def sample(self, 
               shape: Union[Tuple[int, ...], torch.Size],
               **kwargs) -> torch.Tensor:
        """Generate noise samples using Latin Hypercube sampling.
        
        Args:
            shape: Shape of the samples to generate
            **kwargs: Distribution parameters
            
        Returns:
            Tensor of noise samples
        """
        # For LHS, we need to handle the sampling differently
        # We'll treat the first dimension as the number of samples
        # and the remaining dimensions as the feature space
        
        if len(shape) == 1:
            n_samples = shape[0]
            n_dims = 1
            output_shape = shape
        else:
            n_samples = shape[0]
            n_dims = int(np.prod(shape[1:]))
            output_shape = shape
        
        # Generate LHS samples for each dimension
        uniform_samples = self._generate_lhs_samples(n_samples, n_dims)
        
        # Apply distribution
        noise_samples = self._apply_distribution(uniform_samples, **kwargs)
        return noise_samples.view(output_shape)
    
    def _generate_lhs_samples(self, n_samples: int, n_dims: int) -> torch.Tensor:
        """Generate Latin Hypercube samples in uniform [0,1].
        
        Args:
            n_samples: Number of samples
            n_dims: Number of dimensions
            
        Returns:
            Uniform LHS samples
        """
        # Use numpy for LHS generation, then convert to torch
        lhs_samples = np.zeros((n_samples, n_dims))
        
        for dim in range(n_dims):
            # Create evenly spaced bins
            bins = np.linspace(0, 1, n_samples + 1)
            
            # Generate uniform random samples within each bin
            u = np.random.uniform(0, 1, n_samples)
            samples = bins[:-1] + u * (bins[1:] - bins[:-1])
            
            # Randomly permute the samples
            np.random.shuffle(samples)
            
            lhs_samples[:, dim] = samples
        
        # Convert to torch tensor and flatten
        return torch.tensor(lhs_samples, device=self.device, dtype=torch.float32).view(-1)


class HaltonSampler(BaseSampler):
    """Halton sequence noise sampler for low-discrepancy sampling."""
    
    def __init__(self,
                 distribution: NoiseDistribution = NoiseDistribution.NORMAL,
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None):
        """Initialize Halton sampler.
        
        Args:
            distribution: Type of distribution to sample from
            device: Device for tensor operations
            seed: Random seed for reproducibility (affects starting index)
        """
        super().__init__(distribution, device, seed)
        self.start_index = seed if seed is not None else 0
    
    def sample(self, 
               shape: Union[Tuple[int, ...], torch.Size],
               **kwargs) -> torch.Tensor:
        """Generate noise samples using Halton sequence.
        
        Args:
            shape: Shape of the samples to generate
            **kwargs: Distribution parameters
            
        Returns:
            Tensor of noise samples
        """
        if len(shape) == 1:
            n_samples = shape[0]
            n_dims = 1
        else:
            n_samples = shape[0]
            n_dims = int(np.prod(shape[1:]))
        
        # Generate Halton samples
        uniform_samples = self._generate_halton_samples(n_samples, n_dims)
        
        # Apply distribution
        noise_samples = self._apply_distribution(uniform_samples, **kwargs)
        
        return noise_samples.view(shape)
    
    def _generate_halton_samples(self, n_samples: int, n_dims: int) -> torch.Tensor:
        """Generate Halton sequence samples.
        
        Args:
            n_samples: Number of samples
            n_dims: Number of dimensions
            
        Returns:
            Uniform Halton samples
        """
        # Use first n_dims prime numbers as bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        if n_dims > len(primes):
            raise ValueError(f"Too many dimensions ({n_dims}). Maximum supported: {len(primes)}")
        
        halton_samples = np.zeros((n_samples, n_dims))
        
        for dim in range(n_dims):
            base = primes[dim]
            for i in range(n_samples):
                halton_samples[i, dim] = self._halton_value(i + self.start_index, base)
        
        return torch.tensor(halton_samples, device=self.device, dtype=torch.float32).view(-1)
    
    def _halton_value(self, index: int, base: int) -> float:
        """Compute Halton value for given index and base."""
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f = f / base
        return result


class NoiseSamplerFactory:
    """Factory for creating noise samplers."""
    
    @staticmethod
    def create_sampler(sampler_type: str,
                      distribution: Union[str, NoiseDistribution] = NoiseDistribution.NORMAL,
                      device: Optional[torch.device] = None,
                      seed: Optional[int] = None) -> BaseSampler:
        """Create a noise sampler.
        
        Args:
            sampler_type: Type of sampler ('mc', 'lhs', 'halton')
            distribution: Distribution type
            device: Device for tensor operations
            seed: Random seed
            
        Returns:
            Noise sampler instance
        """
        if isinstance(distribution, str):
            distribution = NoiseDistribution(distribution)
        
        if sampler_type.lower() in ['mc', 'monte_carlo']:
            return MonteCarloSampler(distribution, device, seed)
        elif sampler_type.lower() in ['lhs', 'latin_hypercube']:
            return LatinHypercubeSampler(distribution, device, seed)
        elif sampler_type.lower() in ['halton']:
            return HaltonSampler(distribution, device, seed)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")


# Convenience functions
def sample_noise(shape: Union[Tuple[int, ...], torch.Size],
                sampler_type: str = 'mc',
                distribution: Union[str, NoiseDistribution] = NoiseDistribution.NORMAL,
                device: Optional[torch.device] = None,
                seed: Optional[int] = None,
                **kwargs) -> torch.Tensor:
    """Convenience function to sample noise.
    
    Args:
        shape: Shape of noise samples
        sampler_type: Type of sampler ('mc', 'lhs', 'halton')
        distribution: Distribution type
        device: Device for tensor operations
        seed: Random seed
        **kwargs: Distribution parameters
        
    Returns:
        Noise samples
    """
    sampler = NoiseSamplerFactory.create_sampler(sampler_type, distribution, device, seed)
    return sampler.sample(shape, **kwargs)


def sample_normal_noise(shape: Union[Tuple[int, ...], torch.Size],
                       sampler_type: str = 'mc',
                       mean: float = 0.0,
                       std: float = 1.0,
                       device: Optional[torch.device] = None,
                       seed: Optional[int] = None) -> torch.Tensor:
    """Sample normal noise.
    
    Args:
        shape: Shape of noise samples
        sampler_type: Type of sampler ('mc', 'lhs', 'halton')
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
        device: Device for tensor operations
        seed: Random seed
        
    Returns:
        Normal noise samples
    """
    return sample_noise(shape, sampler_type, NoiseDistribution.NORMAL, 
                       device, seed, mean=mean, std=std)


def sample_uniform_noise(shape: Union[Tuple[int, ...], torch.Size],
                        sampler_type: str = 'mc',
                        low: float = -1.0,
                        high: float = 1.0,
                        device: Optional[torch.device] = None,
                        seed: Optional[int] = None) -> torch.Tensor:
    """Sample uniform noise.
    
    Args:
        shape: Shape of noise samples
        sampler_type: Type of sampler ('mc', 'lhs', 'halton')
        low: Lower bound of uniform distribution
        high: Upper bound of uniform distribution
        device: Device for tensor operations
        seed: Random seed
        
    Returns:
        Uniform noise samples
    """
    return sample_noise(shape, sampler_type, NoiseDistribution.UNIFORM, 
                       device, seed, low=low, high=high)
