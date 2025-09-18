"""
Trajectory Sampling Module

A standalone trajectory optimization module that provides:
- Trajectory gradient sampling and optimization
- Weighted Basis Function Optimization (WBFO)
- Spline interpolation utilities
- Compatible with Python 3.10 and multiple projects

This module was extracted from legged_gym to be usable across different projects
without dependencies on specific RL frameworks.
"""

from .traj_grad_sampling import TrajGradSampling
from.optimizer import (
    WeightedBasisFunctionOptimizer,
    ActionValueWBFO,
    create_wbfo_optimizer,
    create_avwbfo_optimizer
)
from .spline import SplineBase, UniBSpline, InterpolatedSpline, CatmullRomSpline

# Try to import JAX implementation if available
try:
    from .spline import InterpolatedSplineJAX, JAX_AVAILABLE
    if JAX_AVAILABLE:
        __all_splines__ = [
            "SplineBase",
            "UniBSpline", 
            "InterpolatedSpline",
            "CatmullRomSpline",
            "InterpolatedSplineJAX"
        ]
    else:
        __all_splines__ = [
            "SplineBase",
            "UniBSpline",
            "InterpolatedSpline",
            "CatmullRomSpline"
        ]
except ImportError:
    __all_splines__ = [
        "SplineBase",
        "UniBSpline",
        "InterpolatedSpline",
        "CatmullRomSpline"
    ]

__version__ = "1.0.0"
__author__ = "MasterYip @ HIT"

__all__ = [
    "TrajGradSampling",
    "TrajGradSamplingCfg",
    "WeightedBasisFunctionOptimizer",
    "ActionValueWBFO",
    "create_wbfo_optimizer",
    "create_avwbfo_optimizer",
] + __all_splines__
