# Trajectory Sampling Package

A standalone trajectory optimization module that provides trajectory gradient sampling, weighted basis function optimization, and spline interpolation utilities for robotics and control applications.

## Features

- **Trajectory Gradient Sampling**: Advanced sampling-based trajectory optimization
- **Weighted Basis Function Optimization (WBFO)**: Efficient trajectory optimization using basis functions
- **Spline Interpolation**: Multiple spline implementations including:
  - Uniform cubic B-splines (`UniBSpline`)
  - SciPy-based splines (`SciPySpline`) 
  - JAX-accelerated splines (`JAXSplineInterpolator`)
- **Batch Processing**: Efficient batch operations for trajectory optimization
- **Flexible Interface**: Compatible with various RL frameworks and control systems

## Installation

### Basic Installation

```bash
# Install from source
cd traj_sampling
pip install -e .
```

### With JAX Support (Optional)

```bash
pip install -e .[jax]
```

### Development Installation

```bash
pip install -e .[dev]
```

### Complete Installation

```bash
pip install -e .[all]
```

## Quick Start

```python
from traj_sampling import TrajGradSampling, TrajGradSamplingCfg
from traj_sampling.wbfo import create_wbfo_optimizer
from traj_sampling.spline import UniBSpline

# Configure trajectory optimization
cfg = TrajGradSamplingCfg()
cfg.trajectory_opt.horizon_samples = 100
cfg.trajectory_opt.horizon_nodes = 20
cfg.trajectory_opt.num_samples = 64
cfg.trajectory_opt.update_method = "wbfo"
cfg.trajectory_opt.interp_method = "spline"

# Initialize trajectory sampler
traj_sampler = TrajGradSampling(
    cfg=cfg,
    device=torch.device('cuda'),
    num_envs=1024,
    num_actions=12,
    dt=0.02,
    main_env_indices=[0]
)

# Use spline interpolation
spline = UniBSpline(
    horizon_nodes=20,
    horizon_samples=100,
    dt=0.02
)

# Convert between node and dense representations
dense_trajectory = spline.node2dense(control_nodes)
recovered_nodes = spline.dense2node(dense_trajectory)
```

## Spline Implementations

### UniBSpline
- Uniform cubic B-spline implementation
- Uses pseudo-inverse for optimal least squares fitting
- Efficient batch processing with precomputed basis matrices

### SciPySpline  
- High-quality spline interpolation using SciPy
- Configurable spline degree (linear, quadratic, cubic)
- Robust numerical algorithms

### JAXSplineInterpolator
- JAX-accelerated spline interpolation
- Compatible with JAX JIT compilation
- Vectorized operations for maximum performance

## Usage in Legged Gym

This package is designed to be used with legged_gym and other robotics frameworks:

```python
# In your legged_gym environment
from traj_sampling import TrajGradSampling, TrajGradSamplingCfg

class YourRobotEnv(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize trajectory optimization
        self.traj_generator = TrajGradSampling(
            cfg=cfg.traj_grad_sampling,
            device=self.device,
            num_envs=self.num_envs,
            num_actions=self.num_actions,
            dt=self.dt,
            main_env_indices=list(range(self.num_envs))
        )
```

## Profiling

```python
# Method 1: Enable all profiling
from traj_sampling.utils.benchmark import enable_profiling
enable_profiling()

# Method 2: Enable specific modes
from traj_sampling.utils.benchmark import set_profiling_mode
set_profiling_mode(time_prof=True, gpu_prof=True)

# Method 3: Environment variables
export TIME_PROFILING=1
export GPU_PROFILING=1
```

## API Reference

### Core Classes

- `TrajGradSampling`: Main trajectory optimization class
- `TrajGradSamplingCfg`: Configuration for trajectory optimization
- `WeightedBasisFunctionOptimizer`: WBFO implementation
- `ActionValueWBFO`: Action-value WBFO with discount factor

### Spline Classes

- `SplineBase`: Abstract base class for all spline implementations
- `UniBSpline`: Uniform cubic B-spline implementation
- `SciPySpline`: SciPy-based spline interpolation
- `JAXSplineInterpolator`: JAX-accelerated spline interpolation

## Requirements

### Core Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.3.0

### Optional Requirements
- JAX ≥ 0.3.0 (for JAX spline support)
- jax-cosmo ≥ 0.1.0 (for JAX spline interpolation)

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{traj_sampling,
  title={Trajectory Sampling Package},
  author={MasterYip @ HIT},
  year={2025},
  url={https://github.com/MasterYip/PredictiveDiffusionPlanner}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.