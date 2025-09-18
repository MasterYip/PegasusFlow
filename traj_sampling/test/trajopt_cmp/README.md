# Trajectory Optimization Comparison Framework

This directory contains a comprehensive framework for comparing different trajectory optimization methods including MPPI, WBFO (Weighted Basis Function Optimization), and AVWBFO (Action-Value WBFO) across multiple environments and noise scheduling strategies.

## Overview

The framework evaluates trajectory optimization methods on:

1. **Optimization Efficiency**: Comparing AVWBFO/WBFO with MPPI in terms of:
   - Number of samples required
   - Trajectory quality (final cost)
   - Computational time
   - Convergence rate

2. **Hierarchical Noise Scheduling**: Evaluating the impact of different noise scheduling strategies:
   - Constant noise
   - Linear/exponential decay
   - Hierarchical scheduling (dimension-specific priorities)
   - Adaptive scheduling

## Files

### Core Implementation

- **`noise_scheduler.py`**: Noise scheduling strategies with hierarchical support
  - `NoiseSchedulerBase`: Abstract base class for all schedulers
  - `ConstantNoiseScheduler`: Fixed noise across all dimensions and time
  - `LinearDecayScheduler`: Linear noise reduction over iterations
  - `ExponentialDecayScheduler`: Exponential noise decay
  - `CosineDecayScheduler`: Cosine annealing schedule
  - `HierarchicalNoiseScheduler`: Dimension and time-specific priorities
  - `AdaptiveNoiseScheduler`: Performance-based noise adaptation

- **`trajopt_env.py`**: Test environments for trajectory optimization
  - `Navigation2DEnv`: 2D navigation with random obstacles
  - `InvertedPendulumEnv`: Inverted pendulum optimal control
  - `MultiTaskEnv`: Wrapper for switching between tasks

- **`trajopt_cmp.py`**: Main comparison framework
  - `TrajectoryOptimizationComparison`: Core comparison class
  - Academic-quality plotting and statistical analysis
  - Comprehensive metrics and performance evaluation

### Utilities

- **`run_comparison.py`**: Simple test scripts for quick evaluation
- **`README.md`**: This documentation file

## Usage

### Quick Test

Run a quick test with minimal parameters:

```bash
cd traj_sampling/test/trajopt_cmp/
python run_comparison.py --test quick
```

### Full Comparison

Run complete comparison for specific environments:

```bash
# Navigation environment
python run_comparison.py --test navigation

# Inverted pendulum environment  
python run_comparison.py --test pendulum

# Both environments
python run_comparison.py --test all
```

### Custom Configuration

Use the main comparison script with custom parameters:

```bash
python trajopt_cmp.py --env navigation2d \
                      --num_samples 100 \
                      --num_iterations 50 \
                      --num_trials 5 \
                      --noise_schedules constant exponential_decay hierarchical \
                      --results_dir my_results
```

### Command Line Arguments

```bash
python trajopt_cmp.py --help
```

Key arguments:
- `--env`: Environment (`navigation2d` or `inverted_pendulum`)
- `--num_samples`: Samples per optimization iteration
- `--num_iterations`: Optimization iterations per trial
- `--num_trials`: Number of statistical trials
- `--noise_schedules`: Noise scheduling strategies to compare
- `--results_dir`: Output directory for results and plots

## Environments

### 2D Navigation Environment

- **Goal**: Navigate from start to goal while avoiding circular obstacles
- **State**: `[x, y, vx, vy]` (position and velocity)
- **Action**: `[vx, vy]` (velocity commands)
- **Rewards**: 
  - Distance to goal (negative)
  - Goal reaching bonus
  - Collision penalty
  - Smoothness reward
  - Efficiency reward

### Inverted Pendulum Environment

- **Goal**: Balance inverted pendulum by controlling cart position
- **State**: `[cart_pos, cart_vel, pole_angle, pole_angular_vel]`
- **Action**: `[force]` (horizontal force on cart)
- **Rewards**:
  - Angle tracking error (want pole upright)
  - Position tracking error
  - Velocity penalties
  - Control effort penalty
  - Upright stability bonus

## Noise Scheduling Strategies

### Constant Noise
- Fixed noise magnitude throughout optimization
- Baseline for comparison

### Exponential Decay
- Noise reduces exponentially with iterations
- `noise(t) = initial_noise * decay_rate^t`

### Hierarchical Scheduling
- Different noise priorities for different dimensions
- Task-specific prioritization:
  - **Navigation**: Higher noise for position vs. velocity
  - **Pendulum**: Higher noise for base control vs. pendulum angle

### Adaptive Scheduling
- Noise adapts based on optimization progress
- Increases noise when stuck, decreases when improving

## Output and Results

The framework generates:

### Data Files
- `individual_results.json`: Raw results from all trials
- `aggregate_results.json`: Statistical summaries
- `config.json`: Experiment configuration

### Academic-Quality Plots
- **Performance Comparison**: Final costs and computation times
- **Convergence Analysis**: Convergence curves and sample efficiency
- **Computational Efficiency**: Time analysis and Pareto frontiers
- **Noise Scheduling Impact**: Strategy-specific performance analysis
- **Trajectory Examples**: Best trajectory visualizations

### Metrics
- Final trajectory cost (higher is better)
- Total computation time
- Convergence iteration
- Sample efficiency
- Success rate
- Statistical significance testing

## Implementation Details

### Optimizers Integration
The framework integrates with the existing optimizer implementations:
- Uses `create_wbfo_optimizer()`, `create_avwbfo_optimizer()`, `create_mppi_optimizer()`
- Maintains compatibility with existing spline and trajectory handling
- Supports batch processing for efficient evaluation

### Noise Generation
- Generates noise based on scheduler output
- Supports full `[horizon_nodes, action_dim]` noise matrices
- Handles both scalar and tensor noise scales
- Respects action space constraints

### Statistical Analysis
- Multiple trials for statistical significance
- Mean, standard deviation, and median calculations
- Confidence intervals and error bars
- Non-parametric statistical tests when appropriate

## Extension Points

### Adding New Optimizers
1. Implement optimizer following `OptimizerBase` interface
2. Add creation function to `trajopt_cmp.py`
3. Include in optimizer dictionary

### Adding New Environments
1. Inherit from `TrajOptEnvBase`
2. Implement required methods: `reset()`, `step()`, `batch_rollout()`
3. Add visualization method
4. Include in environment factory

### Adding New Noise Schedulers
1. Inherit from `NoiseSchedulerBase`
2. Implement `get_noise_scale()` method
3. Add to scheduler factory function
4. Include in predefined configurations

## Research Applications

This framework is designed for:
- Benchmarking trajectory optimization algorithms
- Studying hierarchical noise scheduling impact
- Comparing sample efficiency across methods
- Analyzing computational vs. performance trade-offs
- Generating publication-quality results and figures

The modular design allows easy extension for new algorithms, environments, and analysis methods while maintaining reproducibility and statistical rigor.