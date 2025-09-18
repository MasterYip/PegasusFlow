# Robot Navigation Trajectory Optimization Experiment

This experiment compares different trajectory optimization configurations for robot navigation tasks across multiple scenarios.

## Overview

The experiment performs a comprehensive comparison at two levels:

- **Level 1**: Different trajectory optimization methods/configurations
- **Level 2**: Different navigation tasks/scenes

## Features

1. **Multiple Trajectory Optimization Methods**: Compares AVWBFO, MPPI, and various noise scheduling strategies
2. **Multiple Navigation Tasks**: Tests different start-goal configurations and difficulty levels
3. **Comprehensive Metrics**: Tracks navigation success rate, completion steps, rewards, and optimization time
4. **Academic-Quality Outputs**: Generates publication-ready plots and analysis tables
5. **Multiprocessing Support**: Runs experiments in parallel to avoid Isaac Gym conflicts

## Supported Robots

Currently supports:

- **ElSpider Air** (6-legged robot) - Full support with navigation environment
- **Anymal C** (4-legged robot) - Basic support
- **Other robots** - Default configurations

## Navigation Tasks

### ElSpider Air Tasks

- `short_forward`: Simple forward navigation (2m)
- `medium_diagonal`: Diagonal navigation (3.6m)
- `long_straight`: Long straight navigation (5m)
- `obstacle_avoidance`: Navigation with obstacle avoidance (4.3m)
- `tight_turn`: Navigation requiring tight turns (3.6m)

### Task Parameters

Each task defines:

- Start position and orientation
- Goal position
- Maximum allowed steps
- Task-specific requirements

## Trajectory Optimization Methods

### Available Methods

1. **AVWBFO_MC_Linear**: AVWBFO with Monte Carlo sampling
2. **AVWBFO_LHS_Linear**: AVWBFO with Latin Hypercube sampling
3. **MPPI_MC_Linear**: MPPI with Monte Carlo sampling
4. **MPPI_LHS_Linear**: MPPI with Latin Hypercube sampling
5. **AVWBFO_Hierarchical**: AVWBFO with hierarchical noise scheduling
6. **AVWBFO_ExpDecay**: AVWBFO with exponential noise decay

### Configuration Parameters

- Update method (AVWBFO, MPPI)
- Noise sampler type (Monte Carlo, LHS, Halton)
- Noise scheduler type (S2, S3, hierarchical, adaptive)
- Noise shape and decay functions
- Horizon parameters and diffusion steps

## Usage

### Basic Usage

```bash
# Run with default settings (ElSpider Air, all methods, all tasks)
python exp_robot_nav.py

# Run with specific robot
python exp_robot_nav.py --robot anymal_c

# Run with specific methods
python exp_robot_nav.py --methods AVWBFO_MC_Linear MPPI_MC_Linear

# Run with specific tasks
python exp_robot_nav.py --tasks short_forward medium_diagonal

# Run with multiple seeds for statistical significance
python exp_robot_nav.py --num_seeds 5
```

### Advanced Usage

```bash
# Custom environment settings
python exp_robot_nav.py \
    --num_envs 8 \
    --rollout_envs 128 \
    --headless \
    --device cuda:1

# Custom output directory
python exp_robot_nav.py --output_dir ./my_experiment_results

# Debug visualization
python exp_robot_nav.py --debug_viz --debug_viz_origins
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--robot` | `elspider_air` | Robot type to use |
| `--num_seeds` | `1` | Number of random seeds per method-task |
| `--num_envs` | `4` | Number of main environments |
| `--rollout_envs` | `64` | Rollout environments per main |
| `--methods` | `None` (all) | Specific methods to test |
| `--tasks` | `None` (all) | Specific tasks to test |
| `--headless` | `False` | Run in headless mode |
| `--device` | `cuda:0` | Computation device |
| `--debug_viz` | `True` | Enable debug visualization |
| `--debug_viz_origins` | `False` | Show environment origins |
| `--output_dir` | `None` (auto) | Output directory |

## Output Structure

The experiment generates a timestamped output directory containing:

```
robot_nav_experiment_elspider_air_20241201_143022/
├── experiment_config.json          # Experiment configuration
├── experiment_analysis.json        # Statistical analysis results
├── experiment_raw_data.pkl         # Raw experiment data
├── experiment_summary.txt          # Human-readable summary
├── navigation_success_heatmap.png  # Success rate heatmap
├── completion_steps_comparison.png # Completion efficiency comparison
└── summary_table.png               # Results summary table
```

## Metrics Tracked

### Navigation Metrics

- **Success Rate**: Percentage of environments that reached the goal
- **Completion Steps**: Number of steps required to reach the goal
- **Final Distance**: Distance to goal at the end of episode

### Performance Metrics

- **Average Reward**: Mean reward across all steps
- **Optimization Time**: Time spent on trajectory optimization
- **Total Steps**: Total simulation steps

### Statistical Analysis

- Mean, standard deviation, min, max, median for all metrics
- Cross-method and cross-task comparisons
- Statistical significance testing

## Analysis and Visualization

### Generated Plots

1. **Navigation Success Heatmap**: Shows success rates for all method-task combinations
2. **Completion Steps Comparison**: Bar chart comparing efficiency across methods
3. **Summary Table**: Comprehensive results table with all metrics

### Data Analysis

- Per-method and per-task statistics
- Cross-comparison analysis
- Performance ranking and recommendations

## Extending the Experiment

### Adding New Robots

1. Implement navigation environment class (extend `RobotBatchRolloutNav`)
2. Add robot-specific configuration in `create_trajectory_optimization_configs()`
3. Define navigation tasks in `get_navigation_tasks()`

### Adding New Methods

1. Create new configuration class or modify existing one
2. Add method to `create_trajectory_optimization_configs()`
3. Update color palette in `create_academic_plots()`

### Adding New Tasks

1. Define task parameters in `get_navigation_tasks()`
2. Ensure environment supports the task configuration
3. Update task-specific analysis if needed

## Requirements

- Python 3.8+
- Isaac Gym
- PyTorch
- NumPy, Matplotlib, Seaborn, Pandas
- Legged Gym environment
- Trajectory sampling framework

## Notes

- Uses multiprocessing to avoid Isaac Gym conflicts
- Each experiment runs in a separate process
- Results are automatically saved and analyzed
- Supports both headless and visualization modes
- Designed for academic research and publication

## Troubleshooting

### Common Issues

1. **Isaac Gym conflicts**: Ensure multiprocessing is working correctly
2. **Memory issues**: Reduce `num_envs` or `rollout_envs`
3. **CUDA errors**: Check device availability and memory
4. **Import errors**: Verify all dependencies are installed

### Performance Tips

1. Use headless mode for faster execution
2. Adjust environment counts based on hardware
3. Use multiple seeds for statistical significance
4. Monitor GPU memory usage during execution
