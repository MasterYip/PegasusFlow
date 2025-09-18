# Trajectory Optimization Experiment Documentation

## Paper Context

**Core Problem**: Efficient trajectory score gradient sampling for legged robot diffusion-based planning in complex environments with parallel computation.

**Main Challenge**: Current diffusion models for robotics require extensive expert demonstrations and suffer from computational inefficiency, while reinforcement learning struggles with long-horizon planning in confined spaces with obstacles.

**Key Innovation**: Direct score gradient estimation through rolling-denoising sampling, enabling diffusion policy training via score-matching rather than imitation learning, eliminating the dependency on expert demonstrations.

**Main Solution**: A hierarchical rolling-denoising framework that combines:
- **Weighted Basis Function Optimization (WBFO)** for efficient trajectory score sampling
- **Hierarchical noise scheduling** for adaptive computational resource allocation  
- **Parallel environment simulation** for real-time performance

## Experiment Overview

### Experiment 1: Optimization Algorithm Comparison

**Objective**: Validate WBFO/AVWBFO efficiency against established baselines in trajectory optimization and integrated system optimal control tasks.

**Methods Compared**:
- **AVWBFO** (Action-Value Weighted Basis Function Optimization) - integrated systems approach
- **WBFO** (Weighted Basis Function Optimization) - trajectory optimization approach  
- **MPPI** (Model Predictive Path Integral) - standard baseline equivalent to STOMP/CEM

**Test Scenarios**:
1. **Trajectory Optimization Tasks**: Point-to-point 2D navigation with obstacles
2. **Integrated System Optimal Control**: Inverted pendulum stabilization

**Performance Metrics**:
- Convergence rate (iterations to target cost)
- Sample efficiency (cost per number of samples)
- Computational time (ms per optimization step)
- Trajectory quality (smoothness, constraint satisfaction)
- Final accumulated cost

## Environment Setup Details

### 2D Navigation Environment (`Navigation2DEnv`)

**Task Description**: Navigate from start position to goal position while avoiding circular obstacles in a 2D workspace.

**State Representation**: 
- **Trajectory waypoints**: `[x, y]` position coordinates
- **Trajectory format**: Fixed start and end nodes with optimizable intermediate waypoints
- **Node structure**: `horizon_nodes + 1` total nodes (default: 8 nodes)
- **Dense trajectory**: `horizon_samples + 1` interpolated points (default: 64 samples)

**Environment Configuration**:
```python
EnvConfig:
    horizon_samples: 63  # Results in 64 total samples
    dt: 0.02            # Time step for interpolation
    max_episode_steps: 64
    device: "cuda" if available else "cpu"

Navigation2DEnv:
    workspace_size: (10.0, 10.0)     # 10x10 meter workspace
    num_obstacles: 25                 # Random circular obstacles
    obstacle_radius_range: (0.5, 1.5) # Obstacle size variation
    goal_radius: 0.5                 # Goal region tolerance
    collision_penalty: -5.0          # Penalty for obstacle collision
    goal_reward: 10.0                # Reward for reaching goal
    smoothness_weight: 2.0           # Trajectory smoothness importance
    efficiency_weight: 0.1           # Goal progress reward weight
```

**Reward Structure**:
1. **Distance to goal**: `-distance * 0.1` (encourages approach to goal)
2. **Goal reaching bonus**: Currently disabled to focus on approach behavior
3. **Collision penalty**: `-5.0 * collision_depth` (strong avoidance incentive)
4. **Smoothness reward**: `-segment_change * 2.0` (penalizes sharp direction changes)
5. **Efficiency reward**: `progress_toward_goal * 0.1` (rewards forward movement)

**Key Features**:
- **Fixed boundary conditions**: Start and end positions remain constant during optimization
- **Obstacle randomization**: Reproducible via `torch.manual_seed(4)`
- **Collision detection**: Continuous collision checking along trajectory
- **Batch evaluation**: Parallel trajectory assessment for multiple candidates

### Inverted Pendulum Environment (`InvertedPendulumEnv`)

**Task Description**: Balance an inverted pendulum by applying horizontal forces to the cart base.

**State Representation**: `[cart_pos, cart_vel, pole_angle, pole_angular_vel]`
- **Cart position**: Horizontal position of cart base
- **Cart velocity**: Horizontal velocity of cart
- **Pole angle**: Angle from vertical (0 = upright)
- **Pole angular velocity**: Rate of angle change

**Action Space**: `[force]` - Horizontal force applied to cart (bounded by ±max_force)

**Environment Configuration**:
```python
InvertedPendulumEnv:
    cart_mass: 1.0              # Mass of cart (kg)
    pole_mass: 0.1              # Mass of pole (kg)
    pole_length: 1.0            # Length of pole (m)
    max_force: 20.0             # Maximum applicable force (N)
    gravity: 9.81               # Gravitational acceleration (m/s²)
    target_angle: 0.0           # Target pole angle (upright)
    target_position: 0.0        # Target cart position
    angle_weight: 1.0           # Angle tracking importance
    position_weight: 0.5        # Position tracking importance
    velocity_weight: 0.1        # Velocity penalty weight
    control_weight: 0.01        # Control effort penalty
```

**Dynamics Model**:
- **Integration method**: 4th-order Runge-Kutta (RK4)
- **Physics**: Lagrangian mechanics for cart-pole system
- **Time step**: 0.02 seconds
- **State constraints**: Angle normalized to [-π, π]

**Reward Structure**:
1. **Angle tracking error**: `-angle_weight * (angle - target_angle)²`
2. **Position tracking error**: `-position_weight * (position - target_position)²`
3. **Velocity penalties**: `-velocity_weight * (cart_vel² + angular_vel²)`
4. **Control effort penalty**: `-control_weight * force²`

## Experimental Configuration

### Optimization Parameters

```python
ExperimentConfig:
    # Environment settings
    env_name: "navigation2d" | "inverted_pendulum"
    horizon_nodes: 7            # Optimizable nodes (8 total with boundaries)
    horizon_samples: 63         # Dense samples (64 total)
    dt: 0.02                    # Time discretization
    
    # Optimization settings
    num_samples: 100            # Candidate trajectories per iteration
    num_iterations: 50          # Optimization iterations per trial
    temp_sample: 0.1            # Sampling temperature
    gamma: 1.00                 # Discount factor for AVWBFO
    
    # Statistical analysis
    num_trials: 5               # Independent trials for significance
    
    # Computational settings
    device: "cuda" | "cpu"      # Hardware acceleration
```

### Noise Scheduling Strategies

**1. Constant Noise Scheduler**:
- Fixed noise magnitude throughout optimization
- Baseline for comparison with adaptive methods
- Uniform exploration across all trajectory dimensions

**2. Exponential Decay Scheduler**:
- Exponential reduction: `noise(t) = initial_noise * decay_rate^t`
- Balances exploration (early) with exploitation (late)
- Standard approach in trajectory optimization

**3. Hierarchical Noise Scheduler**:
- **Navigation task priorities**:
  - Higher noise for intermediate waypoints
  - Lower noise for boundary-adjacent nodes
  - Dimension-specific scaling for x/y coordinates
- **Pendulum task priorities**:
  - Higher noise for force magnitude
  - Temporal prioritization for control timing
- Task-specific adaptation based on physical constraints

**4. Adaptive Noise Scheduler**:
- Performance-based noise adjustment
- Increases exploration when improvement stagnates
- Decreases noise when consistent progress observed

### Optimizer Implementation Details

**AVWBFO (Action-Value Weighted Basis Function Optimization)**:
- Integrates trajectory optimization with system dynamics
- Action-value function approximation for long-horizon planning
- Basis function decomposition for computational efficiency
- Gamma-discounted reward accumulation

**WBFO (Weighted Basis Function Optimization)**:
- Direct trajectory optimization approach
- Weighted combination of basis functions
- Efficient gradient-free optimization
- Spline-based trajectory representation

**MPPI (Model Predictive Path Integral)**:
- Sampling-based optimization baseline
- Information-theoretic trajectory selection
- Temperature-controlled sample weighting
- Standard in robotics trajectory optimization

### Trajectory Representation

**Node-based Representation**:
- **Control nodes**: `horizon_nodes + 1` waypoints for optimization
- **Boundary conditions**: Fixed start/goal for navigation, free for pendulum
- **Interpolation**: Spline-based dense trajectory generation
- **Action constraints**: Enforced during sampling and optimization

**Dense Trajectory Conversion**:
- **Purpose**: Environment rollout and visualization
- **Method**: Spline interpolation between control nodes
- **Resolution**: `horizon_samples + 1` points for simulation
- **Temporal consistency**: Uniform time discretization

## Data Collection and Analysis

### Performance Metrics

**Primary Metrics**:
1. **Final Cost**: Accumulated reward over complete trajectory
2. **Convergence Rate**: Iterations required to reach stable performance
3. **Sample Efficiency**: Performance improvement per sample used
4. **Computational Time**: Wall-clock time per optimization iteration
5. **Trajectory Quality**: Smoothness and constraint satisfaction

**Statistical Analysis**:
- Multiple independent trials (default: 5) for significance testing
- Mean, standard deviation, and median calculations
- Confidence intervals and error bars
- Non-parametric tests for robust comparison

### Experimental Protocol

**Trial Structure**:
1. **Environment reset**: New obstacle configuration/initial conditions per trial
2. **Initial trajectory**: Consistent initialization across methods
3. **Optimization loop**: Fixed iteration count for fair comparison
4. **Performance tracking**: Real-time cost and timing measurement
5. **Result aggregation**: Statistical summary across trials

**Reproducibility**:
- Fixed random seeds for environment generation
- Deterministic initialization procedures
- Consistent evaluation metrics across methods
- Comprehensive configuration logging

### Visualization and Output

**Academic-Quality Plots**:
1. **Performance Comparison**: Final costs and computation times
2. **Convergence Analysis**: Learning curves and sample efficiency
3. **Computational Efficiency**: Time analysis and Pareto frontiers
4. **Noise Scheduling Impact**: Strategy-specific performance analysis
5. **Trajectory Examples**: Best trajectory visualizations
6. **Optimization Process**: Evolution of trajectories during optimization

**Data Outputs**:
- `individual_results.json`: Raw trial data with full histories
- `aggregate_results.json`: Statistical summaries and comparisons
- `config.json`: Complete experimental configuration
- High-resolution plots for publication use

## Expected Outcomes

### Hypothesis Testing

**H1**: AVWBFO demonstrates superior sample efficiency compared to MPPI baseline
**H2**: Hierarchical noise scheduling improves convergence rate over constant noise
**H3**: WBFO achieves competitive performance with reduced computational overhead
**H4**: Task-specific noise scheduling outperforms generic exponential decay

### Performance Expectations

**Navigation Task**:
- **AVWBFO**: Expected to excel in long-horizon planning with obstacles
- **WBFO**: Anticipated efficiency in waypoint optimization
- **MPPI**: Baseline performance with consistent but potentially slower convergence

**Pendulum Task**:
- **AVWBFO**: Superior handling of system dynamics and stability requirements
- **WBFO**: Efficient control sequence optimization
- **MPPI**: Standard performance for continuous control tasks

### Computational Efficiency

**Target Performance**:
- Real-time feasibility: <50ms per optimization iteration
- Sample efficiency: 50% improvement over MPPI baseline
- Convergence speed: 30% faster than traditional methods
- Scalability: Linear scaling with parallel computation

## Research Contributions

This experimental framework validates key contributions for diffusion-based trajectory planning:

1. **Algorithmic Innovation**: WBFO/AVWBFO efficiency in score gradient sampling
2. **Adaptive Resource Allocation**: Hierarchical noise scheduling effectiveness
3. **Computational Scalability**: Parallel implementation performance gains
4. **Practical Applicability**: Real-world robotics task performance

The comprehensive comparison provides empirical evidence for the proposed rolling-denoising framework's advantages in computational efficiency and sample complexity, directly supporting the paper's claims about eliminating expert demonstration dependencies while maintaining high-quality trajectory generation.
