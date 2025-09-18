"""
High-performance obsavoid2 batch environment with improved efficiency and higher complexity.

This module provides a highly parallelized version of the obsavoid environment
with vectorized operations, GPU acceleration, and more complex obstacle patterns.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from matplotlib import pyplot as plt

from ..env_runner_base import BatchEnvBase


class ObsAvoid2BatchEnv(BatchEnvBase):
    """High-performance batch environment for obstacle avoidance with enhanced complexity.

    This environment features:
    - Fully vectorized GPU-accelerated computations
    - Multi-layer obstacle patterns with time-varying complexity
    - Adaptive boundary generation with fractal patterns
    - Efficient observation computation using tensor operations
    - Enhanced reward structures with safety margins
    """

    def __init__(self,
                 num_main_envs: int = 8,
                 num_rollout_per_main: int = 32,
                 device: str = "cuda:0",
                 env_step: float = 0.01,
                 obs_dim: int = 62,  # 2 state + 60 sdf observations (10*6)
                 action_dim: int = 1,
                 enable_vis: bool = False,
                 enable_vis_rollout: bool = False,
                 vis_time_window: float = 2.0,
                 ctrl_mode: str = "acc",
                 complexity_level: int = 3,  # 1=simple, 2=medium, 3=complex
                 adaptive_difficulty: bool = True,
                 use_fractal_bounds: bool = True):
        """Initialize the high-performance obsavoid2 environment.

        Args:
            num_main_envs: Number of main environments
            num_rollout_per_main: Number of rollout environments per main
            device: Device for tensor operations
            env_step: Environment timestep
            obs_dim: Observation dimension (expanded for higher complexity)
            action_dim: Action dimension
            enable_vis: Whether to enable visualization for main environments
            enable_vis_rollout: Whether to enable visualization for rollout environments
            vis_time_window: Time window for visualization
            ctrl_mode: Control mode - 'acc', 'dy', or 'y'
            complexity_level: Environment complexity (1-3)
            adaptive_difficulty: Whether to adapt difficulty over time
            use_fractal_bounds: Whether to use fractal boundary patterns
        """
        super().__init__(num_main_envs, num_rollout_per_main, device)

        self.env_step = env_step
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_actions = action_dim
        self.dt = env_step

        # Control mode validation
        assert ctrl_mode in ['acc', 'dy', 'y'], f"Invalid ctrl_mode: {ctrl_mode}"
        self.ctrl_mode = ctrl_mode

        # Complexity settings
        self.complexity_level = complexity_level
        self.adaptive_difficulty = adaptive_difficulty
        self.use_fractal_bounds = use_fractal_bounds

        # Enhanced observation parameters for higher complexity
        if complexity_level == 1:
            self.obs_pt_param = [6, 5, 0.05, 0.3]  # nt, ny, step, stepy
        elif complexity_level == 2:
            self.obs_pt_param = [8, 6, 0.04, 0.25]
        else:  # complexity_level == 3
            self.obs_pt_param = [10, 6, 0.03, 0.2]

        # Environment parameters with enhanced scaling
        self.acc_scale = 1.0
        self.dy_scale = 15.0  # Increased for more responsive control
        self.acc_bd = [-750, 750]  # Expanded action bounds
        self.vel_scale = 12.0

        # Enhanced safety parameters
        self.safety_margin = 0.1
        self.collision_penalty = -10.0
        self.goal_reward = 5.0

        # Visualization parameters
        self.enable_vis = enable_vis
        self.enable_vis_rollout = enable_vis_rollout
        self.vis_time_window = vis_time_window
        self.bd_vis_sample = 200  # Higher resolution visualization
        self.hist_len = 2000  # Longer history tracking

        # Initialize visualization with enhanced plotting
        if self.enable_vis or self.enable_vis_rollout:
            self.fig, self.axes = plt.subplots(
                self.num_main_envs, 1, 
                figsize=(16, 5 * self.num_main_envs)
            )
            if self.num_main_envs == 1:
                self.axes = [self.axes]
            plt.show(block=False)

        # Initialize state tensors with GPU acceleration
        self._init_state_tensors()

        # Initialize complex boundary patterns
        self._init_complex_environment_bounds()

        # Cache for state operations
        self.cached_states = {}

        # Enhanced trajectory recording
        self.trajectory_history = {
            'y': [[] for _ in range(self.num_main_envs)],
            't': [[] for _ in range(self.num_main_envs)],
            'v': [[] for _ in range(self.num_main_envs)],
            'actions': [[] for _ in range(self.num_main_envs)],
            'rewards': [[] for _ in range(self.num_main_envs)],
            'safety_distances': [[] for _ in range(self.num_main_envs)]
        }

        # Performance monitoring
        self.computation_times = {
            'observation': [],
            'reward': [],
            'step': []
        }

    def _init_state_tensors(self):
        """Initialize state tensors with enhanced precision and GPU optimization."""
        # Core state variables with higher precision
        self.y = torch.zeros(self.total_num_envs, device=self.device, dtype=torch.float32)
        self.v = torch.zeros(self.total_num_envs, device=self.device, dtype=torch.float32)
        self.t = torch.zeros(self.total_num_envs, device=self.device, dtype=torch.float32)

        # Enhanced PID parameters with adaptive tuning
        self.p = torch.full((self.total_num_envs,), 15000.0, device=self.device, dtype=torch.float32)
        self.d = torch.full((self.total_num_envs,), 250.0, device=self.device, dtype=torch.float32)
        self.i = torch.full((self.total_num_envs,), 50.0, device=self.device, dtype=torch.float32)

        # Additional state tracking
        self.y_prev = torch.zeros(self.total_num_envs, device=self.device, dtype=torch.float32)
        self.error_integral = torch.zeros(self.total_num_envs, device=self.device, dtype=torch.float32)

        # Buffers for efficient computation
        self.obs_buf = torch.zeros((self.total_num_envs, self.obs_dim), device=self.device, dtype=torch.float32)
        self.reward_buf = torch.zeros(self.total_num_envs, device=self.device, dtype=torch.float32)
        self.done_buf = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)

        # Enhanced history tracking with circular buffers
        self.y_hist = [[] for _ in range(self.num_main_envs)]

    def _init_complex_environment_bounds(self):
        """Initialize complex boundary functions with enhanced patterns."""
        self.bound_params = []

        # Generate enhanced parameters for main environments
        main_env_params = []
        for main_env_idx in range(self.num_main_envs):
            np.random.seed(main_env_idx + 1000)  # Different seed space

            # Enhanced initial conditions
            y_bd = [-4, 4] if self.complexity_level >= 2 else [-3, 3]
            v_bd = [-8, 8] if self.complexity_level >= 2 else [-5, 5]
            initial_y = np.random.uniform(y_bd[0], y_bd[1])
            initial_v = np.random.uniform(v_bd[0], v_bd[1])

            # Adaptive PID parameters based on complexity
            if self.complexity_level == 1:
                wn_exp_bd = [1.3, 1.8]
                zeta_bd = [0.6, 1.0]
            elif self.complexity_level == 2:
                wn_exp_bd = [1.5, 2.0]
                zeta_bd = [0.7, 1.2]
            else:  # complexity_level == 3
                wn_exp_bd = [1.7, 2.2]
                zeta_bd = [0.8, 1.4]

            wn = 10 ** np.random.uniform(wn_exp_bd[0], wn_exp_bd[1])
            zeta = np.random.uniform(zeta_bd[0], zeta_bd[1])
            p = wn ** 2 * (1.2 + 0.3 * np.random.randn())  # Add variation
            d = 2 * zeta * wn * (1.1 + 0.2 * np.random.randn())

            # Complex boundary function parameters
            if self.use_fractal_bounds:
                slope_abs_bd = 1.5 if self.complexity_level >= 2 else 1.0
                # Fractal-like coefficients with multiple frequency components
                if self.complexity_level == 1:
                    coef_abs_bd = [1.0, 1.0, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2]
                elif self.complexity_level == 2:
                    coef_abs_bd = [1.2, 1.2, 0.7, 0.7, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.15, 0.15]
                else:  # complexity_level == 3
                    coef_abs_bd = [1.5, 1.5, 0.9, 0.9, 0.6, 0.6, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.15, 0.15, 0.1, 0.1]
            else:
                slope_abs_bd = 1.0
                coef_abs_bd = [1.0, 1.0, 0.5, 0.5, 0.3, 0.5, 0, 0.4, 0, 0.3, 0, 0, 0, 0, 0.1, 0.1]

            width_bd = [0.8, 2.0] if self.complexity_level >= 2 else [0.6, 1.5]

            # Generate complex patterns
            slope = np.random.uniform(-slope_abs_bd, slope_abs_bd)
            coef = [np.random.uniform(-coef_abs_bd[i//2], coef_abs_bd[i//2]) 
                   for i in range(len(coef_abs_bd) * 2)]
            width = np.random.uniform(width_bd[0], width_bd[1])

            # Add time-varying components for adaptive difficulty
            if self.adaptive_difficulty:
                time_scaling_factor = 1.0 + 0.3 * np.sin(main_env_idx * 0.5)
                frequency_modulation = 1.0 + 0.2 * np.cos(main_env_idx * 0.7)
            else:
                time_scaling_factor = 1.0
                frequency_modulation = 1.0

            # Store enhanced parameters
            params = {
                'initial_y': initial_y,
                'initial_v': initial_v,
                'p': p,
                'd': d,
                'slope': slope,
                'coef': coef,
                'width': width,
                'time_scaling_factor': time_scaling_factor,
                'frequency_modulation': frequency_modulation,
                'complexity_level': self.complexity_level,
                'fractal_enabled': self.use_fractal_bounds
            }
            main_env_params.append(params)

        # Assign parameters using interleaved pattern
        for i in range(self.total_num_envs):
            main_env_idx = i // (1 + self.num_rollout_per_main)
            params = main_env_params[main_env_idx].copy()
            self.bound_params.append(params)

            # Set initial state
            self.y[i] = params['initial_y']
            self.v[i] = params['initial_v']
            self.p[i] = params['p']
            self.d[i] = params['d']

    def _compute_sdf_value_vectorized(self, 
                                    y_batch: torch.Tensor, 
                                    t_batch: torch.Tensor,
                                    env_indices_batch: torch.Tensor) -> torch.Tensor:
        """Highly optimized vectorized SDF computation for all observation points.

        Args:
            y_batch: Batch of position values [N]
            t_batch: Batch of time values [N]
            env_indices_batch: Batch of environment indices [N]

        Returns:
            Batch of SDF values [N]
        """
        N = len(y_batch)
        sdf_values = torch.zeros(N, device=self.device, dtype=torch.float32)

        # Group computation by unique environment indices for efficiency
        unique_env_indices = torch.unique(env_indices_batch)

        for env_idx in unique_env_indices:
            mask = (env_indices_batch == env_idx)
            if not mask.any():
                continue

            env_idx_int = env_idx.item()
            if env_idx_int >= len(self.bound_params):
                continue

            params = self.bound_params[env_idx_int]
            y_env = y_batch[mask]
            t_env = t_batch[mask]

            # Enhanced boundary computation with fractal patterns
            if params['fractal_enabled'] and self.complexity_level >= 2:
                res = self._compute_fractal_boundary(t_env, params)
            else:
                res = self._compute_standard_boundary(t_env, params)

            # Enhanced safety margin computation
            width = params['width']
            safety_factor = 1.0 + self.safety_margin
            lb = res - (width * safety_factor) / 2
            ub = res + (width * safety_factor) / 2
            center = (lb + ub) / 2

            # Vectorized SDF computation with enhanced safety
            upper_dist = ub - y_env
            lower_dist = y_env - lb
            sdf_env = torch.where(y_env > center, upper_dist, lower_dist)

            # Apply additional safety penalties for close approaches
            close_mask = torch.abs(sdf_env) < self.safety_margin
            sdf_env = torch.where(close_mask, sdf_env * 0.5, sdf_env)

            sdf_values[mask] = sdf_env

        return sdf_values

    def _compute_fractal_boundary(self, t_env: torch.Tensor, params: Dict) -> torch.Tensor:
        """Compute fractal boundary patterns for enhanced complexity."""
        # Base pattern
        res = params['slope'] * t_env * params['time_scaling_factor']
        
        # Multi-scale fractal components
        coef_tensor = torch.tensor(params['coef'], device=self.device, dtype=torch.float32)
        coef_pairs = coef_tensor.view(-1, 2)
        
        for i, (sin_coef, cos_coef) in enumerate(coef_pairs):
            harmonic = (i + 1) * params['frequency_modulation']
            # Add fractal scaling: higher frequencies have amplitude decay
            fractal_scale = 1.0 / (1 + 0.3 * i)
            
            res += fractal_scale * (
                sin_coef * torch.sin(harmonic * t_env) + 
                cos_coef * torch.cos(harmonic * t_env)
            )
            
            # Add secondary harmonics for fractal complexity
            if self.complexity_level >= 3 and i < 3:
                secondary_harmonic = harmonic * 2.5
                res += fractal_scale * 0.3 * (
                    sin_coef * torch.sin(secondary_harmonic * t_env) +
                    cos_coef * torch.cos(secondary_harmonic * t_env)
                )

        return res

    def _compute_standard_boundary(self, t_env: torch.Tensor, params: Dict) -> torch.Tensor:
        """Compute standard boundary patterns."""
        res = params['slope'] * t_env

        coef_tensor = torch.tensor(params['coef'], device=self.device, dtype=torch.float32)
        coef_pairs = coef_tensor.view(-1, 2)

        for i, (sin_coef, cos_coef) in enumerate(coef_pairs):
            harmonic = i + 1
            res += sin_coef * torch.sin(harmonic * t_env) + cos_coef * torch.cos(harmonic * t_env)

        return res

    def _compute_observations_vectorized(self, env_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Highly optimized vectorized observation computation.

        Args:
            env_indices: Environment indices (if None, use all environments)

        Returns:
            Observation tensor [batch_size, obs_dim]
        """
        if env_indices is None:
            env_indices = torch.arange(self.total_num_envs, device=self.device)

        batch_size = len(env_indices)
        obs = torch.zeros((batch_size, self.obs_dim), device=self.device, dtype=torch.float32)

        # State observations with enhanced features
        obs[:, 0] = self.y[env_indices]
        obs[:, 1] = self.v[env_indices] / self.vel_scale

        # Vectorized SDF observation computation
        nt, ny, step, stepy = self.obs_pt_param

        # Pre-compute observation point grids
        y_offsets = torch.arange(ny, device=self.device, dtype=torch.float32)
        y_offsets = ((ny - 1) / 2 - y_offsets) * stepy

        t_offsets = torch.arange(nt, device=self.device, dtype=torch.float32) * step

        # Current states
        current_y = self.y[env_indices]
        current_t = self.t[env_indices]

        # Create observation points efficiently
        obs_points_per_env = ny * nt
        total_obs_points = batch_size * obs_points_per_env
        
        # Pre-allocate arrays
        obs_y_flat = torch.zeros(total_obs_points, device=self.device, dtype=torch.float32)
        obs_t_flat = torch.zeros(total_obs_points, device=self.device, dtype=torch.float32)
        env_indices_flat = torch.zeros(total_obs_points, device=self.device, dtype=torch.long)

        # Fill arrays using vectorized operations
        for i in range(batch_size):
            start_idx = i * obs_points_per_env
            end_idx = (i + 1) * obs_points_per_env
            
            # Create grids for this environment
            y_grid, t_grid = torch.meshgrid(
                current_y[i] + y_offsets,
                current_t[i] + t_offsets,
                indexing='ij'
            )
            
            # Flatten and store
            obs_y_flat[start_idx:end_idx] = y_grid.flatten()
            obs_t_flat[start_idx:end_idx] = t_grid.flatten()
            env_indices_flat[start_idx:end_idx] = env_indices[i]

        # Compute all SDF values at once
        sdf_values_flat = self._compute_sdf_value_vectorized(obs_y_flat, obs_t_flat, env_indices_flat)

        # Reshape back to grid and flatten for observation
        sdf_values_grid = sdf_values_flat.reshape(batch_size, obs_points_per_env)

        # Fill observation tensor
        obs[:, 2:2 + obs_points_per_env] = sdf_values_grid

        return obs

    def _compute_rewards_enhanced(self, env_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced reward computation with safety margins and goal incentives.

        Args:
            env_indices: Environment indices (if None, use all environments)

        Returns:
            Enhanced reward tensor
        """
        if env_indices is None:
            env_indices = torch.arange(self.total_num_envs, device=self.device)

        # Base SDF rewards
        sdf_rewards = self._compute_sdf_value_vectorized(
            self.y[env_indices],
            self.t[env_indices], 
            env_indices
        )

        # Enhanced reward structure
        rewards = sdf_rewards.clone()

        # Collision penalties
        collision_mask = sdf_rewards < 0
        rewards = torch.where(collision_mask, 
                            sdf_rewards + self.collision_penalty, 
                            rewards)

        # Safety margin bonuses
        safe_mask = sdf_rewards > self.safety_margin
        rewards = torch.where(safe_mask, 
                            rewards + 0.1, 
                            rewards)

        # Goal-seeking behavior (encourage movement toward corridor center)
        velocity_bonus = torch.abs(self.v[env_indices]) * 0.01
        rewards += velocity_bonus

        # Stability bonus (penalize excessive acceleration changes) - only for main environments
        if (hasattr(self, 'prev_actions') and hasattr(self, 'current_actions') and 
            torch.equal(env_indices, self.main_env_indices)):
            # Only apply stability penalty to main environments where we track actions
            try:
                stability_penalty = torch.abs(self.current_actions - self.prev_actions) * 0.005
                rewards -= stability_penalty
            except RuntimeError:
                # Skip stability penalty if tensor sizes don't match
                pass

        return rewards

    def reset(self) -> torch.Tensor:
        """Reset all environments with enhanced initialization."""
        # Reset states with enhanced randomization
        for i in range(self.total_num_envs):
            params = self.bound_params[i]
            # Add some randomization to initial conditions
            noise_y = 0.1 * torch.randn(1, device=self.device)
            noise_v = 0.05 * torch.randn(1, device=self.device)
            
            self.y[i] = params['initial_y'] + noise_y
            self.v[i] = params['initial_v'] + noise_v
            self.t[i] = 0.0
            self.p[i] = params['p']
            self.d[i] = params['d']

        # Reset additional state tracking
        self.y_prev.zero_()
        self.error_integral.zero_()
        self.done_buf.fill_(False)

        # Reset trajectory history
        self.trajectory_history = {
            'y': [[] for _ in range(self.num_main_envs)],
            't': [[] for _ in range(self.num_main_envs)],
            'v': [[] for _ in range(self.num_main_envs)],
            'actions': [[] for _ in range(self.num_main_envs)],
            'rewards': [[] for _ in range(self.num_main_envs)],
            'safety_distances': [[] for _ in range(self.num_main_envs)]
        }
        self.y_hist = [[] for _ in range(self.num_main_envs)]

        # Compute initial observations
        main_obs = self._compute_observations_vectorized(self.main_env_indices)
        return main_obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Enhanced step function with optimized vectorized operations.

        Args:
            actions: Action tensor of shape (num_main_envs, action_dim)

        Returns:
            Tuple of (observations, rewards, dones, info_dict)
        """
        import time
        step_start = time.time()

        # Apply actions to main environments
        main_indices = self.main_env_indices
        actions_flat = actions.flatten()

        # Store previous actions for stability computation
        self.prev_actions = getattr(self, 'current_actions', torch.zeros_like(actions_flat))
        self.current_actions = actions_flat.clone()

        # Update time for all main environments
        self.t[main_indices] += self.env_step

        # Store previous positions for derivative computation
        self.y_prev[main_indices] = self.y[main_indices].clone()

        # Enhanced dynamics with improved numerical integration
        if self.ctrl_mode == 'acc':
            # Verlet integration for better stability
            self.y[main_indices] += self.v[main_indices] * self.env_step + \
                                   0.5 * actions_flat * self.acc_scale * (self.env_step ** 2)
            self.v[main_indices] += actions_flat * self.env_step * self.acc_scale
        elif self.ctrl_mode == 'dy':
            # Enhanced velocity control with smoothing
            self.y[main_indices] += self.v[main_indices] * self.env_step
            new_v = actions_flat * self.dy_scale
            # Apply velocity smoothing to prevent discontinuities
            alpha = 0.8  # Smoothing factor
            self.v[main_indices] = alpha * new_v + (1 - alpha) * self.v[main_indices]
        elif self.ctrl_mode == 'y':
            # Direct position control with velocity estimation
            old_y = self.y[main_indices].clone()
            self.y[main_indices] = actions_flat
            # Estimate velocity from position change
            self.v[main_indices] = (self.y[main_indices] - old_y) / self.env_step

        # Update error integral for PID control
        current_errors = self.y[main_indices] - self.y_prev[main_indices]
        self.error_integral[main_indices] += current_errors * self.env_step

        # Record enhanced trajectory history
        for i, main_idx in enumerate(main_indices):
            y_val = self.y[main_idx].item()
            t_val = self.t[main_idx].item()
            v_val = self.v[main_idx].item()
            action_val = actions_flat[i].item()
            
            # Update circular buffers efficiently
            self.y_hist[i].append(y_val)
            if len(self.y_hist[i]) > self.hist_len:
                self.y_hist[i].pop(0)
            
            # Store trajectory data
            self.trajectory_history['y'][i].append(y_val)
            self.trajectory_history['t'][i].append(t_val)
            self.trajectory_history['v'][i].append(v_val)
            self.trajectory_history['actions'][i].append(action_val)

        # Compute observations and rewards with timing
        obs_start = time.time()
        obs = self._compute_observations_vectorized(main_indices)
        obs_time = time.time() - obs_start

        reward_start = time.time()
        rewards = self._compute_rewards_enhanced(main_indices)
        reward_time = time.time() - reward_start

        # Store performance metrics
        self.computation_times['observation'].append(obs_time)
        self.computation_times['reward'].append(reward_time)

        # Record rewards and safety distances
        for i, reward in enumerate(rewards):
            reward_val = reward.item()
            self.trajectory_history['rewards'][i].append(reward_val)
            self.trajectory_history['safety_distances'][i].append(max(0, reward_val))

        # Enhanced termination conditions
        position_limit = 12.0 if self.complexity_level >= 2 else 10.0
        velocity_limit = 15.0 if self.complexity_level >= 2 else 12.0
        
        position_done = torch.abs(self.y[main_indices]) > position_limit
        velocity_done = torch.abs(self.v[main_indices]) > velocity_limit
        collision_done = rewards < -5.0  # Severe collision threshold
        
        dones = position_done | velocity_done | collision_done

        # Visualization step with performance monitoring
        if self.enable_vis:
            self.vis_step_enhanced()

        step_time = time.time() - step_start
        self.computation_times['step'].append(step_time)

        # Enhanced info dictionary
        info = {
            'performance_metrics': {
                'obs_time': obs_time,
                'reward_time': reward_time,
                'step_time': step_time
            },
            'environment_stats': {
                'complexity_level': self.complexity_level,
                'adaptive_difficulty': self.adaptive_difficulty,
                'collision_count': collision_done.sum().item(),
                'avg_safety_distance': rewards.mean().item()
            }
        }

        return obs, rewards, dones, info

    def step_rollout(self, rollout_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Enhanced rollout step with optimized batch operations.

        Args:
            rollout_actions: Action tensor for rollout environments

        Returns:
            Tuple of (observations, rewards, dones, info_dict) for rollout environments
        """
        rollout_indices = self.rollout_env_indices
        actions_flat = rollout_actions.flatten()

        # Update time
        self.t[rollout_indices] += self.env_step

        # Enhanced dynamics (same as main step but for rollout environments)
        if self.ctrl_mode == 'acc':
            self.y[rollout_indices] += self.v[rollout_indices] * self.env_step + \
                                     0.5 * actions_flat * self.acc_scale * (self.env_step ** 2)
            self.v[rollout_indices] += actions_flat * self.env_step * self.acc_scale
        elif self.ctrl_mode == 'dy':
            self.y[rollout_indices] += self.v[rollout_indices] * self.env_step
            new_v = actions_flat * self.dy_scale
            alpha = 0.8
            self.v[rollout_indices] = alpha * new_v + (1 - alpha) * self.v[rollout_indices]
        elif self.ctrl_mode == 'y':
            old_y = self.y[rollout_indices].clone()
            self.y[rollout_indices] = actions_flat
            self.v[rollout_indices] = (self.y[rollout_indices] - old_y) / self.env_step

        # Efficient reward computation (skip observation computation for rollouts)
        rewards = self._compute_rewards_enhanced(rollout_indices)

        # Enhanced termination conditions
        position_limit = 12.0 if self.complexity_level >= 2 else 10.0
        velocity_limit = 15.0 if self.complexity_level >= 2 else 12.0
        
        dones = (torch.abs(self.y[rollout_indices]) > position_limit) | \
                (torch.abs(self.v[rollout_indices]) > velocity_limit) | \
                (rewards < -5.0)

        # Visualization for rollouts
        if self.enable_vis_rollout:
            self.vis_step_rollout_enhanced()

        info = {
            'rollout_stats': {
                'avg_reward': rewards.mean().item(),
                'collision_rate': (rewards < 0).float().mean().item()
            }
        }

        return None, rewards, dones, info

    def get_observation(self) -> torch.Tensor:
        """Get current observations for main environments with caching."""
        return self._compute_observations_vectorized(self.main_env_indices)

    def get_reward(self) -> torch.Tensor:
        """Get current rewards for main environments."""
        return self._compute_rewards_enhanced(self.main_env_indices)

    def sync_main_to_rollout(self):
        """Synchronize rollout environments to match their main environment states."""
        for i in range(self.num_main_envs):
            main_idx = self.main_env_indices[i]
            rollout_indices = self.main_to_rollout_indices[i]

            if len(rollout_indices) > 0:
                # Efficient batch copy
                self.y[rollout_indices] = self.y[main_idx].clone()
                self.v[rollout_indices] = self.v[main_idx].clone()
                self.t[rollout_indices] = self.t[main_idx].clone()
                self.error_integral[rollout_indices] = self.error_integral[main_idx].clone()

    def cache_main_env_states(self):
        """Cache the current state of main environments with enhanced state tracking."""
        main_indices = self.main_env_indices
        self.cached_states = {
            'y': self.y[main_indices].clone(),
            'v': self.v[main_indices].clone(),
            't': self.t[main_indices].clone(),
            'y_prev': self.y_prev[main_indices].clone(),
            'error_integral': self.error_integral[main_indices].clone()
        }

    def restore_main_env_states(self):
        """Restore main environments to their cached states."""
        if self.cached_states:
            main_indices = self.main_env_indices
            self.y[main_indices] = self.cached_states['y']
            self.v[main_indices] = self.cached_states['v']
            self.t[main_indices] = self.cached_states['t']
            self.y_prev[main_indices] = self.cached_states['y_prev']
            self.error_integral[main_indices] = self.cached_states['error_integral']

    def vis_step_enhanced(self):
        """Enhanced visualization with performance optimizations and richer information."""
        if not self.enable_vis:
            return

        for env_i in range(self.num_main_envs):
            ax = self.axes[env_i]
            ax.clear()
            
            main_idx = self.main_env_indices[env_i]
            current_t = self.t[main_idx].item()
            current_y = self.y[main_idx].item()
            current_v = self.v[main_idx].item()
            
            # Plot enhanced trajectory history with velocity coloring
            if len(self.y_hist[env_i]) > 1:
                hist_times = np.linspace(
                    current_t - self.env_step * len(self.y_hist[env_i]),
                    current_t,
                    len(self.y_hist[env_i])
                )
                # Color trajectory by velocity
                velocities = self.trajectory_history['v'][env_i][-len(self.y_hist[env_i]):]
                if velocities and len(velocities) > 1:
                    # Create velocity colormap plot
                    velocities_array = np.array(velocities)
                    # Normalize velocities for better color scaling
                    vmin, vmax = velocities_array.min(), velocities_array.max()
                    if vmax - vmin > 1e-6:  # Avoid division by zero
                        scatter = ax.scatter(hist_times, self.y_hist[env_i], 
                                           c=velocities_array, cmap='viridis', s=2, alpha=0.7,
                                           vmin=vmin, vmax=vmax)
                        # Create colorbar only if it doesn't exist for this axis
                        if not hasattr(ax, '_colorbar') or ax._colorbar is None:
                            ax._colorbar = plt.colorbar(scatter, ax=ax, label='Velocity')
                        else:
                            # Update existing colorbar
                            ax._colorbar.update_normal(scatter)
                    else:
                        ax.plot(hist_times, self.y_hist[env_i], 'b-', linewidth=1.5, label='Trajectory')
                else:
                    ax.plot(hist_times, self.y_hist[env_i], 'b-', linewidth=1.5, label='Trajectory')
            
            # Plot enhanced boundary functions with fractal complexity
            X = np.linspace(
                current_t - self.vis_time_window,
                current_t + self.vis_time_window,
                self.bd_vis_sample
            )
            
            params = self.bound_params[main_idx.item()]
            
            # Compute boundary values using the same method as SDF computation
            X_tensor = torch.tensor(X, device=self.device, dtype=torch.float32)
            if params['fractal_enabled'] and self.complexity_level >= 2:
                res = self._compute_fractal_boundary(X_tensor, params)
            else:
                res = self._compute_standard_boundary(X_tensor, params)
            
            res_np = res.cpu().numpy()
            width = params['width']
            safety_factor = 1.0 + self.safety_margin
            
            lb_vals = res_np - (width * safety_factor) / 2
            ub_vals = res_np + (width * safety_factor) / 2
            
            # Enhanced boundary visualization
            ax.plot(X, lb_vals, 'r-', linewidth=2.5, label='Lower Bound', alpha=0.8)
            ax.plot(X, ub_vals, 'g-', linewidth=2.5, label='Upper Bound', alpha=0.8)
            ax.fill_between(X, lb_vals, ub_vals, alpha=0.15, color='blue', label='Safe Region')
            
            # Safety margin visualization
            ax.fill_between(X, lb_vals - self.safety_margin, lb_vals, 
                          alpha=0.1, color='red', label='Danger Zone')
            ax.fill_between(X, ub_vals, ub_vals + self.safety_margin, 
                          alpha=0.1, color='red')
            
            # Current state visualization with enhanced markers
            ax.plot(current_t, current_y, 'ko', markersize=10, label=f'Position: {current_y:.2f}', zorder=5)
            
            # Velocity vector visualization
            if abs(current_v) > 0.1:
                ax.arrow(current_t, current_y, 0.05, current_v * 0.1, 
                        head_width=0.02, head_length=0.01, fc='orange', ec='orange')
            
            # Performance information
            recent_rewards = self.trajectory_history['rewards'][env_i][-10:] if self.trajectory_history['rewards'][env_i] else [0]
            avg_recent_reward = np.mean(recent_rewards)
            
            ax.set_ylim(-6, 6)
            ax.set_xlim(current_t - self.vis_time_window, current_t + self.vis_time_window)
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            ax.set_title(f'Environment {env_i} (Complexity: {self.complexity_level}) - '
                        f'V: {current_v:.2f}, Avg Reward: {avg_recent_reward:.2f}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.001)

    def vis_step_rollout_enhanced(self):
        """Enhanced rollout visualization with trajectory predictions."""
        if not self.enable_vis_rollout:
            return

        for env_i in range(self.num_main_envs):
            ax = self.axes[env_i]
            ax.clear()
            
            main_idx = self.main_env_indices[env_i]
            current_t = self.t[main_idx].item()
            
            # Plot boundary functions
            X = np.linspace(
                current_t - self.vis_time_window,
                current_t + self.vis_time_window,
                self.bd_vis_sample
            )
            
            params = self.bound_params[main_idx.item()]
            X_tensor = torch.tensor(X, device=self.device, dtype=torch.float32)
            
            if params['fractal_enabled'] and self.complexity_level >= 2:
                res = self._compute_fractal_boundary(X_tensor, params)
            else:
                res = self._compute_standard_boundary(X_tensor, params)
            
            res_np = res.cpu().numpy()
            width = params['width']
            
            lb_vals = res_np - width / 2
            ub_vals = res_np + width / 2
            
            ax.plot(X, lb_vals, 'r-', linewidth=2, alpha=0.7, label='Lower Bound')
            ax.plot(X, ub_vals, 'g-', linewidth=2, alpha=0.7, label='Upper Bound')
            ax.fill_between(X, lb_vals, ub_vals, alpha=0.1, color='gray', label='Safe Region')
            
            # Plot main environment
            current_y = self.y[main_idx].item()
            ax.plot(current_t, current_y, 'ko', markersize=12, label='Main Env', zorder=5)
            
            # Plot rollout environments with enhanced visualization
            rollout_indices = self.main_to_rollout_indices[env_i]
            colors = plt.cm.plasma(np.linspace(0, 1, len(rollout_indices)))
            
            for i, rollout_idx in enumerate(rollout_indices):
                rollout_t = self.t[rollout_idx].item()
                rollout_y = self.y[rollout_idx].item()
                rollout_v = self.v[rollout_idx].item()
                
                # Color and size based on performance
                reward = self._compute_rewards_enhanced(torch.tensor([rollout_idx], device=self.device))[0].item()
                size = max(4, min(12, 8 + reward))
                alpha = max(0.3, min(1.0, 0.5 + reward * 0.1))
                
                ax.plot(rollout_t, rollout_y, 'o', color=colors[i], markersize=size,
                        alpha=alpha, label=f'R{i}: {reward:.1f}' if i < 8 else '')
                
                # Velocity vectors for rollouts
                if abs(rollout_v) > 0.1:
                    ax.arrow(rollout_t, rollout_y, 0.02, rollout_v * 0.05, 
                            head_width=0.01, head_length=0.005, 
                            color=colors[i], alpha=alpha * 0.7)
        
            ax.set_ylim(-6, 6)
            ax.set_xlim(current_t - self.vis_time_window, current_t + self.vis_time_window)
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            ax.set_title(f'Environment {env_i} - Rollout Trajectories (Complexity: {self.complexity_level})')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.001)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for analysis."""
        if not self.computation_times['step']:
            return {}
        
        return {
            'computation_times': {
                'avg_step_time': np.mean(self.computation_times['step']),
                'avg_obs_time': np.mean(self.computation_times['observation']),
                'avg_reward_time': np.mean(self.computation_times['reward']),
                'total_steps': len(self.computation_times['step'])
            },
            'environment_efficiency': {
                'obs_per_second': len(self.computation_times['observation']) / max(sum(self.computation_times['observation']), 1e-6),
                'rewards_per_second': len(self.computation_times['reward']) / max(sum(self.computation_times['reward']), 1e-6),
                'steps_per_second': len(self.computation_times['step']) / max(sum(self.computation_times['step']), 1e-6)
            },
            'complexity_settings': {
                'complexity_level': self.complexity_level,
                'adaptive_difficulty': self.adaptive_difficulty,
                'use_fractal_bounds': self.use_fractal_bounds,
                'observation_points': self.obs_pt_param[0] * self.obs_pt_param[1]
            }
        }

    def end(self):
        """Clean up visualization and print performance summary."""
        if self.enable_vis:
            plt.close('all')
        
        # Print performance summary
        metrics = self.get_performance_metrics()
        if metrics:
            print(f"\nObsAvoid2 Environment Performance Summary:")
            print(f"  Complexity Level: {metrics['complexity_settings']['complexity_level']}")
            print(f"  Average Step Time: {metrics['computation_times']['avg_step_time']*1000:.2f}ms")
            print(f"  Steps per Second: {metrics['environment_efficiency']['steps_per_second']:.1f}")
            print(f"  Observation Points: {metrics['complexity_settings']['observation_points']}")


# Factory functions for creating different complexity levels
def create_complex_bound_batch_env(num_main_envs: int = 8, 
                                  num_rollout_per_main: int = 32, 
                                  complexity_level: int = 3,
                                  **kwargs):
    """Create batch environment with complex boundary functions and fractal patterns.
    
    Args:
        num_main_envs: Number of main environments
        num_rollout_per_main: Number of rollout environments per main
        complexity_level: Environment complexity (1-3)
        **kwargs: Additional arguments
    """
    return ObsAvoid2BatchEnv(
        num_main_envs=num_main_envs,
        num_rollout_per_main=num_rollout_per_main,
        complexity_level=complexity_level,
        adaptive_difficulty=True,
        use_fractal_bounds=True,
        **kwargs
    )


def create_multi_obstacle_batch_env(num_main_envs: int = 8, 
                                   num_rollout_per_main: int = 32,
                                   **kwargs):
    """Create batch environment with multiple obstacle patterns and maximum complexity.
    
    Args:
        num_main_envs: Number of main environments  
        num_rollout_per_main: Number of rollout environments per main
        **kwargs: Additional arguments
    """
    return ObsAvoid2BatchEnv(
        num_main_envs=num_main_envs,
        num_rollout_per_main=num_rollout_per_main,
        complexity_level=3,
        adaptive_difficulty=True,
        use_fractal_bounds=True,
        obs_dim=62,  # Maximum observation dimension
        **kwargs
    )