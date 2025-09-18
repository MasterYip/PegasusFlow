"""
Batch rollout wrapper for Obstacle1dEnv to support trajectory gradient sampling.

This module extends the obsavoid environment to support batch operations and
rollout functionality needed for trajectory optimization.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from matplotlib import pyplot as plt

from ..env_runner_base import BatchEnvBase


class ObsAvoidBatchEnv(BatchEnvBase):
    """Batch environment wrapper for Obstacle1dEnv that supports rollout operations.

    This class creates multiple instances of the obsavoid environment and provides
    batch operations for trajectory gradient sampling.
    """

    def __init__(self,
                 num_main_envs: int = 4,
                 num_rollout_per_main: int = 16,
                 device: str = "cuda:0",
                 env_step: float = 0.01,
                 obs_dim: int = 32,  # 2 state + 30 sdf observations (5*6)
                 action_dim: int = 1,
                 enable_vis: bool = False,
                 enable_vis_rollout: bool = False,
                 vis_time_window: float = 1.0,
                 ctrl_mode: str = "acc"):
        """Initialize the batch obsavoid environment.

        Args:
            num_main_envs: Number of main environments
            num_rollout_per_main: Number of rollout environments per main environment
            device: Device for tensor operations
            env_step: Environment timestep
            obs_dim: Observation dimension
            action_dim: Action dimension
            enable_vis: Whether to enable visualization for main environments
            enable_vis_rollout: Whether to enable visualization for rollout environments
            vis_time_window: Time window for visualization
            ctrl_mode: Control mode - 'acc' for acceleration, 'dy' for velocity change, 'y' for position
        """
        super().__init__(num_main_envs, num_rollout_per_main, device)

        self.env_step = env_step
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_actions = action_dim
        self.dt = env_step

        # Control mode validation and setup
        assert ctrl_mode in ['acc', 'dy', 'y'], f"Invalid ctrl_mode: {ctrl_mode}. Must be 'acc', 'dy', or 'y'"
        self.ctrl_mode = ctrl_mode

        # Environment parameters (from original obsavoid env)
        self.obs_pt_param = [6, 5, 0.05, 0.3]  # nt, ny, step, stepy
        self.acc_scale = 1.0
        self.dy_scale = 10.0
        self.acc_bd = [-500, 500]
        self.vel_scale = 10.0

        # Visualization parameters
        self.enable_vis = enable_vis
        self.enable_vis_rollout = enable_vis_rollout
        self.vis_time_window = vis_time_window
        self.bd_vis_sample = 100
        self.hist_len = 1000

        # Initialize visualization with automatic grid layout
        if self.enable_vis or self.enable_vis_rollout:
            # Calculate optimal grid layout
            if self.num_main_envs <= 4:
                # Single column for <= 4 environments
                self.fig_rows, self.fig_cols = self.num_main_envs, 1
                figsize = (12, 4 * self.num_main_envs)
            else:
                # Grid layout for > 4 environments
                self.fig_cols = int(np.ceil(np.sqrt(self.num_main_envs)))
                self.fig_rows = int(np.ceil(self.num_main_envs / self.fig_cols))
                figsize = (6 * self.fig_cols, 4 * self.fig_rows)
            
            self.fig, self.axes = plt.subplots(self.fig_rows, self.fig_cols, figsize=figsize)
            
            # Ensure axes is always a 2D array for consistent indexing
            if self.fig_rows == 1 and self.fig_cols == 1:
                self.axes = np.array([[self.axes]])
            elif self.fig_rows == 1:
                self.axes = self.axes.reshape(1, -1)
            elif self.fig_cols == 1:
                self.axes = self.axes.reshape(-1, 1)
            
            # Hide unused subplots if num_main_envs < grid size
            total_subplots = self.fig_rows * self.fig_cols
            for i in range(self.num_main_envs, total_subplots):
                row = i // self.fig_cols
                col = i % self.fig_cols
                self.axes[row, col].set_visible(False)
            
            plt.show(block=False)

        # Initialize state tensors for all environments
        self._init_state_tensors()

        # Initialize environments with different bounds
        self._init_environment_bounds()

        # Cache for state saving/restoring
        self.cached_states = {}

        # Trajectory recording for visualization
        self.trajectory_history = {
            'y': [[] for _ in range(self.num_main_envs)],
            't': [[] for _ in range(self.num_main_envs)],
            'actions': [[] for _ in range(self.num_main_envs)],
            'rewards': [[] for _ in range(self.num_main_envs)]
        }

        # Initialize pre-computed arrays for efficient boundary visualization
        self._init_visualization_cache()

    def _init_state_tensors(self):
        """Initialize state tensors for all environments."""
        # Position, velocity, time for each environment
        self.y = torch.zeros(self.total_num_envs, device=self.device)
        self.v = torch.zeros(self.total_num_envs, device=self.device)
        self.t = torch.zeros(self.total_num_envs, device=self.device)

        # PID parameters for each environment
        self.p = torch.full((self.total_num_envs,), 10000.0, device=self.device)
        self.d = torch.full((self.total_num_envs,), 200.0, device=self.device)

        # Observation and reward buffers
        self.obs_buf = torch.zeros((self.total_num_envs, self.obs_dim), device=self.device)
        self.reward_buf = torch.zeros(self.total_num_envs, device=self.device)
        self.done_buf = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)

        # History tracking for visualization (only for main environments)
        self.y_hist = [[] for _ in range(self.num_main_envs)]

    def _init_environment_bounds(self):
        """Initialize different boundary functions for each environment."""
        # Store boundary parameters for each environment
        self.bound_params = []

        # First, generate parameters for main environments
        main_env_params = []
        for main_env_idx in range(self.num_main_envs):
            # Use different random parameters for each main environment
            np.random.seed(main_env_idx)  # Ensure reproducibility

            # Random path parameters (similar to randpath_bound_env)
            y_bd = [-20, 20]
            v_bd = [-20, 20]
            initial_y = np.random.rand() * (y_bd[1] - y_bd[0]) + y_bd[0]
            initial_v = np.random.rand() * (v_bd[1] - v_bd[0]) + v_bd[0]

            # PID parameters
            wn_exp_bd = [1.3, 1.8]
            zeta_bd = [0.6, 1.0]
            wn = 10 ** (np.random.rand() * (wn_exp_bd[1] - wn_exp_bd[0]) + wn_exp_bd[0])
            zeta = np.random.rand() * (zeta_bd[1] - zeta_bd[0]) + zeta_bd[0]
            p = wn ** 2
            d = 2 * zeta * wn

            # Boundary function parameters
            slope_abs_bd = 1.0
            coef_abs_bd = [1.0, 1.0, 0.5, 0.5, 0.3, 0.5, 0, 0.4, 0, 0.3, 0, 0, 0, 0, 0.1, 0.1]
            width_bd = [0.6, 1.5]
            slope = (np.random.rand() - 0.5) * 2 * slope_abs_bd
            coef = [(np.random.rand() - 0.5) * 2 * coef_abs_bd[i // 2] for i in range(len(coef_abs_bd) * 2)]
            width = np.random.rand() * (width_bd[1] - width_bd[0]) + width_bd[0]

            # Store parameters for this main environment
            params = {
                'initial_y': initial_y,
                'initial_v': initial_v,
                'p': p,
                'd': d,
                'slope': slope,
                'coef': coef,
                'width': width
            }
            main_env_params.append(params)

        # Now assign parameters using the correct interleaved pattern
        # Pattern: [main_0, rollout_0_0, rollout_0_1, ..., main_1, rollout_1_0, ...]
        for i in range(self.total_num_envs):
            # Determine which main environment this index corresponds to
            main_env_idx = i // (1 + self.num_rollout_per_main)
            
            # Use the parameters from the corresponding main environment
            params = main_env_params[main_env_idx].copy()
            
            self.bound_params.append(params)

            # Set initial state
            self.y[i] = params['initial_y']
            self.v[i] = params['initial_v']
            self.p[i] = params['p']
            self.d[i] = params['d']

        # Pre-compute coefficient matrices for efficient SDF computation
        self._init_fourier_matrices()

    def _init_fourier_matrices(self):
        """Initialize pre-computed matrices for efficient Fourier series evaluation."""
        # Extract coefficients and parameters for all main environments
        num_harmonics = len(self.bound_params[0]['coef']) // 2  # Number of Fourier harmonics
        
        # Coefficient matrix: [num_main_envs, num_harmonics, 2] where 2 = [sin_coef, cos_coef]
        self.fourier_coef_matrix = torch.zeros((self.num_main_envs, num_harmonics, 2), device=self.device)
        
        # Slope and width vectors for each main environment
        self.slopes = torch.zeros(self.num_main_envs, device=self.device)
        self.widths = torch.zeros(self.num_main_envs, device=self.device)
        
        # Fill the matrices with parameters from main environments
        for main_env_idx in range(self.num_main_envs):
            # Get the actual environment index for this main environment
            actual_env_idx = main_env_idx * (1 + self.num_rollout_per_main)
            params = self.bound_params[actual_env_idx]
            
            # Extract slope and width
            self.slopes[main_env_idx] = params['slope']
            self.widths[main_env_idx] = params['width']
            
            # Extract Fourier coefficients
            coef = params['coef']
            for harmonic_idx in range(num_harmonics):
                if 2 * harmonic_idx + 1 < len(coef):
                    self.fourier_coef_matrix[main_env_idx, harmonic_idx, 0] = coef[2 * harmonic_idx]      # sin coef
                    self.fourier_coef_matrix[main_env_idx, harmonic_idx, 1] = coef[2 * harmonic_idx + 1]  # cos coef
        
        # Create expansion indices to map from main environments to all environments
        # This maps each environment index to its corresponding main environment
        self.env_to_main_idx = torch.zeros(self.total_num_envs, dtype=torch.long, device=self.device)
        for env_idx in range(self.total_num_envs):
            main_env_idx = env_idx // (1 + self.num_rollout_per_main)
            self.env_to_main_idx[env_idx] = main_env_idx

    def _compute_sdf_value_batch(self, y_batch: torch.Tensor, t_batch: torch.Tensor,
                                 env_indices_batch: torch.Tensor) -> torch.Tensor:
        """Compute SDF values for a batch of positions and times efficiently using vectorized operations.

        This optimized version uses pre-computed Fourier coefficient matrices and vectorized operations
        to efficiently compute SDF values for all environments simultaneously.

        Args:
            y_batch: Batch of position values [N]
            t_batch: Batch of time values [N]
            env_indices_batch: Batch of environment indices [N]

        Returns:
            Batch of SDF values [N]
        """
        N = len(y_batch)
        device = self.device
        
        # Step 1: Map environment indices to main environment indices
        main_env_indices = self.env_to_main_idx[env_indices_batch]  # [N]
        
        # Step 2: Evaluate Fourier basis functions at all time points
        # Get number of harmonics from the coefficient matrix
        num_harmonics = self.fourier_coef_matrix.shape[1]
        
        # Create harmonic indices [1, 2, 3, ..., num_harmonics]
        harmonics = torch.arange(1, num_harmonics + 1, device=device, dtype=torch.float32)  # [num_harmonics]
        
        # Compute harmonic * t for all combinations: [N, num_harmonics]
        harmonic_t = t_batch.unsqueeze(1) * harmonics.unsqueeze(0)  # [N, num_harmonics]
        
        # Compute sin and cos values: [N, num_harmonics]
        sin_vals = torch.sin(harmonic_t)  # [N, num_harmonics]
        cos_vals = torch.cos(harmonic_t)  # [N, num_harmonics]
        
        # Step 3: Get coefficients for each point's main environment
        # fourier_coef_matrix: [num_main_envs, num_harmonics, 2]
        # main_env_indices: [N]
        point_coefs = self.fourier_coef_matrix[main_env_indices]  # [N, num_harmonics, 2]
        sin_coefs = point_coefs[:, :, 0]  # [N, num_harmonics]
        cos_coefs = point_coefs[:, :, 1]  # [N, num_harmonics]
        
        # Step 4: Compute Fourier series contributions
        fourier_contrib = torch.sum(sin_coefs * sin_vals + cos_coefs * cos_vals, dim=1)  # [N]
        
        # Step 5: Add slope contributions
        slopes_expanded = self.slopes[main_env_indices]  # [N]
        centers = slopes_expanded * t_batch + fourier_contrib  # [N]
        
        # Step 6: Get widths for each point
        widths_expanded = self.widths[main_env_indices]  # [N]
        
        # Step 7: Compute SDF values efficiently
        # SDF = 0.5 * width - |center - y|
        sdf_values = 0.5 * widths_expanded - torch.abs(centers - y_batch)  # [N]
        
        return sdf_values

    def _compute_observations(self, env_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute observations for specified environments.

        Args:
            env_indices: Environment indices (if None, use all environments)

        Returns:
            Observation tensor
        """
        if env_indices is None:
            env_indices = torch.arange(self.total_num_envs, device=self.device)

        batch_size = len(env_indices)
        obs = torch.zeros((batch_size, self.obs_dim), device=self.device)

        # State observations: [y, v/vel_scale] - vectorized
        obs[:, 0] = self.y[env_indices]
        obs[:, 1] = self.v[env_indices] / self.vel_scale

        # SDF observations at observation points - vectorized
        nt, ny, step, stepy = self.obs_pt_param

        # Pre-compute all observation points for all environments
        # Create offset grids
        y_offsets = torch.arange(ny, device=self.device, dtype=torch.float32)
        y_offsets = ((ny - 1) / 2 - y_offsets) * stepy  # Shape: [ny]

        t_offsets = torch.arange(nt, device=self.device, dtype=torch.float32) * step  # Shape: [nt]

        # Get current positions and times for all environments
        current_y = self.y[env_indices]  # [batch_size]
        current_t = self.t[env_indices]  # [batch_size]

        # Create all observation points using explicit loops to avoid broadcasting issues
        obs_y_list = []
        obs_t_list = []
        env_idx_list = []

        for i, env_idx in enumerate(env_indices):
            for yi in range(ny):
                for ti in range(nt):
                    obs_y = current_y[i] + y_offsets[yi]
                    obs_t = current_t[i] + t_offsets[ti]
                    obs_y_list.append(obs_y)
                    obs_t_list.append(obs_t)
                    env_idx_list.append(env_idx)
        
        # Convert to tensors
        obs_y_flat = torch.stack(obs_y_list)  # [batch_size * ny * nt]
        obs_t_flat = torch.stack(obs_t_list)  # [batch_size * ny * nt]
        env_indices_flat = torch.stack(env_idx_list)  # [batch_size * ny * nt]

        # Compute SDF values for all points at once
        sdf_values_flat = self._compute_sdf_value_batch(obs_y_flat, obs_t_flat, env_indices_flat)

        # Reshape back to [batch_size, ny * nt]
        sdf_values_ordered = sdf_values_flat.reshape(batch_size, ny * nt)

        # Fill the observation tensor
        obs[:, 2:2 + ny * nt] = sdf_values_ordered

        return obs

    def _compute_rewards(self, env_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute rewards for specified environments.

        Args:
            env_indices: Environment indices (if None, use all environments)

        Returns:
            Reward tensor
        """
        if env_indices is None:
            env_indices = torch.arange(self.total_num_envs, device=self.device)

        rewards = self._compute_sdf_value_batch(
            self.y[env_indices],
            self.t[env_indices],
            env_indices
        )

        return rewards

    def reset(self) -> torch.Tensor:
        """Reset all environments and return initial observations."""
        # Reset states
        for i in range(self.total_num_envs):
            params = self.bound_params[i]
            self.y[i] = params['initial_y']
            self.v[i] = params['initial_v']
            self.t[i] = 0.0
            self.p[i] = params['p']
            self.d[i] = params['d']

        # Reset buffers
        self.done_buf.fill_(False)

        # Reset trajectory history and visualization history
        self.trajectory_history = {
            'y': [[] for _ in range(self.num_main_envs)],
            't': [[] for _ in range(self.num_main_envs)],
            'actions': [[] for _ in range(self.num_main_envs)],
            'rewards': [[] for _ in range(self.num_main_envs)]
        }
        self.y_hist = [[] for _ in range(self.num_main_envs)]

        # Compute initial observations
        main_obs = self._compute_observations(self.main_env_indices)
        return main_obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step the main environments with given actions.

        Args:
            actions: Action tensor of shape (num_main_envs, action_dim)

        Returns:
            Tuple of (observations, rewards, dones, info_dict)
        """
        # Apply actions to main environments
        main_indices = self.main_env_indices
        actions_flat = actions.flatten()

        # Ensure we have the right number of actions
        assert len(actions_flat) == len(main_indices), f"Actions length {len(actions_flat)} != main envs {len(main_indices)}"

        # Update dynamics for main environments
        self.t[main_indices] += self.env_step

        if self.ctrl_mode == 'acc':
            # Acceleration control
            self.y[main_indices] += self.v[main_indices] * self.env_step
            self.v[main_indices] += actions_flat * self.env_step * self.acc_scale
        elif self.ctrl_mode == 'dy':
            # Velocity change control
            self.y[main_indices] += self.v[main_indices] * self.env_step
            self.v[main_indices] = actions_flat * self.env_step * self.dy_scale
        elif self.ctrl_mode == 'y':
            # Direct position control
            self.y[main_indices] = actions_flat
            # Velocity remains unchanged

        # Record trajectory history for visualization
        for i, main_idx in enumerate(main_indices):
            y_val = self.y[main_idx].item()
            t_val = self.t[main_idx].item()
            action_val = actions_flat[i].item()
            
            self.y_hist[i].append(y_val)
            if len(self.y_hist[i]) > self.hist_len:
                self.y_hist[i].pop(0)
            
            self.trajectory_history['y'][i].append(y_val)
            self.trajectory_history['t'][i].append(t_val)
            self.trajectory_history['actions'][i].append(action_val)

        # Compute observations and rewards
        obs = self._compute_observations(main_indices)
        rewards = self._compute_rewards(main_indices)

        # Record rewards
        for i, reward in enumerate(rewards):
            self.trajectory_history['rewards'][i].append(reward.item())

        # Check for termination (simple termination condition)
        dones = torch.abs(self.y[main_indices]) > 10.0

        # Visualization step
        if self.enable_vis:
            self.vis_step()

        info = {}
        return obs, rewards, dones, info

    def step_rollout(self, rollout_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step rollout environments with given actions.

        Args:
            rollout_actions: Action tensor for rollout environments

        Returns:
            Tuple of (observations, rewards, dones, info_dict) for rollout environments
        """
        # Apply actions to rollout environments
        rollout_indices = self.rollout_env_indices
        actions_flat = rollout_actions.flatten()

        # Update dynamics for rollout environments
        self.t[rollout_indices] += self.env_step

        if self.ctrl_mode == 'acc':
            # Acceleration control
            self.y[rollout_indices] += self.v[rollout_indices] * self.env_step
            self.v[rollout_indices] += actions_flat * self.env_step * self.acc_scale
        elif self.ctrl_mode == 'dy':
            # Velocity change control
            self.y[rollout_indices] += self.v[rollout_indices] * self.env_step
            self.v[rollout_indices] = actions_flat * self.env_step * self.dy_scale
        elif self.ctrl_mode == 'y':
            # Direct position control
            self.y[rollout_indices] = actions_flat
            # Velocity remains unchanged

        # Compute observations and rewards
        # TODO: this is too time consuming, disable for now
        # obs = self._compute_observations(rollout_indices)
        obs = None
        rewards = self._compute_rewards(rollout_indices)

        # Check for termination
        dones = torch.abs(self.y[rollout_indices]) > 10.0

        # Visualization step
        if self.enable_vis_rollout:
            self.vis_step_rollout()

        info = {}
        return obs, rewards, dones, info

    def get_observation(self) -> torch.Tensor:
        """Get current observations for main environments."""
        return self._compute_observations(self.main_env_indices)

    def get_reward(self) -> torch.Tensor:
        """Get current rewards for main environments."""
        return self._compute_rewards(self.main_env_indices)

    def sync_main_to_rollout(self):
        """Synchronize rollout environments to match their main environment states."""
        for i in range(self.num_main_envs):
            main_idx = self.main_env_indices[i]
            rollout_indices = self.main_to_rollout_indices[i]

            if len(rollout_indices) > 0:
                # Copy state from main environment to its rollout environments
                self.y[rollout_indices] = self.y[main_idx].clone()
                self.v[rollout_indices] = self.v[main_idx].clone()
                self.t[rollout_indices] = self.t[main_idx].clone()

    def cache_main_env_states(self):
        """Cache the current state of main environments."""
        main_indices = self.main_env_indices
        self.cached_states = {
            'y': self.y[main_indices].clone(),
            'v': self.v[main_indices].clone(),
            't': self.t[main_indices].clone()
        }

    def restore_main_env_states(self):
        """Restore main environments to their cached states."""
        if self.cached_states:
            main_indices = self.main_env_indices
            self.y[main_indices] = self.cached_states['y']
            self.v[main_indices] = self.cached_states['v']
            self.t[main_indices] = self.cached_states['t']

    def _compute_boundaries_vectorized(self, current_times: torch.Tensor) -> torch.Tensor:
        """Compute boundaries for all main environments efficiently using vectorized operations.
        
        Args:
            current_times: Current time for each main environment [num_main_envs]
            
        Returns:
            Boundaries tensor [num_main_envs, bd_vis_sample, 2] where 2 = [lower, upper]
        """
        # Create time arrays for each environment: [num_main_envs, bd_vis_sample]
        time_arrays = current_times.unsqueeze(1) + self.vis_time_array.unsqueeze(0)  # [num_main_envs, bd_vis_sample]
        
        # Compute Fourier series for all environments and all time points
        # harmonic_t: [num_main_envs, bd_vis_sample, num_harmonics]
        harmonic_t = time_arrays.unsqueeze(2) * self.vis_harmonics.unsqueeze(0).unsqueeze(0)
        
        # Compute sin and cos values: [num_main_envs, bd_vis_sample, num_harmonics]
        sin_vals = torch.sin(harmonic_t)
        cos_vals = torch.cos(harmonic_t)
        
        # Get coefficients for all main environments: [num_main_envs, num_harmonics, 2]
        sin_coefs = self.fourier_coef_matrix[:, :, 0]  # [num_main_envs, num_harmonics]
        cos_coefs = self.fourier_coef_matrix[:, :, 1]  # [num_main_envs, num_harmonics]
        
        # Compute Fourier contributions: [num_main_envs, bd_vis_sample]
        fourier_contrib = torch.sum(
            sin_coefs.unsqueeze(1) * sin_vals + cos_coefs.unsqueeze(1) * cos_vals, 
            dim=2
        )
        
        # Add slope contributions: [num_main_envs, bd_vis_sample]
        centers = self.slopes.unsqueeze(1) * time_arrays + fourier_contrib
        
        # Compute boundaries: [num_main_envs, bd_vis_sample, 2]
        half_widths = self.widths.unsqueeze(1) / 2  # [num_main_envs, 1]
        boundaries = torch.stack([
            centers - half_widths,  # lower bounds
            centers + half_widths   # upper bounds
        ], dim=2)
        
        return boundaries

    def vis_step(self):
        """Optimized visualization for all main environments using vectorized boundary computation."""
        if not self.enable_vis:
            return
        
        # Get current times for all main environments
        current_times = self.t[self.main_env_indices]  # [num_main_envs]
        
        # Compute boundaries for all environments at once
        boundaries = self._compute_boundaries_vectorized(current_times)  # [num_main_envs, bd_vis_sample, 2]
        boundaries_cpu = boundaries.cpu().numpy()
        
        # Convert time array to CPU for plotting
        vis_time_cpu = self.vis_time_array.cpu().numpy()
        current_times_cpu = current_times.cpu().numpy()
        
        # Plot each environment
        for env_i in range(self.num_main_envs):
            # Calculate subplot position in grid
            row = env_i // self.fig_cols
            col = env_i % self.fig_cols
            ax = self.axes[row, col]
            ax.clear()
            
            current_t = current_times_cpu[env_i]
            
            # Time array for this environment (shifted to current time)
            X = vis_time_cpu + current_t
            
            # Get boundary values for this environment
            lb_vals = boundaries_cpu[env_i, :, 0]  # lower bounds
            ub_vals = boundaries_cpu[env_i, :, 1]  # upper bounds
            
            # Plot trajectory history
            if len(self.y_hist[env_i]) > 0:
                hist_times = np.linspace(
                    current_t - self.env_step * len(self.y_hist[env_i]),
                    current_t,
                    len(self.y_hist[env_i])
                )
                ax.plot(hist_times, self.y_hist[env_i], 'b-', linewidth=2, label='Trajectory')
            
            # Plot boundaries
            ax.plot(X, lb_vals, 'r-', linewidth=2, label='Lower Bound')
            ax.plot(X, ub_vals, 'g-', linewidth=2, label='Upper Bound')
            ax.fill_between(X, lb_vals, ub_vals, alpha=0.1, color='gray', label='Safe Region')
            
            # Current position marker
            main_idx = self.main_env_indices[env_i]
            current_y = self.y[main_idx].item()
            ax.plot(current_t, current_y, 'ko', markersize=8, label='Current Position')
            
            # Formatting
            ax.set_ylim(-4, 4)
            ax.set_xlim(current_t - self.vis_time_window, current_t + self.vis_time_window)
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            ax.set_title(f'Environment {env_i}')
            if env_i == 0:  # Only show legend on first subplot to avoid clutter
                ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.001)

    def vis_step_rollout(self):
        """Optimized visualization for rollout environments using vectorized boundary computation."""
        if not self.enable_vis_rollout:
            return
        
        # Get current times for all main environments
        current_times = self.t[self.main_env_indices]  # [num_main_envs]
        
        # Compute boundaries for all environments at once
        boundaries = self._compute_boundaries_vectorized(current_times)  # [num_main_envs, bd_vis_sample, 2]
        boundaries_cpu = boundaries.cpu().numpy()
        
        # Convert time array to CPU for plotting
        vis_time_cpu = self.vis_time_array.cpu().numpy()
        current_times_cpu = current_times.cpu().numpy()
        
        # Plot each environment
        for env_i in range(self.num_main_envs):
            # Calculate subplot position in grid
            row = env_i // self.fig_cols
            col = env_i % self.fig_cols
            ax = self.axes[row, col]
            ax.clear()
            
            current_t = current_times_cpu[env_i]
            
            # Time array for this environment (shifted to current time)
            X = vis_time_cpu + current_t
            
            # Get boundary values for this environment
            lb_vals = boundaries_cpu[env_i, :, 0]  # lower bounds
            ub_vals = boundaries_cpu[env_i, :, 1]  # upper bounds
            
            # Plot boundaries
            ax.plot(X, lb_vals, 'r-', linewidth=2, alpha=0.7, label='Lower Bound')
            ax.plot(X, ub_vals, 'g-', linewidth=2, alpha=0.7, label='Upper Bound')
            ax.fill_between(X, lb_vals, ub_vals, alpha=0.1, color='gray', label='Safe Region')
            
            # Plot main environment position
            main_idx = self.main_env_indices[env_i]
            current_y = self.y[main_idx].item()
            ax.plot(current_t, current_y, 'ko', markersize=10,
                    label='Main Env Position', zorder=5)
            
            # Plot all rollout environments for this main environment
            rollout_indices = self.main_to_rollout_indices[env_i]
            if len(rollout_indices) > 0:
                colors = plt.cm.viridis(np.linspace(0, 1, len(rollout_indices)))
                
                # Get all rollout positions and times at once
                rollout_times = self.t[rollout_indices].cpu().numpy()
                rollout_positions = self.y[rollout_indices].cpu().numpy()
                
                # Plot rollout environments
                for i, (rollout_t, rollout_y) in enumerate(zip(rollout_times, rollout_positions)):
                    ax.plot(rollout_t, rollout_y, 'o', color=colors[i], markersize=6,
                            alpha=0.7, label=f'Rollout {i}' if i < 3 else '')  # Only label first 3
            
            # Formatting
            ax.set_ylim(-4, 4)
            ax.set_xlim(current_t - self.vis_time_window, current_t + self.vis_time_window)
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            ax.set_title(f'Environment {env_i} - Rollout')
            if env_i == 0:  # Only show legend on first subplot
                ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.001)

    def replay_trajectory(self, env_idx: int = 0, save_path: Optional[str] = None):
        """Replay the recorded trajectory for visualization.
        
        Args:
            env_idx: Environment index to replay
            save_path: Optional path to save the replay as an image
        """
        if env_idx >= self.num_main_envs:
            print(f"Error: env_idx {env_idx} >= num_main_envs {self.num_main_envs}")
            return

        if not self.trajectory_history['t'][env_idx]:
            print(f"No trajectory data recorded for environment {env_idx}")
            return

        # Create a new figure for replay
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Get trajectory data
        times = np.array(self.trajectory_history['t'][env_idx])
        positions = np.array(self.trajectory_history['y'][env_idx])
        actions = np.array(self.trajectory_history['actions'][env_idx])
        rewards = np.array(self.trajectory_history['rewards'][env_idx])
        
        main_idx = self.main_env_indices[env_idx].item()
        params = self.bound_params[main_idx]
        
        # Plot 1: Trajectory with boundaries
        ax1.plot(times, positions, 'b-', linewidth=2, label='Trajectory')
        
        # Plot boundary functions over the entire time range
        t_range = np.linspace(times.min(), times.max(), 1000)
        lb_vals = []
        ub_vals = []
        
        for t_val in t_range:
            res = params['slope'] * t_val
            for i in range(len(params['coef']) // 2):
                res += params['coef'][i * 2] * np.sin((i + 1) * t_val) + \
                       params['coef'][i * 2 + 1] * np.cos((i + 1) * t_val)
            
            width = params['width']
            lb_vals.append(res - width / 2)
            ub_vals.append(res + width / 2)
        
        ax1.plot(t_range, lb_vals, 'r-', linewidth=1, alpha=0.7, label='Lower Bound')
        ax1.plot(t_range, ub_vals, 'g-', linewidth=1, alpha=0.7, label='Upper Bound')
        ax1.fill_between(t_range, lb_vals, ub_vals, alpha=0.2, color='gray', label='Safe Region')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Position')
        ax1.set_title(f'Environment {env_idx} - Trajectory Replay')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Actions over time
        ax2.plot(times[1:], actions[1:], 'orange', linewidth=1.5, label='Actions')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Action (Acceleration)')
        ax2.set_title('Control Actions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rewards over time
        ax3.plot(times, rewards, 'purple', linewidth=1.5, label='Rewards (SDF)')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Collision Threshold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Reward (SDF Value)')
        ax3.set_title('Safety Distance (SDF) Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory replay saved to {save_path}")
        
        plt.show(block=False)
        
        # Print summary statistics
        total_reward = np.sum(rewards)
        avg_reward = np.mean(rewards)
        min_sdf = np.min(rewards)
        violations = np.sum(rewards < 0)
        
        print(f"\nTrajectory Summary for Environment {env_idx}:")
        print(f"  Total steps: {len(times)}")
        print(f"  Total time: {times[-1] - times[0]:.3f} seconds")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  Minimum SDF: {min_sdf:.3f}")
        print(f"  Safety violations: {violations} steps")
        print(f"  Success rate: {(len(times) - violations) / len(times) * 100:.1f}%")

    def replay_all_trajectories(self, save_dir: Optional[str] = None):
        """Replay trajectories for all main environments.
        
        Args:
            save_dir: Optional directory to save replay images
        """
        for env_idx in range(self.num_main_envs):
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"trajectory_replay_env_{env_idx}.png")
            
            self.replay_trajectory(env_idx, save_path)

    def get_trajectory_data(self, env_idx: int = 0) -> Dict[str, np.ndarray]:
        """Get trajectory data for analysis.
        
        Args:
            env_idx: Environment index
            
        Returns:
            Dictionary containing trajectory data
        """
        if env_idx >= self.num_main_envs:
            return {}
        
        return {
            'times': np.array(self.trajectory_history['t'][env_idx]),
            'positions': np.array(self.trajectory_history['y'][env_idx]),
            'actions': np.array(self.trajectory_history['actions'][env_idx]),
            'rewards': np.array(self.trajectory_history['rewards'][env_idx])
        }

    def end(self):
        """Clean up visualization resources."""
        if self.enable_vis:
            plt.close('all')

    def _init_visualization_cache(self):
        """Initialize pre-computed arrays for efficient boundary visualization."""
        if not (self.enable_vis or self.enable_vis_rollout):
            return
        
        # Pre-compute time arrays for visualization
        self.vis_time_array = torch.linspace(
            -self.vis_time_window, 
            self.vis_time_window, 
            self.bd_vis_sample, 
            device=self.device
        )
        
        # Pre-compute harmonic indices for boundary computation
        num_harmonics = len(self.bound_params[0]['coef']) // 2
        self.vis_harmonics = torch.arange(1, num_harmonics + 1, device=self.device, dtype=torch.float32)
        
        # Cache for boundary computation to avoid repeated calculations
        self.boundary_cache = {
            'last_update_time': torch.full((self.num_main_envs,), -1.0, device=self.device),
            'cached_boundaries': torch.zeros((self.num_main_envs, self.bd_vis_sample, 2), device=self.device)  # 2 for [lower, upper]
        }


# Factory functions to create different obsavoid batch environments
def create_sine_bound_batch_env(num_main_envs: int = 4, num_rollout_per_main: int = 16, **kwargs):
    """Create batch environment with sine boundary functions."""
    env = ObsAvoidBatchEnv(num_main_envs, num_rollout_per_main, **kwargs)

    # Override boundary parameters for sine functions
    for i in range(env.total_num_envs):
        params = {
            'initial_y': 0.0,
            'initial_v': 0.0,
            'p': 10000.0,
            'd': 200.0,
            'slope': 0.0,
            'coef': [0.5, 0.0] + [0.0] * 30,  # Only sine component
            'width': 0.6
        }
        env.bound_params[i] = params
        env.y[i] = 0.0
        env.v[i] = 0.0
        env.p[i] = 10000.0
        env.d[i] = 200.0

    return env


def create_randpath_bound_batch_env(num_main_envs: int = 4, num_rollout_per_main: int = 16, **kwargs):
    """Create batch environment with random path boundary functions."""
    return ObsAvoidBatchEnv(num_main_envs, num_rollout_per_main, **kwargs)
