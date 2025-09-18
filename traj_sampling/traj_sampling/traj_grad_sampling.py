"""
Trajectory gradient sampling and optimization module.

This module provides trajectory optimization capabilities without dependencies
on legged_gym, making it compatible with Python 3.10 and usable across
different projects.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from torch.nn import functional as F

from .trajopt_policy import (
    TrajOptPolicyBase,
    TrajOptPolicySampling,
    TrajOptPolicyTF,
    TrajOptMode,
    create_sampling_policy,
    create_transformer_policy,
    create_policy_wrapper
)
from .spline import UniBSpline, CatmullRomSpline, InterpolatedSpline, LinearSpline
from .utils.benchmark import do_cprofile, time_profile, gpu_profile, benchmark
from .config.trajectory_optimization_config import TrajectoryOptimizationCfg


# Import for RL policy loading
from rsl_rl.modules import ActorCritic
from rsl_rl.modules import ActorCriticRecurrent


class TrajGradSampling:
    """Trajectory gradient sampling and optimization module.

    This class provides trajectory optimization capabilities including:
    1. Storage and management of future trajectories
    2. Optimization of trajectories using sampling-based methods
    3. Evaluation of gradients for trajectory optimization
    4. Optional RL policy warmstart for trajectory initialization
    """

    def __init__(self, cfg: TrajectoryOptimizationCfg, device, num_envs, num_actions, dt, main_env_indices):
        """Initialize trajectory gradient sampling module.

        Args:
            cfg: Configuration object for trajectory optimization (TrajectoryOptimizationCfg)
            device: Device for computations
            num_envs: Total number of environments (not rollout environments)
            num_actions: Number of action dimensions
            dt: Environment timestep
            main_env_indices: Indices of main environments
        """
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.dt = dt
        self.main_env_indices = main_env_indices

        # Initialize RL policy for warmstart if enabled
        self.rl_policy = None
        self.use_rl_warmstart = False
        self.obs_mean = None
        self.obs_var = None

        if hasattr(self.cfg, 'rl_warmstart') and getattr(self.cfg.rl_warmstart, "enable", False):
            self.use_rl_warmstart = True
            self.rl_traj_initialized = False

        # Initialize data collection components
        self.data_collector = None
        self.trainer = None
        self.enable_data_collection = False

        # Initialize trajectory optimization if enabled
        if hasattr(self.cfg, 'trajectory_opt') and getattr(self.cfg.trajectory_opt, "enable_traj_opt", False):
            self._init_traj_opt()

    # Initialization methods
    def init_rl_policy(self, num_obs, num_privileged_obs=None):
        """Initialize RL policy for trajectory warmstart.

        Args:
            num_obs: Number of observation dimensions
            num_privileged_obs: Number of privileged observation dimensions
        """
        if not self.use_rl_warmstart:
            return

        # Extract RL warmstart configuration
        rl_config = self.cfg.rl_warmstart
        device = rl_config.device
        checkpoint_path = rl_config.policy_checkpoint

        # Ensure checkpoint exists
        if not os.path.exists(checkpoint_path):
            # print(f"RL policy checkpoint not found at {checkpoint_path}, disabling RL warmstart")
            raise FileNotFoundError(f"RL policy checkpoint not found at {checkpoint_path}, disabling RL warmstart")
            return

        try:
            print(f"Loading RL policy from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Create actor network based on configuration
            if rl_config.actor_network == "lstm":
                # For recurrent policy
                actor_critic = ActorCriticRecurrent(
                    num_obs,
                    num_privileged_obs if num_privileged_obs else num_obs,
                    self.num_actions,
                    actor_hidden_dims=rl_config.actor_hidden_dims,
                    critic_hidden_dims=rl_config.critic_hidden_dims,
                    activation=rl_config.activation,
                ).to(device)
            else:
                # For MLP policy
                actor_critic = ActorCritic(
                    num_obs,
                    num_privileged_obs if num_privileged_obs else num_obs,
                    self.num_actions,
                    actor_hidden_dims=rl_config.actor_hidden_dims,
                    critic_hidden_dims=rl_config.critic_hidden_dims,
                    activation=rl_config.activation,
                ).to(device)

            # Load state dict from checkpoint
            if "model_state_dict" in checkpoint:
                actor_critic.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded model from 'model_state_dict'")
            elif "actor_state_dict" in checkpoint:
                actor_critic.actor.load_state_dict(checkpoint["actor_state_dict"])
                print("Loaded model from 'actor_state_dict'")
            else:
                # print("Unsupported checkpoint format, disabling RL warmstart")
                raise ValueError("Unsupported checkpoint format, disabling RL warmstart")
                return

            # Set to evaluation mode
            actor_critic.eval()
            self.rl_policy = actor_critic
            print("RL policy loaded successfully")

            # Set standardization parameters if available
            if rl_config.standardize_obs and "obs_mean" in checkpoint and "obs_var" in checkpoint:
                self.obs_mean = checkpoint["obs_mean"].to(device)
                self.obs_var = checkpoint["obs_var"].to(device)
                print("Loaded observation standardization parameters from checkpoint")
            else:
                self.obs_mean = None
                self.obs_var = None
                print("No observation standardization parameters found")

        except Exception as e:
            print(f"Error loading RL policy: {e}")
            import traceback
            traceback.print_exc()
            self.rl_policy = None

    def init_trajectories_from_rl(self, rollout_callback):
        """Initialize trajectories by rolling out the RL policy.

        Args:
            rollout_callback: Function to perform rollout with RL policy
        """
        if self.rl_policy is None:
            print("No RL policy available for trajectory initialization")
            return

        print("Initializing trajectories using RL policy rollout")

        # Use the callback to perform rollout and get trajectories
        traj_actions = rollout_callback(self.rl_policy, self.obs_mean, self.obs_var)

        # Convert dense action trajectories to node trajectories
        self.node_trajectories = self.u2node_batch(traj_actions)
        self.action_trajectories = traj_actions
        self.rl_traj_initialized = True
        print("RL policy trajectory initialization complete")

    def _init_traj_opt(self):
        """Initialize trajectory optimization components."""
        # Extract config parameters
        self.horizon_samples = self.cfg.trajectory_opt.horizon_samples
        self.horizon_nodes = self.cfg.trajectory_opt.horizon_nodes
        self.num_samples = self.cfg.trajectory_opt.num_samples
        self.num_diffuse_steps = self.cfg.trajectory_opt.num_diffuse_steps
        self.num_diffuse_steps_init = self.cfg.trajectory_opt.num_diffuse_steps_init
        self.temp_sample = self.cfg.trajectory_opt.temp_sample
        self.horizon_diffuse_factor = self.cfg.trajectory_opt.horizon_diffuse_factor
        self.traj_diffuse_factor = self.cfg.trajectory_opt.traj_diffuse_factor
        self.gamma = self.cfg.trajectory_opt.gamma
        self.update_method = self.cfg.trajectory_opt.update_method
        self.interp_method = self.cfg.trajectory_opt.interp_method
        self.noise_scaling = self.cfg.trajectory_opt.noise_scaling

        # Initialize trajectories for each main environment
        self.action_size = self.num_actions
        self.node_trajectories = torch.zeros(
            (self.num_envs, self.horizon_nodes + 1, self.action_size),
            device=self.device
        )

        # For storing dense trajectories (interpolated from nodes)
        self.action_trajectories = torch.zeros(
            (self.num_envs, self.horizon_samples + 1, self.action_size),
            device=self.device
        )

        # For storing predicted states along trajectories
        if self.cfg.trajectory_opt.compute_predictions:
            self.predicted_states = {
                'q': torch.zeros((self.num_envs, self.horizon_samples + 1, 12), device=self.device),  # Assuming 12 DOF
                'qd': torch.zeros((self.num_envs, self.horizon_samples + 1, 12), device=self.device),
                'pos': torch.zeros((self.num_envs, self.horizon_samples + 1, 3), device=self.device),
                'rewards': torch.zeros((self.num_envs), device=self.device)
            }

        # Initialize noise schedule for diffusion
        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = torch.log(torch.tensor(sigma1 / sigma0)) / self.num_diffuse_steps
        # Ensure sigmas has length equal to max(num_diffuse_steps, num_diffuse_steps_init)
        max_diffuse_steps = max(self.num_diffuse_steps, self.num_diffuse_steps_init)
        self.sigmas = A * torch.exp(B * torch.arange(max_diffuse_steps, device=self.device))

        # Create reversed range for sigma_control
        self.sigma_control = torch.flip(
            self.horizon_diffuse_factor ** torch.arange(self.horizon_nodes + 1, device=self.device),
            dims=[0]
        ) * self.noise_scaling
        print(f"Sigma control initialized: {self.sigma_control}")

        # Initialize time steps for interpolation
        self.ctrl_dt = self.dt
        self.step_us = torch.linspace(0, self.ctrl_dt * self.horizon_samples, self.horizon_samples + 1, device=self.device)
        self.step_nodes = torch.linspace(0, self.ctrl_dt * self.horizon_samples, self.horizon_nodes + 1, device=self.device)
        self.node_dt = self.ctrl_dt * (self.horizon_samples) / (self.horizon_nodes)

        # Initialize spline interpolation
        self._init_spline_interpolation()

        # Initialize trajectory optimization policy
        self._init_trajopt_policy()

        print(f"Trajectory optimization initialized with {self.horizon_samples} horizon samples, {self.horizon_nodes} nodes")

    def _init_spline_interpolation(self):
        """Initialize spline interpolation based on configuration."""
        if self.interp_method == "spline":
            # Use Catmull-Rom spline interpolation
            self.spline_interpolator = CatmullRomSpline(
                horizon_nodes=self.horizon_nodes + 1,
                horizon_samples=self.horizon_samples + 1,
                dt=self.ctrl_dt,
                device=self.device
            )
            print(f"Using CatmullRomSpline interpolation for trajectory conversion")
        else:
            # Use LinearSpline as fallback instead of inline linear interpolation
            self.spline_interpolator = LinearSpline(
                horizon_nodes=self.horizon_nodes + 1,
                horizon_samples=self.horizon_samples + 1,
                dt=self.ctrl_dt,
                device=self.device
            )
            print(f"Using LinearSpline interpolation for trajectory conversion")

    def _init_trajopt_policy(self):
        """Initialize trajectory optimization policy based on configuration."""
        traj_opt_cfg = self.cfg.trajectory_opt

        # Get policy type and mode from config
        policy_type = getattr(traj_opt_cfg, 'policy_type', 'sampling')
        policy_mode = getattr(traj_opt_cfg, 'policy_mode', 'traj')

        # Convert string mode to enum
        mode_enum = TrajOptMode.TRAJ if policy_mode == 'traj' else TrajOptMode.DELTA_TRAJ

        if policy_type == 'transformer':
            # Initialize transformer policy
            self.trajopt_policy = create_transformer_policy(
                horizon_nodes=self.horizon_nodes+1,
                action_dim=self.action_size,
                obs_dim=None,  # Would need to be configured if using observations
                mode=mode_enum,
                device=self.device,
                config=self.cfg.trajectory_opt  # Pass config for noise scheduler
            )
            print(f"Using Transformer trajectory optimization policy in {policy_mode} mode")

        elif policy_type == 'sampling':
            # Get noise sampler configuration from config
            noise_sampler_type = getattr(traj_opt_cfg, 'noise_sampler_type', 'mc')
            noise_distribution = getattr(traj_opt_cfg, 'noise_distribution', 'normal')
            noise_sampler_seed = getattr(traj_opt_cfg, 'noise_sampler_seed', None)
            
            # Initialize sampling policy with existing parameters and noise sampler config
            self.trajopt_policy = create_sampling_policy(
                horizon_nodes=self.horizon_nodes+1,
                horizon_samples=self.horizon_samples+1,
                action_dim=self.action_size,
                num_samples=self.num_samples,
                num_diffuse_steps=self.num_diffuse_steps,
                temp_sample=self.temp_sample,
                noise_scaling=self.noise_scaling,
                traj_diffuse_factor=self.traj_diffuse_factor,
                horizon_diffuse_factor=self.horizon_diffuse_factor,
                gamma=self.gamma,
                update_method=self.update_method,
                dt=self.ctrl_dt,
                mode=mode_enum,
                device=self.device,
                noise_sampler_type=noise_sampler_type,
                noise_distribution=noise_distribution,
                noise_sampler_seed=noise_sampler_seed,
                config=self.cfg.trajectory_opt  # Pass config for noise scheduler
            )
            print(f"Using Sampling trajectory optimization policy ({self.update_method}) in {policy_mode} mode")
            
            if noise_sampler_type is not None:
                print(f"Noise sampler: {noise_sampler_type} ({noise_distribution})")
                if noise_sampler_seed is not None:
                    print(f"Noise sampler seed: {noise_sampler_seed}")
            else:
                print("Noise sampler: fallback torch.randn")
            
            # Print noise scheduler info if available
            if self.trajopt_policy.noise_scheduler is not None:
                print(f"Noise scheduler: {type(self.trajopt_policy.noise_scheduler).__name__}")
            else:
                print("Using fallback noise scheduling")

        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        # Wrap the policy for easy switching
        self.trajopt_policy_wrapper = create_policy_wrapper(self.trajopt_policy)

    # Conversion methods for control sequences
    def node2u(self, nodes: torch.Tensor) -> torch.Tensor:
        """Convert control nodes to dense control sequence using interpolation.

        Args:
            nodes: Control nodes [Hnode+1, action_dim]

        Returns:
            Dense control sequence [Hsample+1, action_dim]
        """
        return self.spline_interpolator.node2dense(nodes)

    @time_profile("TrajGradSampling.node2u_batch")
    @gpu_profile("TrajGradSampling.node2u_batch")
    def node2u_batch(self, nodes_batch: torch.Tensor) -> torch.Tensor:
        """Convert multiple control nodes to dense control sequences at once.

        Args:
            nodes_batch: Batch of control nodes [batch_size, Hnode+1, action_dim]

        Returns:
            Batch of dense control sequences [batch_size, Hsample+1, action_dim]
        """
        return self.spline_interpolator.node2dense(nodes_batch)

    def u2node(self, us: torch.Tensor) -> torch.Tensor:
        """Convert dense control sequence to control nodes using interpolation.

        Args:
            us: Dense control sequence [Hsample+1, action_dim]

        Returns:
            Control nodes [Hnode+1, action_dim]
        """
        return self.spline_interpolator.dense2node(us)

    def u2node_batch(self, us_batch: torch.Tensor) -> torch.Tensor:
        """Convert multiple dense control sequences to control nodes at once.

        Args:
            us_batch: Batch of dense control sequences [batch_size, Hsample+1, action_dim]

        Returns:
            Batch of control nodes [batch_size, Hnode+1, action_dim]
        """
        return self.spline_interpolator.dense2node(us_batch)

    # Shift and optimization methods
    def _shift_nodetraj_batch(self, trajs: torch.Tensor, n_steps: int = 1,
                             policy_obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shift multiple trajectories by n time steps in a batch.

        Args:
            trajs: Trajectories to shift [batch_size, length, action_dim]
            n_steps: Number of steps to shift by
            policy_obs: Optional observations for RL policy inference

        Returns:
            Shifted trajectories [batch_size, length, action_dim]
        """
        # Convert to dense control sequences first
        u_batch = self.node2u_batch(trajs)

        # Shift all dense controls by n steps
        u_batch = torch.roll(u_batch, -n_steps, dims=1)

        # If RL policy is enabled and configured to be used for appending
        if (self.use_rl_warmstart and
            self.cfg.rl_warmstart.use_for_append and
            self.rl_policy is not None and
            self.rl_traj_initialized and
                policy_obs is not None):

            # Standardize observations if needed
            if self.obs_mean is not None and self.obs_var is not None:
                policy_obs = (policy_obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)

            with torch.no_grad():
                # Get actions from policy using act_inference method
                actions = self.rl_policy.act_inference(policy_obs)

            # Fill the last n_steps controls with policy actions
            for i in range(n_steps):
                if i < u_batch.shape[1]:
                    u_batch[:, -n_steps + i, :] = actions
        else:
            # Fill the last n_steps controls with last known controls
            u_batch[:, -n_steps:, :] = torch.zeros_like(u_batch[:, -n_steps:, :])
            # BUG: this affects the optimization performance
            # u_batch[:, -n_steps:, :] = u_batch[:, -n_steps-1, :].unsqueeze(1)

        # Convert back to nodes
        shifted = self.u2node_batch(u_batch)

        return shifted

    def shift_trajectory_batch(self, policy_obs: Optional[torch.Tensor] = None) -> None:
        """Update the node trajectories for all environments based on new actions.

        Args:
            policy_obs: Optional observations for RL policy inference
        """
        # Shift all node trajectories in batch
        self.node_trajectories = self._shift_nodetraj_batch(self.node_trajectories, 1, policy_obs)

        # Update the dense trajectories by interpolation
        self.action_trajectories = self.node2u_batch(self.node_trajectories)

    @time_profile("TrajGradSampling.optimize_all_trajectories")
    @gpu_profile("TrajGradSampling.optimize_all_trajectories")
    @do_cprofile("results/optimize_all_trajectories.prof")
    def optimize_all_trajectories(self, rollout_callback, n_diffuse: Optional[int] = None,
                                  initial: bool = False, obs: Optional[torch.Tensor] = None) -> None:
        """Optimize trajectories for all main environments in batch.

        Args:
            rollout_callback: Function to perform batch rollout
            n_diffuse: Optional number of diffusion steps; if None, uses default
            initial: Whether this is the initial optimization (uses more diffusion steps)
            obs: Optional observations for transformer policy
        """
        # Get current trajectories for all main environments
        curr_trajs = self.node_trajectories.clone()

        # Determine number of diffusion steps
        if n_diffuse is None:
            n_diffuse = self.num_diffuse_steps_init if initial else self.num_diffuse_steps

        # Use the trajectory optimization policy for optimization
        self.node_trajectories = self.trajopt_policy.optimize_trajectories(
            curr_trajs, rollout_callback, n_diffuse, obs=obs
        )

        self.action_trajectories = self.node2u_batch(self.node_trajectories)

    # Policy management methods
    def set_trajopt_policy(self, policy: TrajOptPolicyBase):
        """Set a new trajectory optimization policy.

        Args:
            policy: New trajectory optimization policy
        """
        if hasattr(self, 'trajopt_policy_wrapper'):
            self.trajopt_policy = policy
            self.trajopt_policy_wrapper.set_policy(policy)
        else:
            self.trajopt_policy = policy
            self.trajopt_policy_wrapper = create_policy_wrapper(policy)

        print(f"Trajectory optimization policy changed to: {type(policy).__name__}")

    def switch_to_transformer_policy(self, obs_dim: int, **kwargs):
        """Switch to transformer-based trajectory optimization policy.

        Args:
            obs_dim: Observation dimension
            **kwargs: Additional arguments for transformer policy
        """
        transformer_policy = create_transformer_policy(
            horizon_nodes=self.horizon_nodes,
            action_dim=self.action_size,
            obs_dim=obs_dim,
            device=self.device,
            config=self.cfg.trajectory_opt,  # Pass config for noise scheduler
            **kwargs
        )
        self.set_trajopt_policy(transformer_policy)

    def switch_to_sampling_policy(self, **kwargs):
        """Switch to sampling-based trajectory optimization policy.

        Args:
            **kwargs: Additional arguments for sampling policy
        """
        # Get noise sampler configuration from config
        traj_opt_cfg = self.cfg.trajectory_opt
        noise_sampler_type = getattr(traj_opt_cfg, 'noise_sampler_type', 'mc')
        noise_distribution = getattr(traj_opt_cfg, 'noise_distribution', 'normal')
        noise_sampler_seed = getattr(traj_opt_cfg, 'noise_sampler_seed', None)
        
        # Merge config values with any provided kwargs
        sampling_kwargs = {
            'noise_sampler_type': noise_sampler_type,
            'noise_distribution': noise_distribution,
            'noise_sampler_seed': noise_sampler_seed,
            **kwargs  # Allow overriding config values
        }
        
        sampling_policy = create_sampling_policy(
            horizon_nodes=self.horizon_nodes,
            action_dim=self.action_size,
            device=self.device,
            config=self.cfg.trajectory_opt,  # Pass config for noise scheduler
            **sampling_kwargs
        )
        self.set_trajopt_policy(sampling_policy)

    # Data collection and training methods
    def enable_data_collect(self, 
                             mode: TrajOptMode = TrajOptMode.DELTA_TRAJ,
                             max_samples: int = 10000):
        """Enable data collection for imitation learning.
        
        Args:
            mode: Target mode for data collection (TRAJ or DELTA_TRAJ)
            max_samples: Maximum number of samples to collect
        """
        from .trajopt_policy import TrajOptDataCollector
        
        self.data_collector = TrajOptDataCollector(mode, max_samples, self.device)
        self.enable_data_collection = True
        print(f"Data collection enabled in {mode.value} mode with max {max_samples} samples")

    def collect_optimization_data(self, 
                                input_traj: torch.Tensor,
                                output_traj: torch.Tensor,
                                obs: Optional[torch.Tensor] = None,
                                meta: Optional[Dict[str, Any]] = None):
        """Collect trajectory optimization data for training.
        
        Args:
            input_traj: Input trajectory before optimization
            output_traj: Output trajectory after optimization
            obs: Optional observations
            meta: Optional metadata
        """
        if not self.enable_data_collection or self.data_collector is None:
            return
            
        self.data_collector.collect_sample(input_traj, output_traj, obs, meta)

    @time_profile("TrajGradSampling.optimize_and_collect_data")
    @gpu_profile("TrajGradSampling.optimize_and_collect_data")
    def optimize_and_collect_data(self, 
                                rollout_callback,
                                obs: Optional[torch.Tensor] = None,
                                n_diffuse: Optional[int] = None,
                                initial: bool = False) -> None:
        """Optimize trajectories and collect data for training.
        
        Args:
            rollout_callback: Function to perform batch rollout
            obs: Optional observations for data collection
            n_diffuse: Optional number of diffusion steps
            initial: Whether this is the initial optimization
        """
        # Store input trajectories before optimization
        input_trajs = self.node_trajectories.clone()
        
        # Perform optimization
        self.optimize_all_trajectories(rollout_callback, n_diffuse, initial)
        
        # Collect data if enabled
        if self.enable_data_collection and self.data_collector is not None:
            output_trajs = self.node_trajectories.clone()
            
            # Create metadata
            meta = {
                'n_diffuse': n_diffuse if n_diffuse is not None else (self.num_diffuse_steps_init if initial else self.num_diffuse_steps),
                'initial': initial,
                'optimization_method': self.update_method
            }
            
            self.collect_optimization_data(input_trajs, output_trajs, obs, meta)

    def get_collected_dataset(self) -> Optional[Dict[str, Any]]:
        """Get the collected dataset for training.
        
        Returns:
            Dataset dictionary if data collection is enabled, None otherwise
        """
        if not self.enable_data_collection or self.data_collector is None:
            return None
            
        return self.data_collector.get_dataset()

    def save_collected_dataset(self, filepath: str) -> None:
        """Save the collected dataset to file.
        
        Args:
            filepath: Path to save the dataset
        """
        if not self.enable_data_collection or self.data_collector is None:
            print("No data collector available to save dataset")
            return
            
        self.data_collector.save_dataset(filepath)

    def clear_collected_data(self) -> None:
        """Clear all collected data."""
        if self.data_collector is not None:
            self.data_collector.clear()

    def setup_transformer_training(self,
                                 obs_dim: Optional[int] = None,
                                 learning_rate: float = 1e-4,
                                 weight_decay: float = 1e-5,
                                 **transformer_kwargs):
        """Setup training for a transformer policy using collected data.
        
        Args:
            obs_dim: Observation dimension (if using observations)
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            **transformer_kwargs: Additional arguments for transformer policy
            
        Returns:
            Configured trainer instance
        """
        from .trajopt_policy import TrajOptTrainer
        
        # Create transformer policy to train
        mode = self.data_collector.mode if self.data_collector else TrajOptMode.DELTA_TRAJ
        
        # The transformer should expect the same horizon_nodes that are used in data collection
        # which is horizon_nodes + 1 in the trajectory optimization setup
        transformer_horizon_nodes = self.horizon_nodes + 1
        
        student_policy = create_transformer_policy(
            horizon_nodes=transformer_horizon_nodes,
            action_dim=self.action_size,
            obs_dim=obs_dim,
            mode=mode,
            device=self.device,
            config=self.cfg.trajectory_opt,  # Pass config for noise scheduler
            **transformer_kwargs
        )
        
        # Ensure the policy is on the correct device
        student_policy.to(self.device)
        
        # Create trainer
        self.trainer = TrajOptTrainer(
            student_policy=student_policy,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=self.device
        )
        
        print(f"Transformer training setup complete with mode: {mode.value}")
        print(f"Transformer horizon nodes: {transformer_horizon_nodes} (original: {self.horizon_nodes})")
        print(f"Model device: {next(student_policy.parameters()).device}")
        return self.trainer

    def train_transformer_on_data(self,
                                dataset: Optional[Dict[str, Any]] = None,
                                num_epochs: int = 100,
                                batch_size: int = 32,
                                validation_split: float = 0.2,
                                print_interval: int = 10) -> Dict[str, List[float]]:
        """Train transformer policy on collected data.
        
        Args:
            dataset: Dataset to train on (if None, uses collected data)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            print_interval: Interval for printing progress
            
        Returns:
            Training metrics dictionary
        """
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_transformer_training() first.")
            
        # Use provided dataset or collected data
        if dataset is None:
            dataset = self.get_collected_dataset()
            if dataset is None:
                raise ValueError("No dataset provided and no data collected")
        
        # Train the model
        metrics = self.trainer.train_on_dataset(
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            print_interval=print_interval
        )
        
        return metrics

    def deploy_trained_transformer(self) -> None:
        """Deploy the trained transformer policy for trajectory optimization.
        
        This replaces the current sampling policy with the trained transformer.
        """
        if self.trainer is None:
            raise ValueError("No trained transformer available")
            
        # Set the trained transformer as the active policy
        self.set_trajopt_policy(self.trainer.student_policy)
        print("Trained transformer policy deployed for trajectory optimization")

    def save_training_checkpoint(self, filepath: str, epoch: int, metadata: Optional[Dict] = None) -> None:
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            metadata: Optional metadata to save
        """
        if self.trainer is None:
            raise ValueError("No trainer available to save checkpoint")
            
        self.trainer.save_checkpoint(filepath, epoch, metadata)

    def load_training_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
            
        Returns:
            Checkpoint metadata
        """
        if self.trainer is None:
            raise ValueError("No trainer available to load checkpoint")
            
        return self.trainer.load_checkpoint(filepath)
