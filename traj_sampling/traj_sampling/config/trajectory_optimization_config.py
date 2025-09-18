"""Configuration classes for trajectory optimization in legged gym environments.

This module provides decoupled configuration classes for trajectory optimization
that can be used independently of specific environment configurations.
"""

from legged_gym import LEGGED_GYM_ROOT_DIR

class TrajectoryOptimizationCfg:
    """Configuration for trajectory optimization parameters."""
    
    class trajectory_opt:
        # Enable trajectory optimization
        enable_traj_opt = True
        
        # Diffusion parameters
        num_diffuse_steps = 1  # Number of diffusion steps (Ndiffuse)
        num_diffuse_steps_init = 6  # Number of initial diffusion steps (Ndiffuse_init)
        num_samples = 127  # NOTE: overloaded by LeggedGymEnvRunner2. Number of samples to generate at each diffusion step (Nsample)
        # Traj parameters
        horizon_samples = 16  # Horizon length in samples (Hsample)
        horizon_nodes = 4  # Number of control nodes within horizon (Hnode)
        interp_method = "spline"  # Options: ["linear", "spline"]

        # Update method
        update_method = "avwbfo"  # Update method, options: ["mppi", "wbfo", "avwbfo"]
        gamma = 1.000  # Discount factor for rewards in avwbfo
        temp_sample = 0.1  # Temperature parameter for softmax weighting

        # Noise scheduling
        horizon_diffuse_factor = 0.9  # How much more noise to add for further horizon
        traj_diffuse_factor = 0.5  # Diffusion factor for trajectory
        noise_scaling = 1.5
        # New noise scheduler (overrides the above if provided)
        noise_scheduler_type = "s2"  # Options: ["s2", "s3", "constant", "linear_decay", "exponential_decay", "cosine_decay", "hierarchical", "adaptive"]
        noise_shape_fn = "linear"  # Options: ["sine", "linear", "quadratic", "exponential", "constant"]
        noise_decay_fn = "exponential"  # Options: ["constant", "linear", "exponential", "cosine"]
        noise_base_scale = 1.0  # Base scaling factor for noise
        noise_dim_scale = None  # Per-dimension scaling factors [action_dim] or scalar (for S3 scheduler only)
        noise_decay_kwargs = {}  # Additional arguments for decay function (e.g., {"decay_rate": 0.9, "final_ratio": 0.1})
        
        # Hierarchical noise scheduler specific parameters
        noise_hierarchical_activate_start = None  # Per-dimension activation start points [action_dim] or scalar
        noise_hierarchical_activate_len = None   # Per-dimension activation lengths [action_dim] or scalar
        noise_hierarchical_activation_pattern = None  # Predefined patterns: ["staggered", "early_late", "overlapping"]
        
        # Noise sampler configuration
        noise_sampler_type = "lhs"  # Options: ["mc", "lhs", "halton", None] - None uses fallback torch.randn
        noise_distribution = "normal"  # Options: ["normal", "uniform"]
        noise_sampler_seed = None  # Random seed for noise sampler (None for random)
        
        # policy config
        policy_type = "sampling"  # "sampling" or "transformer"
        policy_mode = "traj"  # "traj" or "delta_traj"
        
        # Whether to compute and store predicted trajectories
        compute_predictions = False

    class rl_warmstart:
        enable = True
        policy_checkpoint = ""  # Path to policy checkpoint
        actor_network = "mlp"  # options: ["mlp", "lstm"]
        device = "cuda:0"
        
        # Network architecture settings
        actor_hidden_dims = [128, 64, 32]    # Hidden dimensions for actor network
        critic_hidden_dims = [128, 64, 32]   # Hidden dimensions for critic network
        activation = 'elu'                   # Activation function: elu, relu, selu, etc.
        
        # Whether to use RL policy for appending new actions during shift
        use_for_append = True
        
        # Whether to standardize observations for policy input
        standardize_obs = True
        
        # Input type for the policy
        obs_type = "non_privileged"  # options: ["privileged", "non_privileged"]
    
    class env:
        num_actions = 18  # Default for ElSpider Air
    
    # Device configuration
    sim_device = "cuda:0"


class ElSpiderAirTrajectoryOptCfg(TrajectoryOptimizationCfg):
    """ElSpider Air specific trajectory optimization configuration."""
    
    class trajectory_opt(TrajectoryOptimizationCfg.trajectory_opt):
        # ElSpider Air specific parameters
        num_diffuse_steps = 1
        horizon_samples = 16
        horizon_nodes = 4
        num_samples = 127
        noise_scaling = 1.3
        
        # ElSpider Air specific noise scheduling
        noise_scheduler_type = None  # Use S2 scheduler to maintain current behavior
        noise_shape_fn = "linear"
        noise_decay_fn = "exponential"
        noise_base_scale = 1.5  # Match original noise_scaling
        noise_decay_kwargs = {"decay_rate": 0.9}
        
        # Hierarchical scheduling for 18-DOF ElSpider (6 legs * 3 DOF each)
        # Stagger activation across leg groups for coordinated movement
        # noise_hierarchical_activation_pattern = "staggered"
        # noise_dim_scale = [1.2] * 6 + [1.0] * 6 + [0.8] * 6  # Higher priority for first 6 DOF, lower for last 6
        
        # ElSpider Air specific noise sampler
        noise_sampler_type = "lhs"  # Use Latin Hypercube for better exploration
        noise_distribution = "normal"
        noise_sampler_seed = 42  # Fixed seed for reproducibility
        
    class rl_warmstart(TrajectoryOptimizationCfg.rl_warmstart):
        # ElSpider Air specific RL policy settings
        enable = True
        policy_checkpoint = f"{LEGGED_GYM_ROOT_DIR}/ckpt/elspider_air/plane_walk_300.pt"
        # policy_checkpoint = "/home/user/CodeSpace/Python/PredictiveDiffusionPlanner_Dev/legged_gym_cmp/legged_gym/logs/rough_elspider_air/Aug09_16-24-07_/model_300.pt"
        actor_network = "mlp"
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'
        use_for_append = True
        standardize_obs = True
        obs_type = "non_privileged"
    
    class env(TrajectoryOptimizationCfg.env):
        num_actions = 18  # ElSpider Air has 18 actions


class AnymalCTrajectoryOptCfg(TrajectoryOptimizationCfg):
    """Anymal C specific trajectory optimization configuration."""
    
    class trajectory_opt(TrajectoryOptimizationCfg.trajectory_opt):
        # Anymal C specific parameters
        horizon_samples = 24
        horizon_nodes = 6
        num_samples = 127
        noise_scaling = 1.5
        interp_method = "spline"  # Options: ["linear", "spline"]

        # New noise scheduler (overrides the above if provided)
        noise_scheduler_type = "hierarchical"  # Options: ["s2", "s3", "constant", "linear_decay", "exponential_decay", "cosine_decay", "hierarchical", "adaptive"]
        noise_shape_fn = "linear"  # Options: ["sine", "linear", "quadratic", "exponential", "constant"]
        noise_decay_fn = "exponential"  # Options: ["constant", "linear", "exponential", "cosine"]
        noise_base_scale = 1.0  # Base scaling factor for noise
        noise_dim_scale = None  # Per-dimension scaling factors [action_dim] or scalar (for S3 scheduler only)
        noise_decay_kwargs = {}  # Additional arguments for decay function (e.g., {"decay_rate": 0.9, "final_ratio": 0.1})

        # Hierarchical scheduling for 12-DOF ElSpider (4 legs * 3 DOF each)
        # Stagger activation across leg groups for coordinated movement
        noise_hierarchical_activation_pattern = "staggered"
        noise_dim_scale = [1.2] * 4 + [1.0] * 4 + [0.8] * 4  # Higher priority for first 6 DOF, lower for last 6

        # Update method
        update_method = "avwbfo"  # Update method, options: ["mppi", "wbfo", "avwbfo"]
        gamma = 1.000  # Discount factor for rewards in avwbfo
        temp_sample = 0.1  # Temperature parameter for softmax weighting

    
    class rl_warmstart(TrajectoryOptimizationCfg.rl_warmstart):
        # Anymal C specific RL policy settings
        enable = True
        policy_checkpoint = f"{LEGGED_GYM_ROOT_DIR}/ckpt/anymal_c/plane_walk_200.pt"
        actor_network = "mlp"
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'
        use_for_append = True
        standardize_obs = True
        obs_type = "non_privileged"
    
    class env(TrajectoryOptimizationCfg.env):
        num_actions = 12  # Anymal C has 12 actions


class Go2TrajectoryOptCfg(TrajectoryOptimizationCfg):
    """Go2 specific trajectory optimization configuration."""
    
    class trajectory_opt(TrajectoryOptimizationCfg.trajectory_opt):
        # Go2 specific parameters
        horizon_samples = 20
        horizon_nodes = 5
        num_samples = 127
        noise_scaling = 1.0
    
    class rl_warmstart(TrajectoryOptimizationCfg.rl_warmstart):
        # Go2 specific RL policy settings
        enable = True
        policy_checkpoint = ""  # To be set per experiment
        actor_network = "mlp"
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'
        use_for_append = True
        standardize_obs = True
        obs_type = "non_privileged"
    
    class env(TrajectoryOptimizationCfg.env):
        num_actions = 12  # Go2 has 12 actions


class CassieTrajectoryOptCfg(TrajectoryOptimizationCfg):
    """Cassie specific trajectory optimization configuration."""
    
    class trajectory_opt(TrajectoryOptimizationCfg.trajectory_opt):
        # Cassie specific parameters
        horizon_samples = 24
        horizon_nodes = 6
        num_samples = 127
        noise_scaling = 0.8
    
    class rl_warmstart(TrajectoryOptimizationCfg.rl_warmstart):
        # Cassie specific RL policy settings
        enable = True
        policy_checkpoint = ""  # To be set per experiment
        actor_network = "mlp"
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'
        use_for_append = True
        standardize_obs = True
        obs_type = "non_privileged"
    
    class env(TrajectoryOptimizationCfg.env):
        num_actions = 10  # Cassie has 10 actions


class FrankaTrajectoryOptCfg(TrajectoryOptimizationCfg):
    """Franka Panda robot arm specific trajectory optimization configuration."""
    
    class trajectory_opt(TrajectoryOptimizationCfg.trajectory_opt):
        num_diffuse_steps = 1
        # Franka specific parameters for manipulation tasks
        horizon_samples = 48  # Shorter horizon for manipulation tasks
        horizon_nodes = 8    # Fewer nodes for smoother arm motions
        num_samples = 127
        noise_scaling = 10.0 # Lower noise for precise manipulation
        temp_sample = 0.05

        # Franka specific hierarchical noise scheduling - end-effector first
        noise_scheduler_type = "hierarchical"  # None / s2 / hierarchical
        noise_shape_fn = "linear"
        noise_decay_fn = "exponential"
        noise_base_scale = 10.0  # Lower base scale for precise control
        noise_decay_kwargs = {"decay_rate": 0.5}  # Slower decay for smoother motion
        
        # Hierarchical scheduling for 7-DOF Franka (end-effector joints first, root joints last)
        # Joint order: [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3, wrist4]
        # Activation pattern: start from end-effector (wrist joints) and work backwards to shoulder
        noise_hierarchical_activate_start = [0.0, 0.0, 0.1, 0.2, 0.4, 0.4, 0.4]  # End joints activate first
        noise_hierarchical_activate_len = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]    # Longer activation for wrist precision
        noise_dim_scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Higher priority for end-effector joints
        
        # Franka specific noise sampler for precise manipulation
        noise_sampler_type = "lhs"  # Use Halton sequence for low-discrepancy sampling
        noise_distribution = "normal"
        noise_sampler_seed = 123  # Fixed seed for deterministic behavior
        
        # More conservative update method for manipulation
        update_method = "avwbfo"
        gamma = 1.0  # Slightly higher discount for longer horizon importance
        
        # Use spline interpolation for smooth trajectories
        interp_method = "linear"
    
    class rl_warmstart(TrajectoryOptimizationCfg.rl_warmstart):
        # Franka specific RL policy settings
        enable = False  # Disable RL warmstart by default for now
        policy_checkpoint = ""  # To be set per experiment
        actor_network = "mlp"
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'
        use_for_append = False  # Don't use RL for appending for now
        standardize_obs = True
        obs_type = "non_privileged"
    
    class env(TrajectoryOptimizationCfg.env):
        num_actions = 7  # Franka has 7 DOF (without gripper for trajectory optimization)