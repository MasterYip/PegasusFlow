"""
Test environments for trajectory optimization comparison.

This module provides benchmark environments for evaluating different trajectory
optimization methods including MPPI, WBFO, and AVWBFO. All environments support
batch computation using PyTorch for efficient evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import math


@dataclass
class EnvConfig:
    """Configuration for trajectory optimization environments."""
    horizon_samples: int = 64
    dt: float = 0.02
    max_episode_steps: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TrajOptEnvBase(ABC):
    """Abstract base class for trajectory optimization environments."""

    def __init__(self, config: EnvConfig):
        """Initialize the environment.

        Args:
            config: Environment configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.horizon_samples = config.horizon_samples
        self.dt = config.dt
        self.max_episode_steps = config.max_episode_steps

    @abstractmethod
    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """Reset the environment and return initial state.

        Args:
            batch_size: Number of parallel environments

        Returns:
            Initial state [batch_size, state_dim]
        """
        pass

    @abstractmethod
    def step(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take a step in the environment.

        Args:
            state: Current state [batch_size, state_dim]
            action: Action to take [batch_size, action_dim]

        Returns:
            Tuple of (next_state, reward, done)
        """
        pass

    @abstractmethod
    def batch_rollout(self, trajectories: torch.Tensor, initial_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform batch rollout of trajectories.

        Args:
            trajectories: Action trajectories [batch_size, horizon_samples, action_dim]
            initial_states: Initial states [batch_size, state_dim], if None uses current state

        Returns:
            Rewards for each trajectory [batch_size, horizon_samples]
        """
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        """Get state dimension."""
        pass

    @abstractmethod
    def get_action_dim(self) -> int:
        """Get action dimension."""
        pass

    @abstractmethod
    def visualize_trajectory(self, trajectory: torch.Tensor, title: str = "") -> plt.Figure:
        """Visualize a trajectory.

        Args:
            trajectory: Trajectory to visualize
            title: Plot title

        Returns:
            Matplotlib figure
        """
        pass


NavSceneGenMode = "preset2" # "random", "preset1"

class Navigation2DEnv(TrajOptEnvBase):
    """2D navigation trajectory optimization environment with random obstacles.
    
    This is a trajectory optimization task where trajectories represent direct
    position waypoints from start to goal while avoiding obstacles. Rewards are
    evaluated at each trajectory point (not time step).
    """

    def __init__(self,
                 config: EnvConfig,
                 workspace_size: Tuple[float, float] = (10.0, 10.0),
                 num_obstacles: int = 25,
                 obstacle_radius_range: Tuple[float, float] = (0.5, 1.5),
                 goal_radius: float = 0.5,
                 collision_penalty: float = -5.0,
                 goal_reward: float = 10.0,
                 smoothness_weight: float = 2.0,
                 efficiency_weight: float = 0.5,
                 mode: str = NavSceneGenMode):
        """Initialize 2D navigation trajectory optimization environment.

        Args:
            config: Environment configuration
            workspace_size: Size of the workspace (width, height)
            num_obstacles: Number of random obstacles
            obstacle_radius_range: Range of obstacle radii
            goal_radius: Radius of goal region
            collision_penalty: Penalty for collision with obstacles
            goal_reward: Reward for reaching goal
            smoothness_weight: Weight for trajectory smoothness reward
            efficiency_weight: Weight for efficiency (goal progress) reward
        """
        super().__init__(config)
        
        self.workspace_size = workspace_size
        self.num_obstacles = num_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.goal_radius = goal_radius
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.smoothness_weight = smoothness_weight
        self.efficiency_weight = efficiency_weight
        self.mode = mode
        # Generate random obstacles and goals
        self._generate_scenario()

    def _generate_scenario(self):
        """Generate random obstacles and start/goal positions."""
        # Generate obstacles
        self.obstacles = []
        if self.mode == "random":
            torch.manual_seed(4)  # For reproducibility
            for _ in range(self.num_obstacles):
                x = torch.rand(1) * self.workspace_size[0]
                y = torch.rand(1) * self.workspace_size[1]
                radius = (torch.rand(1) * (self.obstacle_radius_range[1] - self.obstacle_radius_range[0]) + 
                        self.obstacle_radius_range[0])
                self.obstacles.append((x.item(), y.item(), radius.item()))

            # Generate start and goal positions (ensure they're not in obstacles)
            self.start_pos = self._sample_free_position()
            self.goal_pos = self._sample_free_position()

            # Ensure start and goal are sufficiently far apart
            while torch.norm(self.goal_pos - self.start_pos) < 6.0:
                self.goal_pos = self._sample_free_position()
        elif self.mode == "preset1":
            # Replace simple preset with a denser maze-like obstacle layout.
            # Build walls from many small circular obstacles; leave gaps to create passages.
            self.obstacles = []
            wall_radius = 0.30
            spacing = 0.3  # spacing between obstacle centers along a wall

            def add_wall_vertical(x, y_start, y_end, gaps=None):
                gaps = gaps or []
                y = y_start
                while y <= y_end:
                    in_gap = any(g_s <= y <= g_e for (g_s, g_e) in gaps)
                    if not in_gap:
                        self.obstacles.append((float(x), float(y), float(wall_radius)))
                    y += spacing

            def add_wall_horizontal(y, x_start, x_end, gaps=None):
                gaps = gaps or []
                x = x_start
                while x <= x_end:
                    in_gap = any(g_s <= x <= g_e for (g_s, g_e) in gaps)
                    if not in_gap:
                        self.obstacles.append((float(x), float(y), float(wall_radius)))
                    x += spacing

            # Outer border walls with an opening near the start and near the goal
            add_wall_vertical(0.25, 0.25, self.workspace_size[1]-0.25, gaps=[(0.8, 1.4)])   # left border (leave start opening)
            add_wall_vertical(self.workspace_size[0]-0.25, 0.25, self.workspace_size[1]-0.25, gaps=[(8.4, 9.4)])  # right border (leave goal approach)
            add_wall_horizontal(0.25, 0.25, self.workspace_size[0]-0.25, gaps=[(0.8, 1.4)])  # bottom border (start opening)
            add_wall_horizontal(self.workspace_size[1]-0.25, 0.25, self.workspace_size[0]-0.25, gaps=[(8.4, 9.4)])  # top border (goal opening)

            # Internal vertical walls with purposeful gaps to form a maze path
            add_wall_vertical(2.8, 0.25, self.workspace_size[1]-0.25, gaps=[(0.1,1.0),(2.4,3.6),(5.4,6.4),(8.6,9.8)])
            add_wall_vertical(4.6, 0.25, self.workspace_size[1]-0.25, gaps=[(1.6,3.4),(4.0,5.0),(7.2,8.4)])
            add_wall_vertical(6.4, 0.25, self.workspace_size[1]-0.25, gaps=[(0.25,0.9),(3.0,3.8),(5.0,5.8),(7.0,7.8)])

            # Internal horizontal walls with gaps
            add_wall_horizontal(2.3, 0.25, self.workspace_size[0]-0.25, gaps=[(0.25,1.0),(2.4,3.2),(6.8,7.6),(8.6,9.4)])
            add_wall_horizontal(5.0, 0.25, self.workspace_size[0]-0.25, gaps=[(1.2,2.0),(7.4,8.2)])
            add_wall_horizontal(7.6, 0.25, self.workspace_size[0]-0.25, gaps=[(0.1,0.9),(3.6,4.4),(5.6,6.4),(7.0,8.8)])

            # Additional short walls / blocking obstacles to increase complexity
            # small block cluster near center-left
            # for xx in [1.8, 2.2, 2.6]:
            #     for yy in [4.0, 4.6]:
            #         self.obstacles.append((float(xx), float(yy), float(wall_radius)))

            # # small staggered obstacles to create narrow turns
            # for xx, yy in [(3.8,2.0), (3.8,2.6), (5.2,3.6), (5.2,4.2), (6.8,5.6), (6.8,6.2)]:
            #     self.obstacles.append((float(xx), float(yy), float(wall_radius)))

            # Start and goal remain at corners
            self.start_pos = torch.tensor([1.0, 1.0], device=self.device)
            self.goal_pos = torch.tensor([9.0, 9.0], device=self.device)
        elif self.mode == "preset2":
            # Preset obstacles
            self.obstacles = [
                (2.0, 2.0, 0.8),
                (2.0, 5.0, 0.8),
                (2.0, 8.0, 0.8),
                (5.0, 2.0, 0.8),
                (5.0, 8.0, 0.8),
                (8.0, 2.0, 0.8),
                (8.0, 5.0, 0.8),
                (8.0, 8.0, 0.8),
                (5.0, 5.0, 1.6),
            ]
            self.start_pos = torch.tensor([1.0, 1.0], device=self.device)
            self.goal_pos = torch.tensor([9.0, 9.0], device=self.device)
        else:
            raise ValueError(f"Unknown scene generation mode: {self.mode}")

    def _sample_free_position(self) -> torch.Tensor:
        """Sample a position that's not inside any obstacle."""
        max_attempts = 100
        for _ in range(max_attempts):
            pos = torch.rand(2, device=self.device) * torch.tensor(self.workspace_size, device=self.device)
            
            # Check if position is free
            collision = False
            for obs_x, obs_y, obs_r in self.obstacles:
                obs_pos = torch.tensor([obs_x, obs_y], device=self.device)
                if torch.norm(pos - obs_pos) < obs_r + 0.2:  # Add small margin
                    collision = True
                    break
            
            if not collision:
                return pos

        # If can't find free position, return a corner
        return torch.tensor([0.5, 0.5], device=self.device)

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """Reset the environment (not used in trajectory optimization)."""
        # Return start position as initial state
        initial_state = self.start_pos.unsqueeze(0).expand(batch_size, 2)
        return initial_state

    def step(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step function (not used in trajectory optimization)."""
        # This method is not used for trajectory optimization
        # Trajectories are evaluated directly in batch_rollout
        raise NotImplementedError("Use batch_rollout for trajectory optimization")

    def _compute_trajectory_point_rewards(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute rewards for trajectory positions.
        
        Args:
            positions: Trajectory positions [batch_size, horizon_samples, 2]
            
        Returns:
            Rewards for each position [batch_size, horizon_samples]
        """
        batch_size, horizon_samples, _ = positions.shape
        rewards = torch.zeros(batch_size, horizon_samples, device=self.device)
        
        for t in range(horizon_samples):
            current_positions = positions[:, t, :]  # [batch_size, 2]
            
            # 1. Distance to goal reward (negative distance)
            goal_dist = torch.norm(current_positions - self.goal_pos.unsqueeze(0), dim=1)
            goal_reward = -goal_dist * 0.1
            
            # 2. Goal reached bonus
            # goal_reached = goal_dist < self.goal_radius
            # goal_bonus = goal_reached.float() * self.goal_reward
            goal_bonus = torch.zeros(batch_size, device=self.device)
            
            # 3. Collision penalty
            collision = self._check_collisions(current_positions)
            collision_penalty = collision.float() * self.collision_penalty
            
            # 4. Smoothness reward (penalize large changes in direction)
            smoothness_reward = torch.zeros(batch_size, device=self.device)
            if t > 0:
                # Current segment vector
                prev_pos = positions[:, t-1, :]
                current_segment = current_positions - prev_pos
                
                if t > 1:
                    # Previous segment vector
                    prev_prev_pos = positions[:, t-2, :]
                    prev_segment = prev_pos - prev_prev_pos
                    
                    # Penalize large changes in direction (curvature)
                    segment_change = torch.norm(current_segment - prev_segment, dim=1)
                    smoothness_reward = -segment_change * self.smoothness_weight
            
            # 5. Efficiency reward (encourage progress towards goal)
            efficiency_reward = torch.zeros(batch_size, device=self.device)
            if t > 0:
                prev_pos = positions[:, t-1, :]
                prev_goal_dist = torch.norm(prev_pos - self.goal_pos.unsqueeze(0), dim=1)
                progress = prev_goal_dist - goal_dist  # Positive if getting closer
                efficiency_reward = progress * self.efficiency_weight
            
            # Combine all rewards
            total_reward = ( collision_penalty + 
                          smoothness_reward + efficiency_reward)
            rewards[:, t] = total_reward
        
        return rewards

    def _check_collisions(self, positions: torch.Tensor) -> torch.Tensor:
        """Check for collisions with obstacles."""
        batch_size = positions.shape[0]
        collision = torch.zeros(batch_size, device=self.device)
        
        for obs_x, obs_y, obs_r in self.obstacles:
            obs_pos = torch.tensor([obs_x, obs_y], device=self.device)
            dist = torch.norm(positions - obs_pos.unsqueeze(0), dim=1)
            collision += torch.clamp(obs_r - dist, min=0) * 3
            
        return collision

    def batch_rollout(self, trajectories: torch.Tensor, initial_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform batch rollout of position trajectories.
        
        Args:
            trajectories: Position trajectories [batch_size, horizon_samples, 2]
            initial_states: Not used for this trajectory optimization task
            
        Returns:
            Rewards for each trajectory point [batch_size, horizon_samples]
        """
        batch_size, horizon_samples, pos_dim = trajectories.shape
        
        if pos_dim != 2:
            raise ValueError(f"Expected 2D positions, got {pos_dim}D")
        
        # Trajectories now include start and end nodes
        # Compute rewards for each point in the trajectory
        rewards = self._compute_trajectory_point_rewards(trajectories)
        
        return rewards

    def get_state_dim(self) -> int:
        """Get state dimension."""
        return 2  # [x, y] positions

    def get_action_dim(self) -> int:
        """Get action dimension."""
        return 2  # [x, y] position waypoints

    def visualize_trajectory(self, trajectory: torch.Tensor, title: str = "") -> plt.Figure:
        """Visualize a position trajectory."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        # Handle batch dimension
        if trajectory.dim() == 3:
            trajectory = trajectory[0]  # Take first batch item
        
        # Ensure trajectory is on correct device
        trajectory = trajectory.to(self.device)
        
        # Trajectory already includes all nodes (start to end)
        positions = trajectory.cpu().numpy()
        

        
        # Plot obstacles
        for obs_x, obs_y, obs_r in self.obstacles:
            circle = patches.Circle((obs_x, obs_y), obs_r, fill=True, alpha=0.3, color='red')
            ax.add_patch(circle)
        
        # Plot goal
        goal_circle = patches.Circle(self.goal_pos.cpu().numpy(), self.goal_radius, 
                                   fill=True, alpha=0.3, color='green')
        ax.add_patch(goal_circle)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
        
        # Plot trajectory nodes (control points)
        # ax.plot(positions[:, 0], positions[:, 1], 'ko', markersize=6, alpha=0.7, label='Nodes')
        
        # Mark start and end nodes specially
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start (Fixed)')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End (Fixed)')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title or '2D Navigation Trajectory')
        # make legend at left down corner
        ax.legend(loc='lower left')
        ax.grid(True)
        ax.set_aspect('equal')
        
        # Plot workspace
        ax.set_xlim(-2, self.workspace_size[0]+2)
        ax.set_ylim(-2, self.workspace_size[1]+2)
        return fig


class InvertedPendulumEnv(TrajOptEnvBase):
    """Inverted pendulum environment for optimal control.
    
    The goal is to balance an inverted pendulum by applying forces to the base.
    The state includes cart position, cart velocity, pendulum angle, and angular velocity.
    The action is the force applied to the cart.
    """

    def __init__(self,
                 config: EnvConfig,
                 cart_mass: float = 1.0,
                 pole_mass: float = 0.1,
                 pole_length: float = 1.0,
                 max_force: float = 20.0,
                 gravity: float = 9.81,
                 target_angle: float = 0.0,
                 target_position: float = 0.0,
                 angle_weight: float = 1.0,
                 position_weight: float = 0.5,
                 velocity_weight: float = 0.1,
                 control_weight: float = 0.01):
        """Initialize inverted pendulum environment.

        Args:
            config: Environment configuration
            cart_mass: Mass of the cart
            pole_mass: Mass of the pole
            pole_length: Length of the pole
            max_force: Maximum force that can be applied
            gravity: Gravitational acceleration
            target_angle: Target pendulum angle (0 = upright)
            target_position: Target cart position
            angle_weight: Weight for angle tracking error
            position_weight: Weight for position tracking error
            velocity_weight: Weight for velocity penalty
            control_weight: Weight for control effort penalty
        """
        super().__init__(config)
        
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.max_force = max_force
        self.gravity = gravity
        self.target_angle = target_angle
        self.target_position = target_position
        self.angle_weight = angle_weight
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.control_weight = control_weight

        # Total mass
        self.total_mass = cart_mass + pole_mass
        
        # Pole moment of inertia (around center of mass)
        self.pole_moment = pole_mass * pole_length**2 / 12
        
        # Current state
        self.current_state = None

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """Reset the environment."""
        # State: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        initial_state = torch.zeros(batch_size, 4, device=self.device)
        
        # Random initial conditions
        initial_state[:, 0] = 1.0  # Cart position
        initial_state[:, 1] = 0.2  # Cart velocity
        initial_state[:, 2] = 0.2  # Pole angle
        initial_state[:, 3] = 0.1  # Angular velocity
        self.current_state = initial_state
        return initial_state

    def step(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take a step in the environment using RK4 integration."""
        batch_size = state.shape[0]
        
        # Clip actions
        action = torch.clamp(action, -self.max_force, self.max_force)
        
        # RK4 integration
        dt = self.dt
        k1 = self._dynamics(state, action)
        k2 = self._dynamics(state + 0.5 * dt * k1, action)
        k3 = self._dynamics(state + 0.5 * dt * k2, action)
        k4 = self._dynamics(state + dt * k3, action)
        
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize angle to [-pi, pi]
        next_state[:, 2] = self._normalize_angle(next_state[:, 2])
        
        # Compute rewards
        rewards = self._compute_step_rewards(state, action, next_state)
        
        # Check if done (episode length or failure)
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        return next_state, rewards, done

    def _dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute system dynamics."""
        batch_size = state.shape[0]
        
        # Extract state variables
        x = state[:, 0]      # Cart position
        x_dot = state[:, 1]  # Cart velocity
        theta = state[:, 2]  # Pole angle
        theta_dot = state[:, 3]  # Pole angular velocity
        
        # Extract force
        force = action.squeeze(-1) if action.dim() > 1 else action
        
        # Precompute trigonometric functions
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # Dynamics equations (derived from Lagrangian mechanics)
        # Mass matrix and forces for the cart-pole system
        
        # Common terms
        ml = self.pole_mass * self.pole_length
        ml_cos = ml * cos_theta
        ml_sin = ml * sin_theta
        
        # Mass matrix components
        M11 = self.total_mass
        M12 = ml_cos
        M22 = self.pole_moment + ml * self.pole_length**2
        
        # Determinant of mass matrix
        det_M = M11 * M22 - M12 * ml_cos
        
        # Force vector
        F1 = force + ml * self.pole_length * theta_dot**2 * sin_theta
        F2 = -ml * self.gravity * self.pole_length * sin_theta
        
        # Solve for accelerations
        x_ddot = (M22 * F1 - M12 * F2) / det_M
        theta_ddot = (-ml_cos * F1 + M11 * F2) / det_M
        
        # State derivative
        state_dot = torch.zeros_like(state)
        state_dot[:, 0] = x_dot
        state_dot[:, 1] = x_ddot
        state_dot[:, 2] = theta_dot
        state_dot[:, 3] = theta_ddot
        
        return state_dot

    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]."""
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _compute_step_rewards(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Compute rewards for a single step."""
        batch_size = state.shape[0]
        
        # Extract state variables
        cart_pos = next_state[:, 0]
        cart_vel = next_state[:, 1]
        pole_angle = next_state[:, 2]
        pole_angular_vel = next_state[:, 3]
        
        # Angle tracking error (want to keep pole upright)
        angle_error = self._normalize_angle(pole_angle - self.target_angle)
        angle_cost = self.angle_weight * angle_error**2
        
        # Position tracking error
        position_error = cart_pos - self.target_position
        position_cost = self.position_weight * position_error**2
        
        # Velocity penalties (encourage stable motion)
        velocity_cost = self.velocity_weight * (cart_vel**2 + pole_angular_vel**2)
        
        # Control effort penalty
        force = action.squeeze(-1) if action.dim() > 1 else action
        control_cost = self.control_weight * force**2
        
        # Total reward (negative cost)
        rewards = -(angle_cost + position_cost + velocity_cost + control_cost)
        # Bonus for staying upright
        # upright_bonus = torch.exp(-5 * angle_error**2) * 0.1
        # rewards += upright_bonus
        
        return rewards

    def batch_rollout(self, trajectories: torch.Tensor, initial_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform batch rollout of trajectories."""
        batch_size, horizon_samples, action_dim = trajectories.shape
        
        if initial_states is None:
            states = self.reset(batch_size)
        else:
            states = initial_states.clone()
        
        rewards = torch.zeros(batch_size, horizon_samples, device=self.device)
        
        for t in range(horizon_samples):
            actions = trajectories[:, t, :]
            states, step_rewards, done = self.step(states, actions)
            rewards[:, t] = step_rewards
        
        return rewards

    def get_state_dim(self) -> int:
        """Get state dimension."""
        return 4  # [cart_pos, cart_vel, pole_angle, pole_angular_vel]

    def get_action_dim(self) -> int:
        """Get action dimension."""
        return 1  # [force]

    def visualize_trajectory(self, trajectory: torch.Tensor, title: str = "") -> plt.Figure:
        """Visualize a trajectory."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        if trajectory.dim() == 3:
            trajectory = trajectory[0]  # Take first batch item
        
        # Ensure trajectory is on correct device
        trajectory = trajectory.to(self.device)
        
        # Simulate trajectory to get states
        initial_state = self.reset(1)
        states = [initial_state[0].cpu().numpy()]
        current_state = initial_state
        
        for t in range(trajectory.shape[0]):
            action = trajectory[t:t+1].unsqueeze(0)
            current_state, _, _ = self.step(current_state, action)
            states.append(current_state[0].cpu().numpy())
        
        states = np.array(states)
        times = np.arange(len(states)) * self.dt
        
        # Plot cart position
        axes[0, 0].plot(times, states[:, 0])
        axes[0, 0].axhline(y=self.target_position, color='r', linestyle='--', label='Target')
        axes[0, 0].set_ylabel('Cart Position [m]')
        axes[0, 0].set_title('Cart Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot pendulum angle
        axes[0, 1].plot(times, states[:, 2] * 180 / np.pi)
        axes[0, 1].axhline(y=self.target_angle * 180 / np.pi, color='r', linestyle='--', label='Target')
        axes[0, 1].set_ylabel('Pole Angle [deg]')
        axes[0, 1].set_title('Pole Angle')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot velocities
        axes[1, 0].plot(times, states[:, 1], label='Cart Velocity')
        axes[1, 0].plot(times, states[:, 3], label='Angular Velocity')
        axes[1, 0].set_ylabel('Velocity')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_title('Velocities')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot control input
        control_times = np.arange(trajectory.shape[0]) * self.dt
        axes[1, 1].plot(control_times, trajectory.cpu().numpy())
        axes[1, 1].set_ylabel('Force [N]')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_title('Control Input')
        axes[1, 1].grid(True)
        
        fig.suptitle(title or 'Inverted Pendulum Trajectory')
        plt.tight_layout()
        
        return fig


class MultiTaskEnv(TrajOptEnvBase):
    """Multi-task environment that can switch between different tasks."""

    def __init__(self, 
                 config: EnvConfig,
                 task_name: str = "navigation2d"):
        """Initialize multi-task environment.

        Args:
            config: Environment configuration
            task_name: Name of the task ("navigation2d" or "inverted_pendulum")
        """
        super().__init__(config)
        self.task_name = task_name
        
        if task_name == "navigation2d":
            self.env = Navigation2DEnv(config)
        elif task_name == "inverted_pendulum":
            self.env = InvertedPendulumEnv(config)
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """Reset the environment."""
        return self.env.reset(batch_size)

    def step(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take a step in the environment."""
        return self.env.step(state, action)

    def batch_rollout(self, trajectories: torch.Tensor, initial_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform batch rollout of trajectories."""
        return self.env.batch_rollout(trajectories, initial_states)

    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.env.get_state_dim()

    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.env.get_action_dim()

    def visualize_trajectory(self, trajectory: torch.Tensor, title: str = "") -> plt.Figure:
        """Visualize a trajectory."""
        return self.env.visualize_trajectory(trajectory, title)

    def set_task(self, task_name: str):
        """Switch to a different task."""
        self.task_name = task_name
        if task_name == "navigation2d":
            self.env = Navigation2DEnv(self.config)
        elif task_name == "inverted_pendulum":
            self.env = InvertedPendulumEnv(self.config)
        else:
            raise ValueError(f"Unknown task: {task_name}")


# Factory functions for easy environment creation
def create_navigation2d_env(config: Optional[EnvConfig] = None, **kwargs) -> Navigation2DEnv:
    """Create a 2D navigation environment."""
    if config is None:
        config = EnvConfig()
    return Navigation2DEnv(config, **kwargs)


def create_inverted_pendulum_env(config: Optional[EnvConfig] = None, **kwargs) -> InvertedPendulumEnv:
    """Create an inverted pendulum environment."""
    if config is None:
        config = EnvConfig()
    return InvertedPendulumEnv(config, **kwargs)


def create_multitask_env(task_name: str, config: Optional[EnvConfig] = None) -> MultiTaskEnv:
    """Create a multi-task environment."""
    if config is None:
        config = EnvConfig()
    return MultiTaskEnv(config, task_name)

