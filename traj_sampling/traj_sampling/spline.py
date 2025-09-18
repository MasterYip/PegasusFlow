import torch
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from typing import Optional, Union
from torch.nn import functional as F


class SplineBase:
    """Base class for spline trajectory representation.

    This class provides the basic structure and methods for spline-based trajectory
    representations, including cubic B-splines and other spline types.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize the spline base class.

        Args:
            device: Device to use for tensor operations
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_basis_mask_matrix(self, num_knots: int, num_samples: int) -> torch.Tensor:
        """Compute the basis mask matrix Φ that maps from knot space to dense trajectory space.

        Args:
            num_knots: Number of knot points in the trajectory
            num_samples: Number of sample points in the dense trajectory

        Returns:
            Basis mask matrix Φ [num_samples, num_knots]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate_spline(self, knot_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate spline at parameter t.
        Args:
            knot_points: Control points [num_knots, dim]
            t: Parameter value or batch of parameter values in range [0, 1]
        Returns:
            Position(s) along the spline
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def interpolate_batch(self, knot_points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Interpolate from knot points to dense trajectory points for a batch.

        Args:
            knot_points: Batch of knot points [batch_size, num_knots, dim]
            num_samples: Number of samples in the output trajectory

        Returns:
            Batch of dense trajectories [batch_size, num_samples, dim]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def node2dense(self, nodes: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert control nodes to dense control sequence.

        Args:
            nodes: Control nodes [Hnode+1, action_dim] or batch [batch_size, Hnode+1, action_dim]

        Returns:
            Dense control sequence [Hsample+1, action_dim] or batch [batch_size, Hsample+1, action_dim]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def dense2node(self, dense: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert dense control sequence to control nodes.

        Args:
            dense: Dense control sequence [Hsample+1, action_dim] or batch [batch_size, Hsample+1, action_dim]

        Returns:
            Control nodes [Hnode+1, action_dim] or batch [batch_size, Hnode+1, action_dim]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class LinearSpline(SplineBase):
    """Linear interpolation-based spline trajectory representation.

    This class implements simple linear interpolation between control nodes,
    providing an efficient fallback when more complex spline interpolation
    is not needed or available.
    """

    def __init__(self,
                 horizon_nodes: int,
                 horizon_samples: int,
                 dt: float = 0.02,
                 device: Optional[torch.device] = None):
        """Initialize the linear spline class.

        Args:
            horizon_nodes: Number of control nodes
            horizon_samples: Number of sample points
            dt: Time step for trajectory
            device: Device to use for tensor operations
        """
        super().__init__(device)
        self.horizon_nodes = horizon_nodes
        self.horizon_samples = horizon_samples
        self.dt = dt

        print(f"LinearSpline initialized: {horizon_nodes} nodes -> {horizon_samples} samples")

    def compute_basis_mask_matrix(self, num_knots: int, num_samples: int) -> torch.Tensor:
        """Compute the basis mask matrix Φ that maps from knot space to dense trajectory space.

        Args:
            num_knots: Number of knot points in the trajectory
            num_samples: Number of sample points in the dense trajectory

        Returns:
            Basis mask matrix Φ [num_samples, num_knots]
        """
        # Create time grids
        knot_times = torch.linspace(0, 1, num_knots, device=self.device)
        sample_times = torch.linspace(0, 1, num_samples, device=self.device)

        # Initialize basis matrix
        phi = torch.zeros((num_samples, num_knots), device=self.device)

        # For each sample point, find which knots it interpolates between
        for i, t_sample in enumerate(sample_times):
            # Find the segment this sample falls into
            segment_idx = torch.searchsorted(knot_times[1:], t_sample, right=False)
            segment_idx = torch.clamp(segment_idx, 0, num_knots - 2)

            # Get the two knot times that bound this sample
            t0 = knot_times[segment_idx]
            t1 = knot_times[segment_idx + 1]

            # Compute linear interpolation weights
            if t1 - t0 > 1e-10:
                alpha = (t_sample - t0) / (t1 - t0)
            else:
                alpha = 0.0

            # Set the weights for linear interpolation
            phi[i, segment_idx] = 1.0 - alpha
            phi[i, segment_idx + 1] = alpha

        return phi

    def evaluate_spline(self, knot_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate spline at parameter t using linear interpolation.

        Args:
            knot_points: Control points [num_knots, dim]
            t: Parameter value or batch of parameter values in range [0, 1]

        Returns:
            Position(s) along the spline
        """
        num_knots, dim = knot_points.shape

        # Handle scalar t
        if t.dim() == 0:
            t = t.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = t.shape[0]
        result = torch.zeros((batch_size, dim), device=self.device)

        # Create time grid for knots
        knot_times = torch.linspace(0, 1, num_knots, device=self.device)

        for i, t_val in enumerate(t):
            # Find the segment this t_val falls into
            segment_idx = torch.searchsorted(knot_times[1:], t_val, right=False)
            segment_idx = torch.clamp(segment_idx, 0, num_knots - 2)

            # Get the two knot times that bound this sample
            t0 = knot_times[segment_idx]
            t1 = knot_times[segment_idx + 1]

            # Compute linear interpolation weight
            if t1 - t0 > 1e-10:
                alpha = (t_val - t0) / (t1 - t0)
            else:
                alpha = 0.0

            # Linear interpolation between the two control points
            result[i] = (1.0 - alpha) * knot_points[segment_idx] + alpha * knot_points[segment_idx + 1]

        return result.squeeze(0) if squeeze_output else result

    def interpolate_batch(self, knot_points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Interpolate from knot points to dense trajectory points for a batch using linear interpolation.

        Args:
            knot_points: Batch of knot points [batch_size, num_knots, dim]
            num_samples: Number of samples in the output trajectory

        Returns:
            Batch of dense trajectories [batch_size, num_samples, dim]
        """
        batch_size, num_knots, dim = knot_points.shape

        # Use torch.nn.functional.interpolate for efficient batch linear interpolation
        # Reshape to [batch_size * dim, 1, num_knots] for interpolation
        knot_points_reshaped = knot_points.permute(0, 2, 1).contiguous()  # [batch_size, dim, num_knots]
        knot_points_reshaped = knot_points_reshaped.view(batch_size * dim, 1, num_knots)

        # Interpolate to desired length
        dense_reshaped = F.interpolate(
            knot_points_reshaped,
            size=num_samples,
            mode='linear',
            align_corners=True
        )  # [batch_size * dim, 1, num_samples]

        # Reshape back to [batch_size, num_samples, dim]
        dense_trajectories = dense_reshaped.view(batch_size, dim, num_samples).permute(0, 2, 1)

        return dense_trajectories

    def node2dense(self, nodes: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert control nodes to dense control sequence using linear interpolation.

        Args:
            nodes: Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]

        Returns:
            Dense control sequence [horizon_samples, action_dim] or batch [batch_size, horizon_samples, action_dim]
        """
        if isinstance(nodes, list):
            nodes = torch.stack(nodes)

        if nodes.dim() == 2:
            # Single trajectory [num_knots, action_dim]
            return self.interpolate_batch(nodes.unsqueeze(0), self.horizon_samples).squeeze(0)
        elif nodes.dim() == 3:
            # Batch of trajectories [batch_size, num_knots, action_dim]
            return self.interpolate_batch(nodes, self.horizon_samples)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {nodes.dim()}D")

    def dense2node(self, dense: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert dense control sequence to control nodes using linear downsampling.

        Args:
            dense: Dense control sequence [horizon_samples, action_dim] or batch [batch_size, horizon_samples, action_dim]

        Returns:
            Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]
        """
        if isinstance(dense, list):
            dense = torch.stack(dense)

        if dense.dim() == 2:
            # Single trajectory [num_samples, action_dim]
            num_samples, action_dim = dense.shape

            # Use torch.nn.functional.interpolate for downsampling
            # Reshape to [1, action_dim, num_samples] for interpolation
            dense_reshaped = dense.T.unsqueeze(0)  # [1, action_dim, num_samples]

            # Interpolate to desired node count
            nodes_reshaped = F.interpolate(
                dense_reshaped,
                size=self.horizon_nodes,
                mode='linear',
                align_corners=True
            )  # [1, action_dim, horizon_nodes]

            # Reshape back to [horizon_nodes, action_dim]
            return nodes_reshaped.squeeze(0).T

        elif dense.dim() == 3:
            # Batch of trajectories [batch_size, num_samples, action_dim]
            batch_size, num_samples, action_dim = dense.shape

            # Use torch.nn.functional.interpolate for batch downsampling
            # Reshape to [batch_size * action_dim, 1, num_samples] for interpolation
            dense_reshaped = dense.permute(0, 2, 1).contiguous()  # [batch_size, action_dim, num_samples]
            dense_reshaped = dense_reshaped.view(batch_size * action_dim, 1, num_samples)

            # Interpolate to desired node count
            nodes_reshaped = F.interpolate(
                dense_reshaped,
                size=self.horizon_nodes,
                mode='linear',
                align_corners=True
            )  # [batch_size * action_dim, 1, horizon_nodes]

            # Reshape back to [batch_size, horizon_nodes, action_dim]
            nodes = nodes_reshaped.view(batch_size, action_dim, self.horizon_nodes).permute(0, 2, 1)

            return nodes
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {dense.dim()}D")


class UniBSpline(SplineBase):
    """Uniform cubic B-spline trajectory representation.

    This class implements cubic B-splines with uniform knot spacing for
    trajectory interpolation and manipulation.
    """

    def __init__(self,
                 horizon_nodes: int,
                 horizon_samples: int,
                 dt: float = 0.02,
                 device: Optional[torch.device] = None):
        """Initialize the uniform B-spline class.

        Args:
            horizon_nodes: Number of control nodes
            horizon_samples: Number of sample points
            dt: Time step for trajectory
            device: Device to use for tensor operations
        """
        super().__init__(device)
        self.horizon_nodes = horizon_nodes
        self.horizon_samples = horizon_samples
        self.dt = dt

        # Precompute the basis matrix phi and its pseudo-inverse for efficient dense2node conversion
        self.phi = self.compute_basis_mask_matrix(horizon_nodes, horizon_samples)
        self.phi_pinv = torch.linalg.pinv(self.phi)  # [horizon_nodes, horizon_samples]

        print(f"UniBSpline initialized: {horizon_nodes} nodes -> {horizon_samples} samples")
        print(f"Phi matrix shape: {self.phi.shape}, Phi_pinv shape: {self.phi_pinv.shape}")

    @staticmethod
    def get_cubic_bspline_basis_matrix():
        """Get the basis matrix for uniform cubic B-splines.

        Returns:
            torch.Tensor: The 4x4 basis matrix for cubic B-splines
        """
        return torch.tensor([
            [1.0, 4.0, 1.0, 0.0],
            [-3.0, 0.0, 3.0, 0.0],
            [3.0, -6.0, 3.0, 0.0],
            [-1.0, 3.0, -3.0, 1.0]
        ]) / 6.0

    @staticmethod
    def get_power_basis_vector(t: torch.Tensor) -> torch.Tensor:
        """Compute the power basis vector [1, t, t^2, t^3] for a given parameter t.

        Args:
            t: Parameter value or batch of parameter values

        Returns:
            Power basis vector(s)
        """
        if t.dim() > 0:
            ones = torch.ones_like(t)
            t_squared = t ** 2
            t_cubed = t ** 3
            return torch.stack([ones, t, t_squared, t_cubed], dim=-1)
        else:
            return torch.tensor([1.0, t, t**2, t**3], device=t.device)

    @staticmethod
    def get_velocity_basis_vector(t: torch.Tensor) -> torch.Tensor:
        """Compute the velocity basis vector [0, 1, 2t, 3t^2] for a given parameter t."""
        if t.dim() > 0:
            zeros = torch.zeros_like(t)
            ones = torch.ones_like(t)
            t_doubled = 2 * t
            t_squared_tripled = 3 * (t ** 2)
            return torch.stack([zeros, ones, t_doubled, t_squared_tripled], dim=-1)
        else:
            return torch.tensor([0.0, 1.0, 2 * t, 3 * (t**2)], device=t.device)

    @staticmethod
    def get_acceleration_basis_vector(t: torch.Tensor) -> torch.Tensor:
        """Compute the acceleration basis vector [0, 0, 2, 6t] for a given parameter t."""
        if t.dim() > 0:
            zeros = torch.zeros_like(t)
            twos = 2 * torch.ones_like(t)
            t_six = 6 * t
            return torch.stack([zeros, zeros, twos, t_six], dim=-1)
        else:
            return torch.tensor([0.0, 0.0, 2.0, 6 * t], device=t.device)

    def compute_basis_mask_matrix(self, num_knots: int, num_samples: int) -> torch.Tensor:
        """Compute the basis mask matrix Φ that maps from knot space to dense trajectory space.

        Args:
            num_knots: Number of knot points in the trajectory
            num_samples: Number of sample points in the dense trajectory

        Returns:
            Basis mask matrix Φ [num_samples, num_knots]
        """
        t_samples = torch.linspace(0, 1, num_samples, device=self.device)
        phi = torch.zeros((num_samples, num_knots), device=self.device)
        M_B = self.get_cubic_bspline_basis_matrix().to(self.device)

        num_segments = max(1, num_knots - 3)
        segment_length = 1.0 / num_segments

        for i in range(num_samples):
            t = t_samples[i]
            segment_idx = min(int(t * num_segments), num_segments - 1)
            local_t = (t - segment_idx * segment_length) / segment_length
            T = self.get_power_basis_vector(local_t)
            start_knot = segment_idx

            for j in range(4):
                knot_idx = start_knot + j
                if knot_idx < num_knots:
                    weight = torch.matmul(T, M_B[:, j])
                    phi[i, knot_idx] = weight

        # Normalize rows to ensure partition of unity
        row_sums = phi.sum(dim=1, keepdim=True)
        non_zero_rows = row_sums.squeeze(1) > 0
        for i in range(num_samples):
            if non_zero_rows[i]:
                phi[i] = phi[i] / row_sums[i]

        return phi

    def evaluate_spline(self, knot_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate spline at parameter t.

        Args:
            knot_points: Control points [num_knots, dim]
            t: Parameter value or batch of parameter values in range [0, 1]

        Returns:
            Position(s) along the spline
        """
        num_points = knot_points.shape[0]
        dim = knot_points.shape[1]

        # Map t to the appropriate segment
        segment_idx = torch.clamp(
            torch.floor(t * (num_points - 3)),
            0, num_points - 4
        ).long()

        # Local t within the segment [0, 1]
        local_t = t * (num_points - 3) - segment_idx

        # Get basis matrix and power basis
        M_B = self.get_cubic_bspline_basis_matrix().to(self.device)

        # Handle batched t values
        if t.dim() > 0:
            positions = torch.zeros((t.shape[0], dim), device=self.device)
            for i, (seg_idx, l_t) in enumerate(zip(segment_idx, local_t)):
                P = knot_points[seg_idx:seg_idx + 4]
                T_i = self.get_power_basis_vector(l_t)
                positions[i] = torch.matmul(torch.matmul(T_i, M_B), P)
            return positions
        else:
            P = knot_points[segment_idx:segment_idx + 4]
            T = self.get_power_basis_vector(local_t)
            return torch.matmul(torch.matmul(T, M_B), P)

    def interpolate_batch(self, knot_points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Interpolate from knot points to dense trajectory points for a batch.

        Args:
            knot_points: Batch of knot points [batch_size, num_knots, dim]
            num_samples: Number of samples in the output trajectory

        Returns:
            Batch of dense trajectories [batch_size, num_samples, dim]
        """
        batch_size, num_knots, dim = knot_points.shape
        phi = self.compute_basis_mask_matrix(num_knots, num_samples)
        dense_trajectories = torch.zeros((batch_size, num_samples, dim), device=self.device)

        for d in range(dim):
            phi_reshaped = phi.unsqueeze(0)
            knots_d = knot_points[:, :, d].unsqueeze(2)
            dense_d = torch.bmm(phi_reshaped.expand(batch_size, -1, -1), knots_d)
            dense_trajectories[:, :, d] = dense_d.squeeze(2)

        return dense_trajectories

    def node2dense(self, nodes: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert control nodes to dense control sequence.

        Args:
            nodes: Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]

        Returns:
            Dense control sequence [horizon_samples+1, action_dim] or batch [batch_size, horizon_samples+1, action_dim]
        """
        if isinstance(nodes, list):
            nodes = torch.stack(nodes)

        if nodes.dim() == 2:
            # Single trajectory [horizon_nodes, action_dim]
            return self.interpolate_batch(nodes.unsqueeze(0), self.horizon_samples).squeeze(0)
        elif nodes.dim() == 3:
            # Batch of trajectories [batch_size, horizon_nodes, action_dim]
            return self.interpolate_batch(nodes, self.horizon_samples)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {nodes.dim()}D")

    def dense2node(self, dense: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert dense control sequence to control nodes using least squares fitting.

        Args:
            dense: Dense control sequence [horizon_samples+1, action_dim] or batch [batch_size, horizon_samples+1, action_dim]

        Returns:
            Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]
        """
        if isinstance(dense, list):
            dense = torch.stack(dense)

        # Check if we're dealing with a batch or a single trajectory
        is_batch = dense.dim() == 3

        if is_batch:
            batch_size, num_samples, action_dim = dense.shape

            # Verify input dimensions match expected
            if num_samples != self.horizon_samples:
                raise ValueError(f"Expected dense trajectory with {self.horizon_samples} samples, got {num_samples}")

            nodes = torch.zeros((batch_size, self.horizon_nodes, action_dim), device=self.device)

            for d in range(action_dim):
                # Use batched matrix multiplication for efficiency
                # Reshape phi_pinv to [1, horizon_nodes, horizon_samples+1] for batch matmul
                phi_pinv_reshaped = self.phi_pinv.unsqueeze(0)
                # Reshape dense to [batch_size, horizon_samples+1, 1]
                dense_d = dense[:, :, d].unsqueeze(2)
                # Perform batched matrix multiplication
                nodes_d = torch.bmm(phi_pinv_reshaped.expand(batch_size, -1, -1), dense_d)
                # Store the result [batch_size, horizon_nodes]
                nodes[:, :, d] = nodes_d.squeeze(2)

            return nodes
        else:
            # Single trajectory [horizon_samples+1, action_dim]
            num_samples, action_dim = dense.shape

            # Verify input dimensions match expected
            if num_samples != self.horizon_samples:
                raise ValueError(f"Expected dense trajectory with {self.horizon_samples} samples, got {num_samples}")

            # Add batch dimension, convert, then remove batch dimension
            dense_batched = dense.unsqueeze(0)
            nodes_batched = self.dense2node(dense_batched)
            return nodes_batched.squeeze(0)


class CatmullRomSpline(SplineBase):
    """PyTorch-based interpolated spline(Catmull-Rom) using basis function matrices.

    This class implements cubic spline interpolation using precomputed basis function
    matrices instead of solving linear systems, making it highly efficient for batch
    calculations and GPU acceleration while maintaining strict interpolation.
    """

    def __init__(self,
                 horizon_nodes: int,
                 horizon_samples: int,
                 dt: float = 0.02,
                 spline_degree: int = 3,
                 device: Optional[torch.device] = None):
        """Initialize the torch interpolated spline.

        Args:
            horizon_nodes: Number of control nodes
            horizon_samples: Number of sample points
            dt: Time step for trajectory
            spline_degree: Degree of spline (only 3 supported for now)
            device: Device to use for tensor operations
        """
        super().__init__(device)
        self.horizon_nodes = horizon_nodes
        self.horizon_samples = horizon_samples
        self.dt = dt
        self.spline_degree = spline_degree

        if spline_degree != 3:
            raise NotImplementedError("Only cubic splines (degree=3) are currently supported")

        # Create time grids
        self.knot_times = torch.linspace(0, 1, horizon_nodes, device=self.device)
        self.sample_times = torch.linspace(0, 1, horizon_samples, device=self.device)

        # Precompute basis matrices for efficiency
        self.phi_interpolation = self.compute_interpolation_basis_matrix()

        # Precompute pseudo-inverse for dense2node conversion
        self.phi_pinv = torch.linalg.pinv(self.phi_interpolation)

        print(f"CatmullRomSpline initialized: {horizon_nodes} nodes -> {horizon_samples} samples")
        print(f"Interpolation matrix shape: {self.phi_interpolation.shape}")

    def compute_lagrange_basis_functions(self, t: torch.Tensor, knot_positions: torch.Tensor) -> torch.Tensor:
        """Compute Lagrange interpolation basis functions.

        This ensures exact interpolation through all knot points using
        Lagrange polynomials.

        Args:
            t: Parameter values [n_eval]
            knot_positions: Knot parameter positions [n_knots]

        Returns:
            Basis matrix [n_eval, n_knots]
        """
        n_eval = t.shape[0]
        n_knots = knot_positions.shape[0]

        # Initialize basis matrix
        basis = torch.zeros(n_eval, n_knots, device=self.device)

        # For each knot point
        for j in range(n_knots):
            # Compute Lagrange basis polynomial L_j(t)
            basis_j = torch.ones(n_eval, device=self.device)

            for k in range(n_knots):
                if k != j:
                    # Avoid division by zero
                    denominator = knot_positions[j] - knot_positions[k]
                    if torch.abs(denominator) > 1e-10:
                        basis_j *= (t - knot_positions[k]) / denominator
                    else:
                        # Handle case where knots are too close
                        basis_j *= 0.0

            basis[:, j] = basis_j

        return basis

    def compute_cubic_spline_basis_piecewise(self, t: torch.Tensor, knot_positions: torch.Tensor) -> torch.Tensor:
        """Compute piecewise cubic spline basis functions using Catmull-Rom interpolation.
        
        Catmull-Rom splines guarantee interpolation through all knot points while maintaining
        C1 continuity. Each basis function has a maximum value of 1 at its corresponding knot.
        
        Args:
            t: Parameter values [n_eval]
            knot_positions: Knot parameter positions [n_knots]
            
        Returns:
            Basis matrix [n_eval, n_knots]
        """
        n_eval = t.shape[0]
        n_knots = knot_positions.shape[0]
        
        if n_knots < 4:
            # Fall back to Lagrange interpolation for small numbers of knots
            return self.compute_lagrange_basis_functions(t, knot_positions)
        
        # Initialize basis matrix
        basis = torch.zeros(n_eval, n_knots, device=self.device)
        
        # For each evaluation point
        for i, t_val in enumerate(t):
            # Find the segment containing t_val
            segment_idx = torch.searchsorted(knot_positions[1:], t_val, right=False)
            segment_idx = torch.clamp(segment_idx, 0, n_knots - 2)
            
            # Get the segment bounds
            t0 = knot_positions[segment_idx]
            t1 = knot_positions[segment_idx + 1]
            
            # Normalized parameter within segment [0, 1]
            if t1 - t0 > 1e-10:
                u = (t_val - t0) / (t1 - t0)
            else:
                u = 0.0
            
            # Catmull-Rom requires 4 control points: P-1, P0, P1, P2
            # For segment [P0, P1], we need points at indices: segment_idx-1, segment_idx, segment_idx+1, segment_idx+2
            
            # Handle boundary conditions for Catmull-Rom
            p_minus1_idx = max(0, segment_idx - 1)
            p0_idx = segment_idx
            p1_idx = segment_idx + 1
            p2_idx = min(n_knots - 1, segment_idx + 2)
            
            # If we're at boundaries, extend the endpoints
            if segment_idx == 0:
                # Extend first point: P-1 = 2*P0 - P1
                p_minus1_idx = p0_idx
            if segment_idx >= n_knots - 2:
                # Extend last point: P2 = 2*P1 - P0
                p2_idx = p1_idx
            
            # Catmull-Rom basis functions
            u2 = u * u
            u3 = u2 * u
            
            # Basis weights for the 4 control points
            # These are the standard Catmull-Rom basis functions
            w_minus1 = (-u3 + 2*u2 - u) / 2  # Weight for P-1
            w0 = (3*u3 - 5*u2 + 2) / 2       # Weight for P0
            w1 = (-3*u3 + 4*u2 + u) / 2      # Weight for P1  
            w2 = (u3 - u2) / 2               # Weight for P2
            
            # Assign weights to basis matrix
            # Handle boundary extension cases
            if segment_idx == 0 and p_minus1_idx == p0_idx:
                # Extended first point: combine w_minus1 and w0
                basis[i, p0_idx] += w_minus1 + w0
            else:
                basis[i, p_minus1_idx] += w_minus1
                basis[i, p0_idx] += w0
                
            basis[i, p1_idx] += w1
            
            if segment_idx >= n_knots - 2 and p2_idx == p1_idx:
                # Extended last point: combine w1 and w2
                basis[i, p1_idx] += w2
            else:
                basis[i, p2_idx] += w2
        
        return basis
    
    def compute_catmull_rom_basis_matrix(self, t: torch.Tensor, knot_positions: torch.Tensor) -> torch.Tensor:
        """Alternative implementation using direct Catmull-Rom matrix formulation.
        
        This version constructs the basis matrix by evaluating each knot's influence
        function directly, ensuring interpolation property.
        
        Args:
            t: Parameter values [n_eval]
            knot_positions: Knot parameter positions [n_knots]
            
        Returns:
            Basis matrix [n_eval, n_knots]
        """
        n_eval = t.shape[0]
        n_knots = knot_positions.shape[0]
        
        # Initialize basis matrix
        basis = torch.zeros(n_eval, n_knots, device=self.device)
        
        # For each knot, compute its influence on all evaluation points
        for j in range(n_knots):
            for i, t_val in enumerate(t):
                # Find which segments this knot influences
                # A knot can influence up to 4 segments in Catmull-Rom
                
                weight = 0.0
                
                # Check all possible segments where knot j could have influence
                for seg in range(max(0, j-2), min(n_knots-1, j+2)):
                    if seg >= n_knots - 1:
                        continue
                    
                    t_start = knot_positions[seg]
                    t_end = knot_positions[seg + 1]
                    
                    # Check if t_val is in this segment
                    if t_start <= t_val <= t_end:
                        # Compute normalized parameter
                        if t_end - t_start > 1e-10:
                            u = (t_val - t_start) / (t_end - t_start)
                        else:
                            u = 0.0
                        
                        # Determine which control point knot j is for this segment
                        control_point_idx = j - (seg - 1)  # Position relative to segment
                        
                        if 0 <= control_point_idx <= 3:
                            # Compute Catmull-Rom weight for this control point
                            u2 = u * u
                            u3 = u2 * u
                            
                            if control_point_idx == 0:  # P-1
                                weight = (-u3 + 2*u2 - u) / 2
                            elif control_point_idx == 1:  # P0
                                weight = (3*u3 - 5*u2 + 2) / 2
                            elif control_point_idx == 2:  # P1
                                weight = (-3*u3 + 4*u2 + u) / 2
                            elif control_point_idx == 3:  # P2
                                weight = (u3 - u2) / 2
                        
                        break  # Found the segment, no need to check others
                
                basis[i, j] = weight
        
        return basis

    def compute_interpolation_basis_matrix(self) -> torch.Tensor:
        """Compute the basis matrix for interpolation from knots to samples.

        Returns:
            Basis matrix Φ [horizon_samples, horizon_nodes]
        """
        return self.compute_cubic_spline_basis_piecewise(self.sample_times, self.knot_times)

    def compute_basis_mask_matrix(self, num_knots: int, num_samples: int) -> torch.Tensor:
        """Compute the basis mask matrix Φ that maps from knot space to dense trajectory space.

        Args:
            num_knots: Number of knot points
            num_samples: Number of sample points

        Returns:
            Basis mask matrix Φ [num_samples, num_knots]
        """
        # Create time grids for this configuration
        knot_times = torch.linspace(0, 1, num_knots, device=self.device)
        sample_times = torch.linspace(0, 1, num_samples, device=self.device)

        # Compute basis functions
        return self.compute_cubic_spline_basis_piecewise(sample_times, knot_times)

    def evaluate_spline(self, knot_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate spline at parameter t.

        Args:
            knot_points: Control points [num_knots, dim]
            t: Parameter value or batch of parameter values in range [0, 1]

        Returns:
            Position(s) along the spline
        """
        num_knots, dim = knot_points.shape

        # Handle scalar t
        if t.dim() == 0:
            t = t.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute basis functions for evaluation points
        basis = self.compute_cubic_spline_basis_piecewise(t, self.knot_times[:num_knots])

        # Matrix multiplication: [n_eval, n_knots] @ [n_knots, dim] = [n_eval, dim]
        result = torch.matmul(basis, knot_points)

        return result.squeeze(0) if squeeze_output else result

    def interpolate_batch(self, knot_points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Interpolate from knot points to dense trajectory points for a batch.

        Args:
            knot_points: Batch of knot points [batch_size, num_knots, dim]
            num_samples: Number of samples in the output trajectory

        Returns:
            Batch of dense trajectories [batch_size, num_samples, dim]
        """
        batch_size, num_knots, dim = knot_points.shape

        # Get appropriate basis matrix
        if num_samples == self.horizon_samples and num_knots == self.horizon_nodes:
            # Use precomputed matrix
            phi = self.phi_interpolation
        else:
            # Compute new basis matrix
            phi = self.compute_basis_mask_matrix(num_knots, num_samples)

        # Efficient batch matrix multiplication
        # phi: [num_samples, num_knots]
        # knot_points: [batch_size, num_knots, dim]
        # result: [batch_size, num_samples, dim]

        # Method 1: Use torch.bmm with broadcasting
        phi_expanded = phi.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_samples, num_knots]
        result = torch.bmm(phi_expanded, knot_points)  # [batch_size, num_samples, dim]

        return result

    def node2dense(self, nodes: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert control nodes to dense control sequence.

        Args:
            nodes: Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]

        Returns:
            Dense control sequence [horizon_samples, action_dim] or batch [batch_size, horizon_samples, action_dim]
        """
        if isinstance(nodes, list):
            nodes = torch.stack(nodes)

        if nodes.dim() == 2:
            # Single trajectory
            # Use precomputed interpolation matrix
            return torch.matmul(self.phi_interpolation, nodes)
        elif nodes.dim() == 3:
            # Batch of trajectories
            return self.interpolate_batch(nodes, self.horizon_samples)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {nodes.dim()}D")

    def dense2node(self, dense: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert dense control sequence to control nodes using precomputed pseudo-inverse.

        Args:
            dense: Dense control sequence [horizon_samples, action_dim] or batch [batch_size, horizon_samples, action_dim]

        Returns:
            Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]
        """
        if isinstance(dense, list):
            dense = torch.stack(dense)

        if dense.dim() == 2:
            # Single trajectory
            # Use precomputed pseudo-inverse
            return torch.matmul(self.phi_pinv, dense)
        elif dense.dim() == 3:
            # Batch of trajectories
            batch_size, num_samples, dim = dense.shape

            # Verify dimensions
            if num_samples != self.horizon_samples:
                raise ValueError(f"Expected {self.horizon_samples} samples, got {num_samples}")

            # Efficient batch multiplication using pseudo-inverse
            phi_pinv_expanded = self.phi_pinv.unsqueeze(0).expand(batch_size, -1, -1)
            result = torch.bmm(phi_pinv_expanded, dense)

            return result
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {dense.dim()}D")

    def get_derivatives(self, knot_points: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Get derivatives of the spline at parameter t using finite differences.

        Args:
            knot_points: Control points [num_knots, dim]
            t: Parameter value in range [0, 1]
            order: Order of derivative (1=velocity, 2=acceleration)

        Returns:
            Derivative values [dim]
        """
        if order not in [1, 2]:
            raise ValueError("Only first and second derivatives are supported")

        # Use finite differences with small epsilon
        eps = 1e-6

        if order == 1:
            # First derivative: f'(t) ≈ (f(t+ε) - f(t-ε)) / (2ε)
            t_plus = torch.clamp(t + eps, 0, 1)
            t_minus = torch.clamp(t - eps, 0, 1)

            f_plus = self.evaluate_spline(knot_points, t_plus)
            f_minus = self.evaluate_spline(knot_points, t_minus)

            derivative = (f_plus - f_minus) / (2 * eps)

        else:  # order == 2
            # Second derivative: f''(t) ≈ (f(t+ε) - 2f(t) + f(t-ε)) / ε²
            t_plus = torch.clamp(t + eps, 0, 1)
            t_minus = torch.clamp(t - eps, 0, 1)

            f_plus = self.evaluate_spline(knot_points, t_plus)
            f_center = self.evaluate_spline(knot_points, t)
            f_minus = self.evaluate_spline(knot_points, t_minus)

            derivative = (f_plus - 2 * f_center + f_minus) / (eps * eps)

        return derivative

    def torch_interpolate_single(self,
                                 x_knots: torch.Tensor,
                                 y_knots: torch.Tensor,
                                 x_eval: torch.Tensor) -> torch.Tensor:
        """Interpolate using basis functions for single or multi-dimensional y.

        Args:
            x_knots: Knot coordinates [n_knots]
            y_knots: Knot values [n_knots] or [n_knots, dim]
            x_eval: Evaluation points [n_eval]

        Returns:
            Interpolated values [n_eval] or [n_eval, dim]
        """
        # Normalize x_knots and x_eval to [0, 1] range
        x_min, x_max = x_knots.min(), x_knots.max()
        if x_max - x_min > 1e-10:
            x_knots_norm = (x_knots - x_min) / (x_max - x_min)
            x_eval_norm = torch.clamp((x_eval - x_min) / (x_max - x_min), 0, 1)
        else:
            x_knots_norm = x_knots
            x_eval_norm = x_eval

        # Compute basis functions
        basis = self.compute_cubic_spline_basis_piecewise(x_eval_norm, x_knots_norm)

        # Handle single dimension case
        if y_knots.dim() == 1:
            y_knots = y_knots.unsqueeze(-1)
            squeeze_output = True
        else:
            squeeze_output = False

        # Interpolate: [n_eval, n_knots] @ [n_knots, dim] = [n_eval, dim]
        result = torch.matmul(basis, y_knots)

        return result.squeeze(-1) if squeeze_output else result

    def torch_interpolate_batch(self,
                                x_knots: torch.Tensor,
                                y_knots: torch.Tensor,
                                x_eval: torch.Tensor) -> torch.Tensor:
        """Batch interpolation using basis functions.

        Args:
            x_knots: Knot coordinates [n_knots]
            y_knots: Batch of knot values [batch_size, n_knots, dim]
            x_eval: Evaluation points [n_eval]

        Returns:
            Batch of interpolated values [batch_size, n_eval, dim]
        """
        batch_size, n_knots, dim = y_knots.shape
        n_eval = x_eval.shape[0]

        # Normalize coordinates
        x_min, x_max = x_knots.min(), x_knots.max()
        if x_max - x_min > 1e-10:
            x_knots_norm = (x_knots - x_min) / (x_max - x_min)
            x_eval_norm = torch.clamp((x_eval - x_min) / (x_max - x_min), 0, 1)
        else:
            x_knots_norm = x_knots
            x_eval_norm = x_eval

        # Compute basis functions
        basis = self.compute_cubic_spline_basis_piecewise(x_eval_norm, x_knots_norm)

        # Batch matrix multiplication
        basis_expanded = basis.unsqueeze(0).expand(batch_size, -1, -1)
        result = torch.bmm(basis_expanded, y_knots)

        return result


class InterpolatedSpline(SplineBase):
    """Spline interpolation using scipy's InterpolatedUnivariateSpline.

    This class provides high-quality spline interpolation using scipy's proven
    algorithms while maintaining PyTorch tensor compatibility.
    """

    def __init__(self,
                 num_knots: int,
                 num_samples: int,
                 dt: float,
                 spline_degree: int = 2,
                 device: Optional[torch.device] = None):
        """Initialize the scipy spline interpolator.

        Args:
            num_knots: Number of knot points
            num_samples: Number of sample points
            dt: Time step for trajectory
            spline_degree: Degree of spline (1=linear, 2=quadratic, 3=cubic)
            device: Device for tensor operations
        """
        super().__init__(device)

        self.num_knots = num_knots
        self.num_samples = num_samples
        self.dt = dt
        self.spline_degree = spline_degree

        # Create time grids
        self.knot_times = torch.linspace(0, dt * num_samples, num_knots, device=self.device)
        self.sample_times = torch.linspace(0, dt * num_samples, num_samples, device=self.device)

        print(f"InterpolatedSpline initialized: {num_knots} knots -> {num_samples} samples, degree={spline_degree}")

    def scipy_interpolate_single(self,
                                 knot_times: torch.Tensor,
                                 knot_values: torch.Tensor,
                                 sample_times: torch.Tensor,
                                 k: int = 2) -> torch.Tensor:
        """Interpolate using scipy's InterpolatedUnivariateSpline for a single trajectory dimension.

        Args:
            knot_times: Time points for knots [num_knots]
            knot_values: Values at knot points [num_knots]
            sample_times: Time points for sampling [num_samples]
            k: Spline degree (1=linear, 2=quadratic, 3=cubic)

        Returns:
            Interpolated values at sample times [num_samples]
        """
        # Convert to numpy for scipy, ensuring proper numpy arrays
        knot_times_np = knot_times.detach().cpu().numpy()
        knot_values_np = knot_values.detach().cpu().numpy()
        sample_times_np = sample_times.detach().cpu().numpy()

        # Ensure we have proper numpy arrays, not JAX arrays
        if hasattr(knot_times_np, '__array__'):
            knot_times_np = np.array(knot_times_np)
        if hasattr(knot_values_np, '__array__'):
            knot_values_np = np.array(knot_values_np)
        if hasattr(sample_times_np, '__array__'):
            sample_times_np = np.array(sample_times_np)

        # Create spline
        spline = InterpolatedUnivariateSpline(knot_times_np, knot_values_np, k=k)

        # Evaluate at sample points
        interpolated_np = spline(sample_times_np)

        # Convert back to torch, ensuring proper numpy array conversion
        return torch.from_numpy(np.array(interpolated_np)).float().to(self.device)

    def scipy_interpolate_batch(self,
                                knot_times: torch.Tensor,
                                knot_points: torch.Tensor,
                                sample_times: torch.Tensor,
                                k: int = 2) -> torch.Tensor:
        """Interpolate from knot points to dense trajectory using scipy splines.

        Args:
            knot_times: Time points for knots [num_knots]
            knot_points: Batch of knot points [batch_size, num_knots, dim]
            sample_times: Time points for sampling [num_samples]
            k: Spline degree (1=linear, 2=quadratic, 3=cubic)

        Returns:
            Batch of dense trajectories [batch_size, num_samples, dim]
        """
        batch_size, num_knots, dim = knot_points.shape
        num_samples = sample_times.shape[0]

        # Initialize output
        dense_trajectories = torch.zeros((batch_size, num_samples, dim), device=self.device)

        # Interpolate each batch and dimension separately
        for b in range(batch_size):
            for d in range(dim):
                dense_trajectories[b, :, d] = self.scipy_interpolate_single(
                    knot_times, knot_points[b, :, d], sample_times, k
                )

        return dense_trajectories

    def scipy_downsample_batch(self,
                               sample_times: torch.Tensor,
                               dense_trajectories: torch.Tensor,
                               knot_times: torch.Tensor,
                               k: int = 2) -> torch.Tensor:
        """Downsample dense trajectories to knot points using scipy splines.

        Args:
            sample_times: Time points for dense samples [num_samples]
            dense_trajectories: Batch of dense trajectories [batch_size, num_samples, dim]
            knot_times: Time points for knots [num_knots]
            k: Spline degree (1=linear, 2=quadratic, 3=cubic)

        Returns:
            Batch of knot points [batch_size, num_knots, dim]
        """
        batch_size, num_samples, dim = dense_trajectories.shape
        num_knots = knot_times.shape[0]

        # Initialize output
        knot_points = torch.zeros((batch_size, num_knots, dim), device=self.device)

        # Downsample each batch and dimension separately
        for b in range(batch_size):
            for d in range(dim):
                knot_points[b, :, d] = self.scipy_interpolate_single(
                    sample_times, dense_trajectories[b, :, d], knot_times, k
                )

        return knot_points

    def compute_basis_mask_matrix(self, num_knots: int, num_samples: int) -> torch.Tensor:
        """Compute the basis mask matrix Φ that maps from knot space to dense trajectory space.

        Args:
            num_knots: Number of knot points in the trajectory
            num_samples: Number of sample points in the dense trajectory

        Returns:
            Basis mask matrix Φ [num_samples, num_knots]
        """
        # Create time grids for this specific configuration
        knot_times = torch.linspace(0, 1, num_knots, device=self.device)
        sample_times = torch.linspace(0, 1, num_samples, device=self.device)

        # Initialize the basis mask matrix
        phi = torch.zeros((num_samples, num_knots), device=self.device)

        # For each sample point, compute weights from all knot points
        for i in range(num_samples):
            # Create a unit impulse at each knot point and interpolate
            for j in range(num_knots):
                unit_impulse = torch.zeros(num_knots, device=self.device)
                unit_impulse[j] = 1.0

                # Interpolate this impulse to get the influence at sample point i
                interpolated = self.scipy_interpolate_single(
                    knot_times, unit_impulse, sample_times[i:i + 1], self.spline_degree
                )
                phi[i, j] = interpolated[0]

        return phi

    def evaluate_spline(self, knot_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate spline at parameter t.

        Args:
            knot_points: Control points [num_knots, dim]
            t: Parameter value or batch of parameter values in range [0, 1]

        Returns:
            Position(s) along the spline
        """
        num_knots, dim = knot_points.shape

        # Create normalized time grid for knots
        knot_times_norm = torch.linspace(0, 1, num_knots, device=self.device)

        # Handle batched t values
        if t.dim() > 0:
            # Convert parameter t to time values
            t_times = t  # Assume t is already in [0,1] range
            positions = torch.zeros((t.shape[0], dim), device=self.device)

            for d in range(dim):
                positions[:, d] = self.scipy_interpolate_single(
                    knot_times_norm, knot_points[:, d], t_times, self.spline_degree
                )
            return positions
        else:
            # Single t value
            t_tensor = torch.tensor([t], device=self.device)
            position = torch.zeros(dim, device=self.device)

            for d in range(dim):
                result = self.scipy_interpolate_single(
                    knot_times_norm, knot_points[:, d], t_tensor, self.spline_degree
                )
                position[d] = result[0]
            return position

    def interpolate_batch(self, knot_points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Interpolate from knot points to dense trajectory points for a batch.

        Args:
            knot_points: Batch of knot points [batch_size, num_knots, dim]
            num_samples: Number of samples in the output trajectory

        Returns:
            Batch of dense trajectories [batch_size, num_samples, dim]
        """
        batch_size, num_knots, dim = knot_points.shape

        # Create time grids
        knot_times = torch.linspace(0, 1, num_knots, device=self.device)
        sample_times = torch.linspace(0, 1, num_samples, device=self.device)

        return self.scipy_interpolate_batch(knot_times, knot_points, sample_times, self.spline_degree)

    def node2dense(self, nodes: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert control nodes to dense control sequence.

        Args:
            nodes: Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]

        Returns:
            Dense control sequence [horizon_samples+1, action_dim] or batch [batch_size, horizon_samples+1, action_dim]
        """
        # Handle list input
        if isinstance(nodes, list):
            nodes = torch.stack(nodes)

        # Handle single trajectory vs batch
        if nodes.dim() == 2:
            # Single trajectory [horizon_nodes, action_dim]
            return self.scipy_interpolate_batch(
                self.knot_times, nodes.unsqueeze(0), self.sample_times, self.spline_degree
            ).squeeze(0)
        elif nodes.dim() == 3:
            # Batch of trajectories [batch_size, horizon_nodes, action_dim]
            return self.scipy_interpolate_batch(
                self.knot_times, nodes, self.sample_times, self.spline_degree
            )
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {nodes.dim()}D")

    def dense2node(self, dense: Union[torch.Tensor, list]) -> torch.Tensor:
        """Convert dense control sequence to control nodes.

        Args:
            dense: Dense control sequence [horizon_samples+1, action_dim] or batch [batch_size, horizon_samples+1, action_dim]

        Returns:
            Control nodes [horizon_nodes, action_dim] or batch [batch_size, horizon_nodes, action_dim]
        """
        # Handle list input
        if isinstance(dense, list):
            dense = torch.stack(dense)

        # Handle single trajectory vs batch
        if dense.dim() == 2:
            # Single trajectory [num_samples, action_dim]
            return self.scipy_downsample_batch(
                self.sample_times, dense.unsqueeze(0), self.knot_times, self.spline_degree
            ).squeeze(0)
        elif dense.dim() == 3:
            # Batch of trajectories [batch_size, num_samples, action_dim]
            return self.scipy_downsample_batch(
                self.sample_times, dense, self.knot_times, self.spline_degree
            )
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {dense.dim()}D")

    def evaluate_at_time(self, nodes: torch.Tensor, t: float) -> torch.Tensor:
        """Evaluate trajectory at a specific time.

        Args:
            nodes: Control nodes [num_knots, action_dim]
            t: Time to evaluate at

        Returns:
            Control values at time t [action_dim]
        """
        time_tensor = torch.tensor([t], device=self.device)
        result = self.scipy_interpolate_batch(
            self.knot_times, nodes.unsqueeze(0), time_tensor, self.spline_degree
        )
        return result.squeeze(0).squeeze(0)

    def get_derivatives(self, nodes: torch.Tensor, t: float, order: int = 1) -> torch.Tensor:
        """Get derivatives of the trajectory at a specific time.

        Args:
            nodes: Control nodes [num_knots, action_dim]
            t: Time to evaluate at
            order: Order of derivative (1=velocity, 2=acceleration)

        Returns:
            Derivative values at time t [action_dim]
        """
        action_dim = nodes.shape[1]
        derivatives = torch.zeros(action_dim, device=self.device)

        # Convert to numpy for scipy
        knot_times_np = self.knot_times.cpu().numpy()

        for d in range(action_dim):
            knot_values_np = nodes[:, d].cpu().numpy()

            # Create spline and get derivative
            spline = InterpolatedUnivariateSpline(knot_times_np, knot_values_np, k=self.spline_degree)
            derivative_np = spline.derivative(order)(t)

            derivatives[d] = torch.from_numpy(np.array(derivative_np)).to(self.device)

        return derivatives

    def shift_trajectory(self, nodes: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Shift trajectory by n time steps.

        Args:
            nodes: Control nodes [num_knots, action_dim]
            n_steps: Number of steps to shift

        Returns:
            Shifted trajectory nodes [num_knots, action_dim]
        """
        # Convert to dense, shift, and convert back
        dense = self.node2dense(nodes)
        dense_shifted = torch.roll(dense, -n_steps, dims=0)

        # Zero out the last n_steps
        dense_shifted[-n_steps:] = 0.0

        return self.dense2node(dense_shifted)

    def concatenate_trajectories(self,
                                 traj1: torch.Tensor,
                                 traj2: torch.Tensor,
                                 blend_steps: int = 5) -> torch.Tensor:
        """Concatenate two trajectories with smooth blending.

        Args:
            traj1: First trajectory nodes [num_knots, action_dim]
            traj2: Second trajectory nodes [num_knots, action_dim]
            blend_steps: Number of steps for blending

        Returns:
            Concatenated trajectory [2*num_knots-blend_steps, action_dim]
        """
        # Convert both to dense
        dense1 = self.node2dense(traj1)
        dense2 = self.node2dense(traj2)

        # Create blending weights
        blend_weights = torch.linspace(0, 1, blend_steps, device=self.device)

        # Blend the overlapping region
        blended_region = (1 - blend_weights[:, None]) * dense1[-blend_steps:] + \
            blend_weights[:, None] * dense2[:blend_steps]

        # Concatenate
        concatenated = torch.cat([
            dense1[:-blend_steps],
            blended_region,
            dense2[blend_steps:]
        ], dim=0)

        # Convert back to nodes with appropriate number of knots
        new_length = concatenated.shape[0]
        new_knot_times = torch.linspace(0, self.dt * (new_length - 1), self.num_knots, device=self.device)
        new_sample_times = torch.linspace(0, self.dt * (new_length - 1), new_length, device=self.device)

        return self.scipy_downsample_batch(
            new_sample_times, concatenated.unsqueeze(0), new_knot_times, self.spline_degree
        ).squeeze(0)


# JAX implementation (reimplemented according to dial_core.py pattern)
try:
    import jax
    import jax.numpy as jnp
    from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
    import functools
    JAX_AVAILABLE = True

    class InterpolatedSplineJAX(SplineBase):
        """JAX-based spline interpolator following dial_core.py implementation pattern.

        This class provides JAX-accelerated spline interpolation using the same
        vectorization pattern as MBDPI in dial_core.py, while maintaining
        compatibility with the SplineBase interface.
        """

        def __init__(self, horizon_nodes: int, horizon_samples: int, dt: float = 0.02, device: Optional[torch.device] = None):
            """Initialize the JAX spline interpolator.

            Args:
                horizon_nodes: Number of control nodes (Hnode)
                horizon_samples: Number of sample points (Hsample)
                dt: Time step (ctrl_dt)
                device: Device for tensor operations (for PyTorch compatibility)
            """
            super().__init__(device)
            self.horizon_nodes = horizon_nodes
            self.horizon_samples = horizon_samples
            self.dt = dt

            # Time steps - following dial_core.py pattern exactly
            self.step_us = jnp.linspace(0, dt * (horizon_samples - 1), horizon_samples)
            self.step_nodes = jnp.linspace(0, dt * (horizon_samples - 1), horizon_nodes)

            # Setup vectorized functions following dial_core.py pattern
            self.node2u_vmap = jax.jit(
                jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
            )  # process (horizon, node)
            self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
            self.node2u_vvmap = jax.jit(
                jax.vmap(self.node2u_vmap, in_axes=(0))
            )  # process (batch, horizon, node)
            self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))

        @functools.partial(jax.jit, static_argnums=(0,))
        def node2u(self, nodes):
            """Convert control nodes to dense control sequence.

            This is the core JAX spline interpolation method following dial_core.py.
            """
            spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
            us = spline(self.step_us)
            return us

        @functools.partial(jax.jit, static_argnums=(0,))
        def u2node(self, us):
            """Convert dense control sequence to control nodes.

            This is the core JAX spline interpolation method following dial_core.py.
            """
            spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
            nodes = spline(self.step_nodes)
            return nodes

        def compute_basis_mask_matrix(self, num_knots: int, num_samples: int) -> torch.Tensor:
            """Compute the basis mask matrix Φ that maps from knot space to dense trajectory space."""
            # Create time grids
            knot_times = jnp.linspace(0, self.dt * (self.horizon_samples - 1), num_knots)
            sample_times = jnp.linspace(0, self.dt * (self.horizon_samples - 1), num_samples)

            # Initialize the basis mask matrix
            phi = jnp.zeros((num_samples, num_knots))

            # For each knot, compute its influence on all sample points
            def compute_basis_col(j):
                unit_impulse = jnp.zeros(num_knots)
                unit_impulse = unit_impulse.at[j].set(1.0)

                # Use JAX spline interpolation
                spline = InterpolatedUnivariateSpline(knot_times, unit_impulse, k=2)
                return spline(sample_times)

            # Vectorize over all knots
            phi_cols = jax.vmap(compute_basis_col)(jnp.arange(num_knots))
            phi = phi_cols.T  # Transpose to get [num_samples, num_knots]

            # Convert to PyTorch tensor for interface compatibility
            return torch.from_numpy(np.array(phi)).to(self.device)

        def evaluate_spline(self, knot_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Evaluate spline at parameter t."""
            # Convert PyTorch tensors to JAX arrays
            knot_points_np = knot_points.detach().cpu().numpy()
            t_np = t.detach().cpu().numpy() if hasattr(t, 'detach') else float(t)

            knot_points_jax = jnp.array(knot_points_np)
            t_jax = jnp.array(t_np)

            num_knots, dim = knot_points_jax.shape

            # Scale t to match our time steps (convert from [0,1] to actual time)
            t_scaled = t_jax * self.dt * (self.horizon_samples - 1)

            def evaluate_single_dim(knot_values):
                spline = InterpolatedUnivariateSpline(self.step_nodes, knot_values, k=2)
                return spline(t_scaled)

            if t_jax.ndim > 0:
                # Batch evaluation - evaluate each dimension
                positions = jax.vmap(evaluate_single_dim, in_axes=1, out_axes=1)(knot_points_jax)
            else:
                # Single evaluation
                positions = jax.vmap(evaluate_single_dim, in_axes=1)(knot_points_jax)

            # Convert back to PyTorch
            return torch.from_numpy(np.array(positions)).to(self.device)

        def interpolate_batch(self, knot_points: torch.Tensor, num_samples: int) -> torch.Tensor:
            """Interpolate from knot points to dense trajectory points for a batch."""
            # Convert to JAX arrays
            knot_points_np = knot_points.detach().cpu().numpy()
            knot_points_jax = jnp.array(knot_points_np)

            # Use the node2u_vvmap method for batch processing
            dense_trajectories_jax = self.node2u_vvmap(knot_points_jax)

            # Convert back to PyTorch
            return torch.from_numpy(np.array(dense_trajectories_jax)).to(self.device)

        def node2dense(self, nodes: Union[torch.Tensor, list]) -> torch.Tensor:
            """Convert control nodes to dense control sequence.

            This method uses the same vectorization pattern as dial_core.py.
            """
            # Handle list input
            if isinstance(nodes, list):
                nodes = torch.stack(nodes)

            # Convert to JAX
            nodes_np = nodes.detach().cpu().numpy()
            nodes_jax = jnp.array(nodes_np)

            if nodes.dim() == 2:
                # Single trajectory [num_knots, action_dim]
                # Use node2u_vmap which processes (horizon, node) -> (horizon, sample)
                dense_jax = self.node2u_vmap(nodes_jax)
            elif nodes.dim() == 3:
                # Batch of trajectories [batch_size, num_knots, action_dim]
                # Use node2u_vvmap which processes (batch, horizon, node) -> (batch, horizon, sample)
                dense_jax = self.node2u_vvmap(nodes_jax)
            else:
                raise ValueError(f"Expected 2D or 3D tensor, got {nodes.dim()}D")

            # Convert back to PyTorch
            return torch.from_numpy(np.array(dense_jax)).to(self.device)

        def dense2node(self, dense: Union[torch.Tensor, list]) -> torch.Tensor:
            """Convert dense control sequence to control nodes.

            This method uses the same vectorization pattern as dial_core.py.
            """
            # Handle list input
            if isinstance(dense, list):
                dense = torch.stack(dense)

            # Convert to JAX
            dense_np = dense.detach().cpu().numpy()
            dense_jax = jnp.array(dense_np)

            if dense.dim() == 2:
                # Single trajectory [num_samples, action_dim]
                # Use u2node_vmap which processes (horizon, sample) -> (horizon, node)
                nodes_jax = self.u2node_vmap(dense_jax)
            elif dense.dim() == 3:
                # Batch of trajectories [batch_size, num_samples, action_dim]
                # Use u2node_vvmap which processes (batch, horizon, sample) -> (batch, horizon, node)
                nodes_jax = self.u2node_vvmap(dense_jax)
            else:
                raise ValueError(f"Expected 2D or 3D tensor, got {dense.dim()}D")

            # Convert back to PyTorch
            return torch.from_numpy(np.array(nodes_jax)).to(self.device)

except ImportError:
    print("JAX not available, skipping JAX tests")
    JAX_AVAILABLE = False

