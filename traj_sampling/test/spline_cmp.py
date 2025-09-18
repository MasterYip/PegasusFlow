#!/usr/bin/env python3
"""
Spline Comparison Test Suite

This module provides comprehensive testing and comparison of different spline implementations:
- UniBSpline: Uniform cubic B-spline implementation
- InterpolatedSpline: SciPy-based spline interpolation
- InterpolatedSplineJAX: JAX-accelerated spline interpolation

The tests focus on node2dense and dense2node transformations with visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings

from traj_sampling.spline import SplineBase, UniBSpline, InterpolatedSpline, CatmullRomSpline

# Try to import JAX implementation
try:
    from traj_sampling.spline import InterpolatedSplineJAX, JAX_AVAILABLE
except ImportError:
    JAX_AVAILABLE = False
    InterpolatedSplineJAX = None

warnings.filterwarnings('ignore', category=UserWarning)


class SplineComparisonSuite:
    """Comprehensive test suite for comparing spline implementations."""

    def __init__(self,
                 horizon_nodes: int = 21,
                 horizon_samples: int = 100,
                 action_dim: int = 3,
                 dt: float = 0.02,
                 device: Optional[torch.device] = None):
        """Initialize the comparison suite.

        Args:
            horizon_nodes: Number of control nodes
            horizon_samples: Number of sample points
            action_dim: Dimension of action space
            dt: Time step
            device: Device for computations
        """
        self.horizon_nodes = horizon_nodes
        self.horizon_samples = horizon_samples
        self.action_dim = action_dim
        self.dt = dt
        self.device = device if device is not None else torch.device('cpu')

        # Initialize spline implementations
        self.uni_bspline = UniBSpline(
            horizon_nodes=horizon_nodes,
            horizon_samples=horizon_samples,
            dt=dt,
            device=self.device
        )
        self.scipy_spline = CatmullRomSpline(
            horizon_nodes=horizon_nodes,
            horizon_samples=horizon_samples,
            dt=dt,
            # spline_degree=2,
            device=self.device
        )

        # Initialize JAX implementation if available
        self.jax_spline = None
        if JAX_AVAILABLE and InterpolatedSplineJAX is not None:
            self.jax_spline = InterpolatedSplineJAX(
                horizon_nodes=horizon_nodes,
                horizon_samples=horizon_samples,
                dt=dt,
                device=self.device
            )

        # Generate test data
        self.test_nodes = self._generate_test_nodes()
        self.test_dense = self._generate_test_dense()

        print(f"Spline Comparison Suite initialized:")
        print(f"  - Nodes: {horizon_nodes}, Samples: {horizon_samples}, Action dim: {action_dim}")
        print(f"  - UniBSpline: ✓")
        print(f"  - InterpolatedSpline: ✓")
        print(f"  - JAXSpline: {'✓' if JAX_AVAILABLE else '✗'}")

    def _generate_test_nodes(self) -> torch.Tensor:
        """Generate test control nodes with realistic trajectory patterns."""
        # Create a smooth trajectory with some interesting dynamics
        t = torch.linspace(0, 1, self.horizon_nodes, device=self.device)

        nodes = torch.zeros((self.horizon_nodes, self.action_dim), device=self.device)

        # Generate different patterns for each action dimension
        for d in range(self.action_dim):
            if d == 0:
                # Sinusoidal pattern
                nodes[:, d] = torch.sin(2 * np.pi * t) * 0.5
            elif d == 1:
                # Ramp with noise
                nodes[:, d] = t * 2.0 - 1.0 + 0.1 * torch.randn_like(t)
            else:
                # Step function with smoothing
                nodes[:, d] = torch.where(t < 0.5, -0.5, 0.5) + 0.05 * torch.sin(10 * np.pi * t)

        return nodes

    def _generate_test_dense(self) -> torch.Tensor:
        """Generate test dense trajectory."""
        t = torch.linspace(0, 1, self.horizon_samples, device=self.device)

        dense = torch.zeros((self.horizon_samples, self.action_dim), device=self.device)

        for d in range(self.action_dim):
            if d == 0:
                dense[:, d] = torch.cos(3 * np.pi * t) * 0.3
            elif d == 1:
                dense[:, d] = torch.sin(np.pi * t) * 0.8
            else:
                dense[:, d] = torch.tanh(5 * (t - 0.5)) * 0.6

        return dense

    def test_node2dense_accuracy(self) -> Dict[str, Dict[str, float]]:
        """Test accuracy of node2dense conversion across implementations."""
        results = {}

        # Test UniBSpline
        try:
            start_time = time.time()
            uni_dense = self.uni_bspline.node2dense(self.test_nodes)
            uni_time = time.time() - start_time

            results['UniBSpline'] = {
                'success': True,
                'time': uni_time,
                'output_shape': list(uni_dense.shape),
                'mean_value': float(torch.mean(uni_dense)),
                'std_value': float(torch.std(uni_dense))
            }
        except Exception as e:
            results['UniBSpline'] = {'success': False, 'error': str(e)}

        # Test InterpolatedSpline
        try:
            start_time = time.time()
            scipy_dense = self.scipy_spline.node2dense(self.test_nodes)
            scipy_time = time.time() - start_time

            results['InterpolatedSpline'] = {
                'success': True,
                'time': scipy_time,
                'output_shape': list(scipy_dense.shape),
                'mean_value': float(torch.mean(scipy_dense)),
                'std_value': float(torch.std(scipy_dense))
            }
        except Exception as e:
            results['InterpolatedSpline'] = {'success': False, 'error': str(e)}

        # Test JAXSpline
        if JAX_AVAILABLE and self.jax_spline is not None:
            try:
                start_time = time.time()
                jax_dense = self.jax_spline.node2dense(self.test_nodes)
                jax_time = time.time() - start_time

                results['JAXSpline'] = {
                    'success': True,
                    'time': jax_time,
                    'output_shape': list(jax_dense.shape),
                    'mean_value': float(torch.mean(jax_dense)),
                    'std_value': float(torch.std(jax_dense))
                }
            except Exception as e:
                results['JAXSpline'] = {'success': False, 'error': str(e)}

        return results

    def test_dense2node_accuracy(self) -> Dict[str, Dict[str, float]]:
        """Test accuracy of dense2node conversion across implementations."""
        results = {}

        # Test UniBSpline
        try:
            start_time = time.time()
            uni_nodes = self.uni_bspline.dense2node(self.test_dense)
            uni_time = time.time() - start_time

            results['UniBSpline'] = {
                'success': True,
                'time': uni_time,
                'output_shape': list(uni_nodes.shape),
                'mean_value': float(torch.mean(uni_nodes)),
                'std_value': float(torch.std(uni_nodes))
            }
        except Exception as e:
            results['UniBSpline'] = {'success': False, 'error': str(e)}

        # Test InterpolatedSpline
        try:
            start_time = time.time()
            scipy_nodes = self.scipy_spline.dense2node(self.test_dense)
            scipy_time = time.time() - start_time

            results['InterpolatedSpline'] = {
                'success': True,
                'time': scipy_time,
                'output_shape': list(scipy_nodes.shape),
                'mean_value': float(torch.mean(scipy_nodes)),
                'std_value': float(torch.std(scipy_nodes))
            }
        except Exception as e:
            results['InterpolatedSpline'] = {'success': False, 'error': str(e)}

        # Test JAXSpline
        if JAX_AVAILABLE and self.jax_spline is not None:
            try:
                start_time = time.time()
                jax_nodes = self.jax_spline.dense2node(self.test_dense)
                jax_time = time.time() - start_time

                results['JAXSpline'] = {
                    'success': True,
                    'time': jax_time,
                    'output_shape': list(jax_nodes.shape),
                    'mean_value': float(torch.mean(jax_nodes)),
                    'std_value': float(torch.std(jax_nodes))
                }
            except Exception as e:
                results['JAXSpline'] = {'success': False, 'error': str(e)}

        return results

    def test_round_trip_consistency(self) -> Dict[str, Dict[str, float]]:
        """Test round-trip consistency: nodes -> dense -> nodes."""
        results = {}

        # Test UniBSpline
        try:
            dense_from_nodes = self.uni_bspline.node2dense(self.test_nodes)
            nodes_recovered = self.uni_bspline.dense2node(dense_from_nodes)

            mse = float(torch.mean((nodes_recovered - self.test_nodes) ** 2))
            max_error = float(torch.max(torch.abs(nodes_recovered - self.test_nodes)))

            results['UniBSpline'] = {
                'success': True,
                'mse': mse,
                'max_error': max_error,
                'relative_error': max_error / (float(torch.max(torch.abs(self.test_nodes))) + 1e-8)
            }
        except Exception as e:
            results['UniBSpline'] = {'success': False, 'error': str(e)}

        # Test InterpolatedSpline
        try:
            dense_from_nodes = self.scipy_spline.node2dense(self.test_nodes)
            nodes_recovered = self.scipy_spline.dense2node(dense_from_nodes)

            mse = float(torch.mean((nodes_recovered - self.test_nodes) ** 2))
            max_error = float(torch.max(torch.abs(nodes_recovered - self.test_nodes)))

            results['InterpolatedSpline'] = {
                'success': True,
                'mse': mse,
                'max_error': max_error,
                'relative_error': max_error / (float(torch.max(torch.abs(self.test_nodes))) + 1e-8)
            }
        except Exception as e:
            results['InterpolatedSpline'] = {'success': False, 'error': str(e)}

        # Test JAXSpline
        if JAX_AVAILABLE and self.jax_spline is not None:
            try:
                dense_from_nodes = self.jax_spline.node2dense(self.test_nodes)
                nodes_recovered = self.jax_spline.dense2node(dense_from_nodes)

                mse = float(torch.mean((nodes_recovered - self.test_nodes) ** 2))
                max_error = float(torch.max(torch.abs(nodes_recovered - self.test_nodes)))

                results['JAXSpline'] = {
                    'success': True,
                    'mse': mse,
                    'max_error': max_error,
                    'relative_error': max_error / (float(torch.max(torch.abs(self.test_nodes))) + 1e-8)
                }
            except Exception as e:
                results['JAXSpline'] = {'success': False, 'error': str(e)}

        return results

    def test_batch_processing(self) -> Dict[str, Dict[str, float]]:
        """Test batch processing capabilities."""
        batch_size = 5
        batch_nodes = self.test_nodes.unsqueeze(0).repeat(batch_size, 1, 1)
        batch_dense = self.test_dense.unsqueeze(0).repeat(batch_size, 1, 1)

        results = {}

        # Test UniBSpline
        try:
            start_time = time.time()
            uni_batch_dense = self.uni_bspline.node2dense(batch_nodes)
            uni_batch_nodes = self.uni_bspline.dense2node(batch_dense)
            uni_time = time.time() - start_time

            results['UniBSpline'] = {
                'success': True,
                'time': uni_time,
                'batch_dense_shape': list(uni_batch_dense.shape),
                'batch_nodes_shape': list(uni_batch_nodes.shape)
            }
        except Exception as e:
            results['UniBSpline'] = {'success': False, 'error': str(e)}

        # Test InterpolatedSpline
        try:
            start_time = time.time()
            scipy_batch_dense = self.scipy_spline.node2dense(batch_nodes)
            scipy_batch_nodes = self.scipy_spline.dense2node(batch_dense)
            scipy_time = time.time() - start_time

            results['InterpolatedSpline'] = {
                'success': True,
                'time': scipy_time,
                'batch_dense_shape': list(scipy_batch_dense.shape),
                'batch_nodes_shape': list(scipy_batch_nodes.shape)
            }
        except Exception as e:
            results['InterpolatedSpline'] = {'success': False, 'error': str(e)}

        # Test JAXSpline
        if JAX_AVAILABLE and self.jax_spline is not None:
            try:
                start_time = time.time()
                jax_batch_dense = self.jax_spline.node2dense(batch_nodes)
                jax_batch_nodes = self.jax_spline.dense2node(batch_dense)
                jax_time = time.time() - start_time

                results['JAXSpline'] = {
                    'success': True,
                    'time': jax_time,
                    'batch_dense_shape': list(jax_batch_dense.shape),
                    'batch_nodes_shape': list(jax_batch_nodes.shape)
                }
            except Exception as e:
                results['JAXSpline'] = {'success': False, 'error': str(e)}

        return results

    def plot_node2dense_comparison(self, save_path: Optional[str] = None):
        """Plot comparison of node2dense results across implementations."""
        fig, axes = plt.subplots(self.action_dim, 1, figsize=(12, 4 * self.action_dim))
        if self.action_dim == 1:
            axes = [axes]

        # Time arrays for plotting
        node_times = torch.linspace(0, self.dt * (self.horizon_samples - 1), self.horizon_nodes, device=self.device)
        sample_times = torch.linspace(0, self.dt * (self.horizon_samples - 1), self.horizon_samples, device=self.device)

        for d in range(self.action_dim):
            ax = axes[d]

            # Plot original nodes
            ax.plot(node_times.cpu().numpy(), self.test_nodes[:, d].cpu().numpy(),
                    'ko-', markersize=6, linewidth=2, label='Original Nodes')

            # Plot UniBSpline result
            try:
                uni_dense = self.uni_bspline.node2dense(self.test_nodes)
                if uni_dense.shape[0] == self.horizon_samples:
                    ax.plot(sample_times.cpu().numpy(), uni_dense[:, d].cpu().numpy(),
                            'b-', linewidth=2, alpha=0.8, label='UniBSpline')
                else:
                    raise ValueError("UniBSpline output size mismatch")

            except Exception as e:
                ax.text(0.5, 0.5, f'UniBSpline Error: {str(e)[:50]}...',
                        transform=ax.transAxes, ha='center', va='center', color='red')

            # Plot InterpolatedSpline result
            try:
                scipy_dense = self.scipy_spline.node2dense(self.test_nodes)
                ax.plot(sample_times.cpu().numpy(), scipy_dense[:, d].cpu().numpy(),
                        'r--', linewidth=2, alpha=0.8, label='InterpolatedSpline')
            except Exception as e:
                ax.text(0.5, 0.4, f'InterpolatedSpline Error: {str(e)[:50]}...',
                        transform=ax.transAxes, ha='center', va='center', color='red')

            # Plot JAXSpline result
            if JAX_AVAILABLE and self.jax_spline is not None:
                try:
                    jax_dense = self.jax_spline.node2dense(self.test_nodes)
                    if jax_dense.shape[0] == self.horizon_samples:
                        ax.plot(sample_times.cpu().numpy(), jax_dense[:, d].cpu().numpy(),
                                'g:', linewidth=3, alpha=0.8, label='JAXSpline')
                    else:
                        raise ValueError("JAXSpline output size mismatch")
                except Exception as e:
                    ax.text(0.5, 0.3, f'JAXSpline Error: {str(e)[:50]}...',
                            transform=ax.transAxes, ha='center', va='center', color='red')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Action Dim {d}')
            ax.set_title(f'Node2Dense Comparison - Dimension {d}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_dense2node_comparison(self, save_path: Optional[str] = None):
        """Plot comparison of dense2node results across implementations."""
        fig, axes = plt.subplots(self.action_dim, 1, figsize=(12, 4 * self.action_dim))
        if self.action_dim == 1:
            axes = [axes]

        # Time arrays for plotting
        dense_times = torch.linspace(0, self.dt * (self.horizon_samples - 1), self.horizon_samples, device=self.device)
        node_times = torch.linspace(0, self.dt * (self.horizon_samples - 1), self.horizon_nodes, device=self.device)

        for d in range(self.action_dim):
            ax = axes[d]

            # Plot original dense trajectory
            ax.plot(dense_times.cpu().numpy(), self.test_dense[:, d].cpu().numpy(),
                    'k-', linewidth=2, alpha=0.7, label='Original Dense')

            # Plot UniBSpline result
            try:
                uni_nodes = self.uni_bspline.dense2node(self.test_dense)
                if uni_nodes.shape[0] == self.horizon_nodes:
                    ax.plot(node_times.cpu().numpy(), uni_nodes[:, d].cpu().numpy(),
                            'bo-', markersize=6, linewidth=2, label='UniBSpline Nodes')
                else:
                    raise ValueError("UniBSpline output size mismatch")
            except Exception as e:
                ax.text(0.5, 0.5, f'UniBSpline Error: {str(e)[:50]}...',
                        transform=ax.transAxes, ha='center', va='center', color='red')

            # Plot InterpolatedSpline result
            try:
                scipy_nodes = self.scipy_spline.dense2node(self.test_dense)
                ax.plot(node_times.cpu().numpy(), scipy_nodes[:, d].cpu().numpy(),
                        'rs-', markersize=6, linewidth=2, label='InterpolatedSpline Nodes')
            except Exception as e:
                ax.text(0.5, 0.4, f'InterpolatedSpline Error: {str(e)[:50]}...',
                        transform=ax.transAxes, ha='center', va='center', color='red')

            # Plot JAXSpline result
            if JAX_AVAILABLE and self.jax_spline is not None:
                try:
                    jax_nodes = self.jax_spline.dense2node(self.test_dense)
                    if jax_nodes.shape[0] == self.horizon_nodes:
                        ax.plot(node_times.cpu().numpy(), jax_nodes[:, d].cpu().numpy(),
                                'g^-', markersize=6, linewidth=2, label='JAXSpline Nodes')
                    else:
                        raise ValueError("JAXSpline output size mismatch")
                except Exception as e:
                    ax.text(0.5, 0.3, f'JAXSpline Error: {str(e)[:50]}...',
                            transform=ax.transAxes, ha='center', va='center', color='red')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Action Dim {d}')
            ax.set_title(f'Dense2Node Comparison - Dimension {d}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_round_trip_analysis(self, save_path: Optional[str] = None):
        """Plot round-trip error analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        implementations = ['UniBSpline', 'InterpolatedSpline']
        if JAX_AVAILABLE and self.jax_spline is not None:
            implementations.append('JAXSpline')

        colors = ['blue', 'red', 'green']

        # Round-trip: nodes -> dense -> nodes
        for i, impl_name in enumerate(implementations):
            if impl_name == 'UniBSpline':
                impl = self.uni_bspline
            elif impl_name == 'InterpolatedSpline':
                impl = self.scipy_spline
            elif impl_name == 'JAXSpline':
                impl = self.jax_spline

            try:
                # Forward and back
                dense_from_nodes = impl.node2dense(self.test_nodes)
                nodes_recovered = impl.dense2node(dense_from_nodes)

                # Calculate errors
                if nodes_recovered.shape == self.test_nodes.shape:
                    error = torch.abs(nodes_recovered - self.test_nodes)

                    # Plot error heatmap
                    ax = axes[0, 0]
                    im = ax.imshow(error.cpu().numpy().T, aspect='auto', cmap='hot')
                    ax.set_title('Round-trip Error: Nodes->Dense->Nodes')
                    ax.set_xlabel('Node Index')
                    ax.set_ylabel('Action Dimension')
                    plt.colorbar(im, ax=ax)

                    # Plot error by dimension
                    ax = axes[0, 1]
                    for d in range(self.action_dim):
                        ax.plot(error[:, d].cpu().numpy(), color=colors[i % len(colors)],
                                alpha=0.7, label=f'{impl_name} Dim {d}')
                    ax.set_title('Round-trip Error by Dimension')
                    ax.set_xlabel('Node Index')
                    ax.set_ylabel('Absolute Error')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            except Exception as e:
                print(f"Round-trip test failed for {impl_name}: {e}")

        # Round-trip: dense -> nodes -> dense
        for i, impl_name in enumerate(implementations):
            if impl_name == 'UniBSpline':
                impl = self.uni_bspline
            elif impl_name == 'InterpolatedSpline':
                impl = self.scipy_spline
            elif impl_name == 'JAXSpline':
                impl = self.jax_spline

            try:
                # Forward and back
                nodes_from_dense = impl.dense2node(self.test_dense)
                dense_recovered = impl.node2dense(nodes_from_dense)

                # Calculate errors
                if dense_recovered.shape == self.test_dense.shape:
                    error = torch.abs(dense_recovered - self.test_dense)

                    # Plot error heatmap
                    ax = axes[1, 0]
                    im = ax.imshow(error.cpu().numpy().T, aspect='auto', cmap='plasma')
                    ax.set_title('Round-trip Error: Dense->Nodes->Dense')
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Action Dimension')
                    plt.colorbar(im, ax=ax)

                    # Plot error statistics
                    ax = axes[1, 1]
                    mean_error = torch.mean(error, dim=0).cpu().numpy()
                    std_error = torch.std(error, dim=0).cpu().numpy()
                    x = np.arange(self.action_dim)
                    ax.bar(x + i * 0.25, mean_error, 0.25, color=colors[i % len(colors)],
                           alpha=0.7, label=f'{impl_name} Mean')
                    ax.errorbar(x + i * 0.25, mean_error, yerr=std_error,
                                fmt='none', color='black', alpha=0.5)

            except Exception as e:
                print(f"Dense round-trip test failed for {impl_name}: {e}")

        axes[1, 1].set_title('Round-trip Error Statistics')
        axes[1, 1].set_xlabel('Action Dimension')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """Plot performance comparison across implementations."""
        # Run performance tests
        n_trials = 10
        results = {'UniBSpline': [], 'InterpolatedSpline': [], 'JAXSpline': []}

        for trial in range(n_trials):
            # UniBSpline timing
            try:
                start = time.time()
                self.uni_bspline.node2dense(self.test_nodes)
                self.uni_bspline.dense2node(self.test_dense)
                results['UniBSpline'].append(time.time() - start)
            except:
                pass

            # InterpolatedSpline timing
            try:
                start = time.time()
                self.scipy_spline.node2dense(self.test_nodes)
                self.scipy_spline.dense2node(self.test_dense)
                results['InterpolatedSpline'].append(time.time() - start)
            except:
                pass

            # JAXSpline timing
            if JAX_AVAILABLE and self.jax_spline is not None:
                try:
                    start = time.time()
                    self.jax_spline.node2dense(self.test_nodes)
                    self.jax_spline.dense2node(self.test_dense)
                    results['JAXSpline'].append(time.time() - start)
                except:
                    pass

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot of timing results
        valid_results = {k: v for k, v in results.items() if len(v) > 0}
        if valid_results:
            ax1.boxplot(valid_results.values(), labels=valid_results.keys())
            ax1.set_title('Performance Comparison (Round-trip Time)')
            ax1.set_ylabel('Time (seconds)')
            ax1.grid(True, alpha=0.3)

        # Bar plot of mean times
        means = {k: np.mean(v) if len(v) > 0 else 0 for k, v in results.items()}
        stds = {k: np.std(v) if len(v) > 0 else 0 for k, v in results.items()}

        x = np.arange(len(means))
        bars = ax2.bar(x, list(means.values()), yerr=list(stds.values()),
                       capsize=5, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(list(means.keys()))
        ax2.set_title('Mean Performance Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)

        # Color bars
        colors = ['blue', 'red', 'green']
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print("Running comprehensive spline comparison tests...")

        results = {
            'node2dense_accuracy': self.test_node2dense_accuracy(),
            'dense2node_accuracy': self.test_dense2node_accuracy(),
            'round_trip_consistency': self.test_round_trip_consistency(),
            'batch_processing': self.test_batch_processing()
        }

        # Print summary
        print("\n" + "=" * 60)
        print("SPLINE COMPARISON TEST RESULTS")
        print("=" * 60)

        for test_name, test_results in results.items():
            print(f"\n{test_name.upper()}:")
            for impl_name, impl_results in test_results.items():
                if impl_results.get('success', False):
                    print(f"  {impl_name}: ✓")
                    if 'time' in impl_results:
                        print(f"    Time: {impl_results['time']:.4f}s")
                    if 'mse' in impl_results:
                        print(f"    MSE: {impl_results['mse']:.6f}")
                    if 'max_error' in impl_results:
                        print(f"    Max Error: {impl_results['max_error']:.6f}")
                else:
                    print(f"  {impl_name}: ✗ ({impl_results.get('error', 'Unknown error')})")

        return results

    def plot_3d_node2dense_comparison(self, save_path: Optional[str] = None):
        """Create 3D visualization of node2dense transformations."""
        if self.action_dim < 3:
            print("3D plotting requires at least 3 action dimensions. Skipping 3D plot.")
            return

        fig = plt.figure(figsize=(18, 6))

        # Time arrays for plotting
        node_times = torch.linspace(0, self.dt * (self.horizon_samples - 1), self.horizon_nodes, device=self.device).cpu().numpy()
        sample_times = torch.linspace(0, self.dt * (self.horizon_samples - 1),
                                      self.horizon_samples, device=self.device).cpu().numpy()

        # Original nodes trajectory in 3D
        ax1 = fig.add_subplot(131, projection='3d')
        nodes_np = self.test_nodes[:, :3].cpu().numpy()
        ax1.plot(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                 'ko-', markersize=8, linewidth=3, label='Original Nodes')
        ax1.scatter(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                    c=node_times, cmap='viridis', s=60, alpha=0.8)
        ax1.set_xlabel('Action Dim 0')
        ax1.set_ylabel('Action Dim 1')
        ax1.set_zlabel('Action Dim 2')
        ax1.set_title('Original Control Nodes\n(Time-colored)')
        ax1.legend()

        # UniBSpline dense trajectory in 3D
        ax2 = fig.add_subplot(132, projection='3d')
        try:
            uni_dense = self.uni_bspline.node2dense(self.test_nodes)
            if uni_dense.shape[0] == self.horizon_samples:
                dense_np = uni_dense[:, :3].cpu().numpy()
                ax2.plot(dense_np[:, 0], dense_np[:, 1], dense_np[:, 2],
                         'b-', linewidth=2, alpha=0.8, label='UniBSpline Dense')
                ax2.scatter(dense_np[::5, 0], dense_np[::5, 1], dense_np[::5, 2],
                            c=sample_times[::5], cmap='plasma', s=30, alpha=0.7)
                # Also plot original nodes for reference
                ax2.plot(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                         'ko-', markersize=6, linewidth=2, alpha=0.6, label='Original Nodes')
                ax2.set_title('UniBSpline: Nodes → Dense')
            else:
                ax2.text(0.5, 0.5, 0.5, 'Size Mismatch', transform=ax2.transAxes, ha='center')
                ax2.set_title('UniBSpline: Error')
        except Exception as e:
            ax2.text(0.5, 0.5, 0.5, f'Error: {str(e)[:30]}', transform=ax2.transAxes, ha='center')
            ax2.set_title('UniBSpline: Error')

        ax2.set_xlabel('Action Dim 0')
        ax2.set_ylabel('Action Dim 1')
        ax2.set_zlabel('Action Dim 2')
        ax2.legend()

        # SciPy spline dense trajectory in 3D
        ax3 = fig.add_subplot(133, projection='3d')
        try:
            scipy_dense = self.scipy_spline.node2dense(self.test_nodes)
            dense_np = scipy_dense[:, :3].cpu().numpy()
            ax3.plot(dense_np[:, 0], dense_np[:, 1], dense_np[:, 2],
                     'r-', linewidth=2, alpha=0.8, label='InterpolatedSpline Dense')
            ax3.scatter(dense_np[::5, 0], dense_np[::5, 1], dense_np[::5, 2],
                        c=sample_times[::5], cmap='coolwarm', s=30, alpha=0.7)
            # Also plot original nodes for reference
            ax3.plot(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                     'ko-', markersize=6, linewidth=2, alpha=0.6, label='Original Nodes')
            ax3.set_title('InterpolatedSpline: Nodes → Dense')
        except Exception as e:
            ax3.text(0.5, 0.5, 0.5, f'Error: {str(e)[:30]}', transform=ax3.transAxes, ha='center')
            ax3.set_title('InterpolatedSpline: Error')

        ax3.set_xlabel('Action Dim 0')
        ax3.set_ylabel('Action Dim 1')
        ax3.set_zlabel('Action Dim 2')
        ax3.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_3d_dense2node_comparison(self, save_path: Optional[str] = None):
        """Create 3D visualization of dense2node transformations."""
        if self.action_dim < 3:
            print("3D plotting requires at least 3 action dimensions. Skipping 3D plot.")
            return

        fig = plt.figure(figsize=(18, 6))

        # Time arrays for plotting
        dense_times = torch.linspace(0, self.dt * (self.horizon_samples - 1),
                                     self.horizon_samples, device=self.device).cpu().numpy()
        node_times = torch.linspace(0, self.dt * (self.horizon_samples - 1), self.horizon_nodes, device=self.device).cpu().numpy()

        # Original dense trajectory in 3D
        ax1 = fig.add_subplot(131, projection='3d')
        dense_np = self.test_dense[:, :3].cpu().numpy()
        ax1.plot(dense_np[:, 0], dense_np[:, 1], dense_np[:, 2],
                 'k-', linewidth=2, alpha=0.8, label='Original Dense')
        ax1.scatter(dense_np[::5, 0], dense_np[::5, 1], dense_np[::5, 2],
                    c=dense_times[::5], cmap='viridis', s=30, alpha=0.8)
        ax1.set_xlabel('Action Dim 0')
        ax1.set_ylabel('Action Dim 1')
        ax1.set_zlabel('Action Dim 2')
        ax1.set_title('Original Dense Trajectory\n(Time-colored)')
        ax1.legend()

        # UniBSpline nodes from dense in 3D
        ax2 = fig.add_subplot(132, projection='3d')
        try:
            uni_nodes = self.uni_bspline.dense2node(self.test_dense)
            if uni_nodes.shape[0] == self.horizon_nodes:
                nodes_np = uni_nodes[:, :3].cpu().numpy()
                ax2.plot(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                         'bo-', markersize=8, linewidth=3, label='UniBSpline Nodes')
                ax2.scatter(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                            c=node_times, cmap='plasma', s=80, alpha=0.8)
                # Also plot original dense for reference (subsampled)
                ax2.plot(dense_np[::5, 0], dense_np[::5, 1], dense_np[::5, 2],
                         'k-', linewidth=1, alpha=0.4, label='Original Dense (subsampled)')
                ax2.set_title('UniBSpline: Dense → Nodes')
            else:
                ax2.text(0.5, 0.5, 0.5, 'Size Mismatch', transform=ax2.transAxes, ha='center')
                ax2.set_title('UniBSpline: Error')
        except Exception as e:
            ax2.text(0.5, 0.5, 0.5, f'Error: {str(e)[:30]}', transform=ax2.transAxes, ha='center')
            ax2.set_title('UniBSpline: Error')

        ax2.set_xlabel('Action Dim 0')
        ax2.set_ylabel('Action Dim 1')
        ax2.set_zlabel('Action Dim 2')
        ax2.legend()

        # SciPy spline nodes from dense in 3D
        ax3 = fig.add_subplot(133, projection='3d')
        try:
            scipy_nodes = self.scipy_spline.dense2node(self.test_dense)
            nodes_np = scipy_nodes[:, :3].cpu().numpy()
            ax3.plot(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                     'rs-', markersize=8, linewidth=3, label='InterpolatedSpline Nodes')
            ax3.scatter(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                        c=node_times, cmap='coolwarm', s=80, alpha=0.8)
            # Also plot original dense for reference (subsampled)
            ax3.plot(dense_np[::5, 0], dense_np[::5, 1], dense_np[::5, 2],
                     'k-', linewidth=1, alpha=0.4, label='Original Dense (subsampled)')
            ax3.set_title('InterpolatedSpline: Dense → Nodes')
        except Exception as e:
            ax3.text(0.5, 0.5, 0.5, f'Error: {str(e)[:30]}', transform=ax3.transAxes, ha='center')
            ax3.set_title('InterpolatedSpline: Error')

        ax3.set_xlabel('Action Dim 0')
        ax3.set_ylabel('Action Dim 1')
        ax3.set_zlabel('Action Dim 2')
        ax3.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_3d_round_trip_comparison(self, save_path: Optional[str] = None):
        """Create 3D visualization showing round-trip transformations."""
        if self.action_dim < 3:
            print("3D plotting requires at least 3 action dimensions. Skipping 3D plot.")
            return

        fig = plt.figure(figsize=(20, 12))

        # Time arrays for plotting
        node_times = torch.linspace(0, self.dt * (self.horizon_samples - 1), self.horizon_nodes, device=self.device).cpu().numpy()
        sample_times = torch.linspace(0, self.dt * (self.horizon_samples - 1),
                                      self.horizon_samples, device=self.device).cpu().numpy()

        implementations = ['UniBSpline', 'InterpolatedSpline']
        if JAX_AVAILABLE and self.jax_spline is not None:
            implementations.append('JAXSpline')

        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']

        # Round-trip 1: Nodes → Dense → Nodes
        for i, impl_name in enumerate(implementations):
            if impl_name == 'UniBSpline':
                impl = self.uni_bspline
            elif impl_name == 'InterpolatedSpline':
                impl = self.scipy_spline
            elif impl_name == 'JAXSpline':
                impl = self.jax_spline

            ax = fig.add_subplot(2, len(implementations), i + 1, projection='3d')

            try:
                # Original nodes
                nodes_np = self.test_nodes[:, :3].cpu().numpy()
                ax.plot(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                        'k-', linewidth=3, alpha=0.8, label='Original Nodes')
                ax.scatter(nodes_np[:, 0], nodes_np[:, 1], nodes_np[:, 2],
                           c='black', s=80, alpha=0.8, marker='o')

                # Round-trip: nodes -> dense -> nodes
                dense_from_nodes = impl.node2dense(self.test_nodes)
                nodes_recovered = impl.dense2node(dense_from_nodes)

                if nodes_recovered.shape == self.test_nodes.shape:
                    recovered_np = nodes_recovered[:, :3].cpu().numpy()
                    ax.plot(recovered_np[:, 0], recovered_np[:, 1], recovered_np[:, 2],
                            f'{colors[i]}-', linewidth=2, alpha=0.7, label=f'{impl_name} Recovered')
                    ax.scatter(recovered_np[:, 0], recovered_np[:, 1], recovered_np[:, 2],
                               c=colors[i], s=60, alpha=0.7, marker=markers[i])

                    # Calculate and show error vectors
                    error_vectors = recovered_np - nodes_np
                    for j in range(0, len(nodes_np), 2):  # Show every other error vector to avoid clutter
                        ax.quiver(nodes_np[j, 0], nodes_np[j, 1], nodes_np[j, 2],
                                  error_vectors[j, 0], error_vectors[j, 1], error_vectors[j, 2],
                                  color=colors[i], alpha=0.5, arrow_length_ratio=0.1)

                    # Calculate error statistics
                    mse = float(torch.mean((nodes_recovered - self.test_nodes) ** 2))
                    max_error = float(torch.max(torch.abs(nodes_recovered - self.test_nodes)))
                    ax.set_title(f'{impl_name}: Nodes→Dense→Nodes\nMSE: {mse:.4f}, Max Err: {max_error:.4f}')
                else:
                    ax.set_title(f'{impl_name}: Shape Mismatch')

            except Exception as e:
                ax.text(0.5, 0.5, 0.5, f'Error: {str(e)[:30]}', transform=ax.transAxes, ha='center')
                ax.set_title(f'{impl_name}: Error')

            ax.set_xlabel('Action Dim 0')
            ax.set_ylabel('Action Dim 1')
            ax.set_zlabel('Action Dim 2')
            ax.legend()

        # Round-trip 2: Dense → Nodes → Dense
        for i, impl_name in enumerate(implementations):
            if impl_name == 'UniBSpline':
                impl = self.uni_bspline
            elif impl_name == 'InterpolatedSpline':
                impl = self.scipy_spline
            elif impl_name == 'JAXSpline':
                impl = self.jax_spline

            ax = fig.add_subplot(2, len(implementations), len(implementations) + i + 1, projection='3d')

            try:
                # Original dense trajectory
                dense_np = self.test_dense[:, :3].cpu().numpy()
                ax.plot(dense_np[:, 0], dense_np[:, 1], dense_np[:, 2],
                        'k-', linewidth=2, alpha=0.8, label='Original Dense')
                ax.scatter(dense_np[::10, 0], dense_np[::10, 1], dense_np[::10, 2],
                           c='black', s=30, alpha=0.6)

                # Round-trip: dense -> nodes -> dense
                nodes_from_dense = impl.dense2node(self.test_dense)
                dense_recovered = impl.node2dense(nodes_from_dense)

                if dense_recovered.shape == self.test_dense.shape:
                    recovered_np = dense_recovered[:, :3].cpu().numpy()
                    ax.plot(recovered_np[:, 0], recovered_np[:, 1], recovered_np[:, 2],
                            f'{colors[i]}-', linewidth=2, alpha=0.7, label=f'{impl_name} Recovered')
                    ax.scatter(recovered_np[::10, 0], recovered_np[::10, 1], recovered_np[::10, 2],
                               c=colors[i], s=30, alpha=0.7, marker=markers[i])

                    # Calculate error statistics
                    mse = float(torch.mean((dense_recovered - self.test_dense) ** 2))
                    max_error = float(torch.max(torch.abs(dense_recovered - self.test_dense)))
                    ax.set_title(f'{impl_name}: Dense→Nodes→Dense\nMSE: {mse:.4f}, Max Err: {max_error:.4f}')
                else:
                    ax.set_title(f'{impl_name}: Shape Mismatch')

            except Exception as e:
                ax.text(0.5, 0.5, 0.5, f'Error: {str(e)[:30]}', transform=ax.transAxes, ha='center')
                ax.set_title(f'{impl_name}: Error')

            ax.set_xlabel('Action Dim 0')
            ax.set_ylabel('Action Dim 1')
            ax.set_zlabel('Action Dim 2')
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run the spline comparison tests."""
    print("Spline Implementation Comparison Test Suite")
    print("=" * 50)

    # Initialize test suite
    suite = SplineComparisonSuite(
        horizon_nodes=4 + 1,
        horizon_samples=64 + 1,
        action_dim=3,
        dt=0.02
    )

    # Run all tests
    results = suite.run_all_tests()

    # Generate visualizations
    print("\nGenerating visualizations...")
    suite.plot_node2dense_comparison(save_path='node2dense_comparison.png')
    suite.plot_dense2node_comparison(save_path='dense2node_comparison.png')
    suite.plot_round_trip_analysis(save_path='round_trip_analysis.png')
    suite.plot_performance_comparison(save_path='performance_comparison.png')
    suite.plot_3d_node2dense_comparison(save_path='3d_node2dense_comparison.png')
    suite.plot_3d_dense2node_comparison(save_path='3d_dense2node_comparison.png')
    suite.plot_3d_round_trip_comparison(save_path='3d_round_trip_comparison.png')

    print("\nTest suite completed! Check the generated plots for detailed comparisons.")
    return results


if __name__ == "__main__":
    main()
