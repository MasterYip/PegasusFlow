#!/bin/bash
"""
Installation script for traj_sampling package
"""

echo "Installing traj_sampling package..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the traj_sampling directory."
    exit 1
fi

# Install the package in development mode using modern pip
echo "Installing in development mode with modern pip..."
pip install -e . --use-pep517

# Ask user if they want JAX support (compatible with dial_mpc_jax)
read -p "Do you want to install JAX support (compatible with dial_mpc_jax)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing JAX support (same as dial_mpc_jax)..."
    pip install -e .[jax] --use-pep517
fi

# Ask user if they want development dependencies
read -p "Do you want to install development dependencies? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install -e .[dev] --use-pep517
fi

echo "Installation complete!"
echo ""
echo "You can now import the package in Python:"
echo "from traj_sampling import TrajGradSampling, UniBSpline"
echo ""
echo "To run tests:"
echo "cd test && python spline_cmp.py"
echo ""
echo "Note: This package now uses the same JAX dependencies as dial_mpc_jax"