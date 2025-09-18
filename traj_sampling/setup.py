#!/usr/bin/env python3
"""
Setup script for traj_sampling package.

This package provides trajectory optimization capabilities including:
- Trajectory gradient sampling and optimization
- Weighted Basis Function Optimization (WBFO)
- Spline interpolation utilities
- Compatible with Python 3.10+ and multiple projects
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = __doc__

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    try:
        with open(os.path.join(this_directory, filename), 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

# Core requirements - aligned with dial_mpc_jax
install_requires = [
    'numpy<2.0.0',  # Same constraint as dial_mpc_jax
    'torch>=1.9.0',
    'scipy>=1.7.0',
    'matplotlib>=3.3.0',
]

# Optional requirements for JAX support - matching dial_mpc_jax
jax_requires = [
    'jax[cuda12]',  # Same as dial_mpc_jax
    'jax-cosmo',    # Same as dial_mpc_jax
]

# Development requirements
dev_requires = [
    'pytest>=6.0.0',
    'pytest-cov>=2.0.0',
    'black>=21.0.0',
    'flake8>=3.8.0',
    'mypy>=0.900',
]

setup(
    name='traj_sampling',
    version='1.0.0',
    author='MasterYip @ HIT',
    author_email='',
    description='Trajectory optimization and sampling utilities for robotics and control',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MasterYip/PredictiveDiffusionPlanner',
    packages=find_packages(include="traj_sampling"),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'jax': jax_requires,
        'dev': dev_requires,
        'all': jax_requires + dev_requires,
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'trajectory optimization',
        'robotics',
        'control',
        'spline interpolation',
        'reinforcement learning',
        'model predictive control',
        'sampling-based optimization',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/MasterYip/PredictiveDiffusionPlanner/issues',
        'Source': 'https://github.com/MasterYip/PredictiveDiffusionPlanner',
        'Documentation': 'https://github.com/MasterYip/PredictiveDiffusionPlanner/blob/main/README.md',
    },
)