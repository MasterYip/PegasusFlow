<div align="center">

  <h1 style="margin: 0; font-size: 1.8em;">
      <!-- <img src="./assets/logo_white.png" alt="Logo" width="60" style="vertical-align: middle; margin-right: 10px;"> -->
      PegasusFlow: Parallel Rolling-Denoising Score Sampling for Robot Diffusion Planner Flow Matching
  </h1>

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2509.08435)
[![arXiv](https://img.shields.io/badge/arXiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white&color=blue)](https://arxiv.org/abs/2509.08435)
[![Project Page](https://img.shields.io/badge/Project_Page-00CED1?style=for-the-badge&logo=web&logoColor=white)](https://masteryip.github.io/pegasusflow.github.io/)

![Banner](doc/Banner.svg)

</div>

> [!NOTE]
> Code & Documentation coming soon.
>
> We use [extended_legged_gym](https://github.com/MasterYip/extended_legged_gym) as the major simulation environment.

## Installation

1.Install virtual environment

```bash
mamba create -n pegasusflow python=3.8
mamba activate pegasusflow
```

2.Install [IsaacGym](https://developer.nvidia.com/isaac-gym/download)

```bash
cd isaacgym/python
pip install -e .
```

3.Clone and install this repo

```bash
git clone https://github.com/MasterYip/PegasusFlow --recursive
cd PegasusFlow
bash ./install.sh
```

<!-- 4.Install python dependencies

```bash
pip install -r requirements.txt
``` -->

## Getting Started (WIP)

> [!NOTE]
> Still Work In Progress.

### Basic Usage

Run the demo with default settings:

```bash
python demos.py --robot anymal_c
```

### Available Robots

The following robot tasks are available:
- `anymal_c`: ANYmal-C quadruped robot
- `elspider_air_barrier_nav`: ElSpider robot with air barrier navigation
- `elspider_air_timberpile_nav`: ElSpider robot with timber pile navigation  
- `franka`: Franka robot arm

### Key Arguments

- `--num_envs`: Number of main simulation environments (default: 1)
- `--rollout_envs`: Number of rollout environments per main environment for trajectory optimization (default: 128)
- `--horizon_nodes`: Number of control nodes in the planning horizon
- `--horizon_samples`: Number of trajectory samples for optimization
- `--command`: Motion command for the robot

### Example Commands

**ANYmal-C locomotion:**
```bash
python demos.py --robot anymal_c --num_envs 1 --rollout_envs 128 --command walk_forward
```

**ElSpider barrier navigation:**
```bash
python demos.py --robot elspider_air_barrier_nav --num_envs 2 --rollout_envs 128
```

**Franka arm manipulation:**
```bash
python demos.py --robot franka --num_envs 1 --rollout_envs 128 --command reach_backward
```

### Available Commands

**Locomotion commands** (for anymal_c, elspider variants):
- `walk_forward`, `walk_backward`
- `strafe_left`, `strafe_right` 
- `turn_left`, `turn_right`
- `stop`

**Manipulation commands** (for franka):
- `reach_forward`, `reach_backward`
- `reach_left`, `reach_right`
- `reach_up`, `reach_down`
- `home`

### Additional Options

Run headless (no GUI):
```bash
python demos.py --robot anymal_c --headless
```

Disable trajectory optimization (policy only):
```bash
python demos.py --robot anymal_c --disable_trajectory_opt
```

## Acknowledgements

- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [dial-mpc](https://github.com/LeCAR-Lab/dial-mpc)
- [diffusion-policy](https://github.com/real-stanford/diffusion_policy)
- [diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing)
