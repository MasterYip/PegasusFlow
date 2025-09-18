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

2.Install python dependencies

```bash
pip install -r requirements.txt
```

3.Install [IsaacGym](https://developer.nvidia.com/isaac-gym/download)

```bash
cd isaacgym/python
pip install -e .
```

4.Install the package

```bash
bash ./install.sh
```

## Getting Started

Comming soon.

## Acknowledgements

- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [dial-mpc](https://github.com/LeCAR-Lab/dial-mpc)
- [diffusion-policy](https://github.com/real-stanford/diffusion_policy)
- [diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing)
