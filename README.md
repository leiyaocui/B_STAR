<h1 align="center">B*: Robot Trajectory Optimization Framework</h1>

<div align="center">
  <a href="https://arxiv.org/abs/2504.12719"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://bstar-planning.github.io'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</div>

<h4 align="center">Accepted by IEEE Robotics and Automation Letters (RA-L) 2025</h4>

<p align="center">
  <a href="https://zihangzhao.com/" target="_blank">Zihang Zhao</a><sup>⚖️</sup>,
  <a href="https://lycui.com/" target="_blank">Leiyao Cui</a><sup>⚖️</sup>,
  <a href="https://siruixie.github.io/" target="_blank">Sirui Xie</a><sup>⚖️</sup>,
  <a href="https://saiyaozhang.com/" target="_blank">Saiyao Zhang</a>,
  Zhi Han,
  <a href="https://leleucla.github.io/" target="_blank">Lecheng Ruan</a>,
  <a href="https://yzhu.io/" target="_blank">Yixin Zhu</a><sup>✉️</sup>
</p>

<p align="center">
  ⚖️: Equal contributor, ✉️: Corresponding author
</p>

---

This repository provides the official implementation of [B*: Efficient and Optimal Base Placement for Fixed-Base Manipulators](https://bstar-planning.github.io), which also serves as **a fully Python-based alternative to [TrajOpt](https://github.com/joschu/trajopt)**.

## Abstract

Proper base placement is crucial for task execution feasibility and performance of fixed-base manipulators, the dominant solution in robotic automation. Current methods rely on pre-computed kinematics databases generated through sampling to search for solutions. However, they face an inherent trade-off between solution optimality and computational efficiency when determining sampling resolution—a challenge that intensifies when considering long-horizon trajectories, self-collision avoidance, and task-specific requirements. To address these limitations, we present B*, a novel optimization framework for determining the optimal base placement that unifies these multiple objectives without relying on pre-computed databases. B* addresses this inherently non-convex problem via a two-layer hierarchical approach: The outer layer systematically manages terminal constraints through progressively tightening them, particularly the base mobility constraint, enabling feasible initialization and broad solution space exploration. Concurrently, the inner layer addresses the non-convexities of each outer-layer subproblem by sequential local linearization, effectively transforming the original problem into a tractable sequential linear programming (SLP). Comprehensive evaluations across multiple robot platforms and task complexities demonstrate the effectiveness of B*: it achieves solution optimality five orders of magnitude better than sampling-based approaches while maintaining perfect success rates, all with reduced computational overhead. Operating directly in configuration space, B* not only solves the base placement problem but also enables simultaneous path planning with customizable optimization criteria, making it a versatile framework for various robotic motion planning challenges. B* serves as a crucial initialization tool for robotic applications, bridging the gap between theoretical motion planning and practical deployment, where feasible trajectory existence is fundamental. 

<b><p align="center" style="font-size: 1.4rem;">Proper Base Placement is Crucial for Task Execution Feasibility</p></b>

![B* Framework Overview](figures/teaser_website.svg)

## Features

**While the framework effectively solves the base placement problem, it also serves as a general trajectory optimization framework.**

- **Core Optimization Framework**:
  - Two-layer hierarchical optimization with trust-region sequential linear programming
  - Unified framework for both base placement and general trajectory optimization
  - Extensible framework supporting custom terms
  - Built on [CVXPY](https://github.com/cvxpy/cvxpy)
  - Core optimization terms:
    - End-effector pose tracking (`EEPoseTerm`)
    - Collision avoidance with discrete-time checking (`DiscreteCollisionTerm`)
    - Trajectory smoothing (`TrajSmoothTerm`)
    - Base fixation constraints (`BaseFixationTerm`)

- **Multiple Robot Support**: 
  - Compatible with various robot models including:
    - Franka Emika Panda
    - KUKA LBR iiwa
    - Kinova Gen3
    - Universal Robots UR10e
  - Uses [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for robot kinematics and dynamics
  - Supports URDF-based robot models with mesh-based collision models

## Installation

### Prerequisites
- Conda package manager
- C++ compiler for building optimization terms
- License of [COPT](https://www.shanshu.ai/copt) solver

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/B_STAR.git
cd B_STAR
```

2. Create Python environment:
```bash
conda create -n b_star python=3.10 -y
conda activate b_star

conda install conda-forge::pinocchio=3.3.1 -y
pip install "numpy<2" scipy cvxpy coptpy pyyaml loguru prettytable
```

This will:
- Create a conda environment `b_star` with Python 3.10
- Install Pinocchio 3.3.1 and other dependencies
- Build and install the C++ optimization terms

## Project Structure

```
B_STAR/
├── content/                   # Robot models and configurations
│   ├── assets/                # Robot meshes and URDFs
│   └── configs/               # Robot configuration files
├── opt_prob/                  # Optimization problem definitions
├── opt_term/                  # Optimization term implementations
│   └── cpp/                   # C++ implementations
├── utils/                     # Utility functions
└── opt_cfg/                   # Optimization configurations
```

## Usage

### Basic Usage

```bash
python main.py \
    --robot_cfg_path content/configs/robot/franka.yml \
    --traj_data_path path/to/trajectory/data.pkl \
    --opt_cfg_path opt_cfg/default.yml
```

### Command Line Arguments

- `--robot_cfg_path`: Path to robot configuration file (required)
- `--traj_data_path`: Path to trajectory data file (required)
- `--opt_cfg_path`: Path to optimization configuration file (default: opt_cfg/default.yml)
- `--num_timesteps`: Number of timesteps for trajectory (optional)
- `--disable_ground_plane`: Disable ground plane collision checking
- `--disable_initial_guess`: Disable initial guess optimization
- `--ret_all_steps`: Return all optimization steps
- `--log_level`: Logging level (default: INFO)
- `--solver_verbose`: Enable verbose solver output

### Reproducing Paper Results

To reproduce the quantitative results reported in our paper:

1. Ensure you have the benchmark data in `data/benchmark` directory
2. Run the batch script:
```bash
chmod +x batch_run.sh
./batch_run.sh
```

The script will:
- Process benchmark data for all supported robots (Franka, IIWA, Kinova Gen3, UR10e)
- Use robot configurations from `content/configs/robot`
- Save results to `logs/benchmark` directory

Results will include optimization performance metrics and solution quality measures as reported in the paper.

### Optimization Configuration

The optimization behavior can be customized through YAML configuration files:

```yaml
# Example optimization configuration
tr_sp:
  max_iter: 40
  improve_ratio_threshold: 0.2
  min_approx_improve: 1.0e-4
  min_approx_improve_frac: 1.0e-3
  trust_shrink_ratio: 0.1
  trust_expand_ratio: 1.5
  max_merit_coeff_iter: 4
  merit_coeff_increase_ratio: 10.0
traj_smooth_term:
  merit_coeff: 1.0
  threshold: 0.2
self_collision_term:
  sd_check: 0.1
  sd_safe: 0.05
  merit_coeff: 200.0
  threshold: 1.0e-4
```

## Extending with Custom Terms

The framework is designed to be easily extensible with custom optimization terms. Each term inherits from the base `Term` class and can be integrated seamlessly into the optimization problem.

### Term Types
- `COST_SQUARE`: Squared cost terms
- `COST_ABS`: Absolute cost terms
- `CONSTRAINT_EQ`: Equality constraints
- `CONSTRAINT_INEQ`: Inequality constraints

### Implementation Guide

To add a custom optimization term:

1. Create a new term class:
   ```python
   class CustomTerm(Term):
       def __init__(self, name: str = "custom term", threshold: float = 1e-4):
           # Choose appropriate TermType for your optimization objective
           super().__init__(TermType.COST_SQUARE, name, threshold)
           # Initialize any term-specific parameters here
   ```

2. Implement the three required abstract methods:

   a. `_approx(self, variable: np.ndarray) -> None`:
   ```python
   def _approx(self, variable: np.ndarray) -> None:
       """Compute and store approximations needed for optimization.
       
       For simple terms (like TrajSmoothTerm) this can be empty.
       For complex terms (like EEPoseTerm) this computes and stores
       linearization data like Jacobians.

       Args:
           variable: Current optimization variables (n_timesteps, n_dof)
       """
   ```

   b. `_construct_approx_expr(self, variable: cp.Expression) -> cp.Expression`:
   ```python
   def _construct_approx_expr(self, variable: cp.Expression) -> cp.Expression:
       """Construct the CVXPY expression for optimization.
       
       The expression should evaluate to a vector that will be processed
       according to the term type (squared, absolute value, etc).

       Args:
           variable: CVXPY variable for optimization (n_timesteps, n_dof)
       
       Returns:
           CVXPY expression to be minimized/constrained (should be flattened)
       """
   ```

   c. `_eval(self, variable: np.ndarray) -> float | np.ndarray`:
   ```python
   def _eval(self, variable: np.ndarray) -> float | np.ndarray:
       """Evaluate the actual term value for current variables.
       
       This computes the true (non-approximated) value of the term
       for convergence checking.

       Args:
           variable: Current optimization variables (n_timesteps, n_dof)
       
       Returns:
           Term value as numpy array to be processed by term type (should be flattened)
       """
   ```

The framework will automatically handle:
- Term construction and integration
- Slack variable management
- Cost/constraint evaluation
- Optimization problem formulation

## Citation

If you find our research beneficial, please cite both our paper and the prior works that strongly motivated this study as follows:

```bibtex
@article{zhao2025b,
    title={B*: Efficient and Optimal Base Placement for Fixed-Base Manipulators},
    author={Zhao, Zihang and Cui, Leiyao and Xie, Sirui and Zhang, Saiyao and Han, Zhi and Ruan, Lecheng and Zhu, Yixin},
    journal={IEEE Robotics and Automation Letters (RA-L)},
    volume={10},
    number={10},
    pages={10634-10641},
    year={2025},
    publisher={IEEE}
}
```

```bibtex
@article{schulman2014motion,
  title={Motion planning with sequential convex optimization and convex collision checking},
  author={Schulman, John and Duan, Yan and Ho, Jonathan and Lee, Alex and Awwal, Ibrahim and Bradlow, Henry and Pan, Jia and Patil, Sachin and Goldberg, Ken and Abbeel, Pieter},
  journal={The International Journal of Robotics Research},
  volume={33},
  number={9},
  pages={1251--1270},
  year={2014},
  publisher={Sage Publications Sage UK: London, England}
}
```
