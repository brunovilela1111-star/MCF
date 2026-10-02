# Modular Control Framework (MCF)

A modular Python framework developed for the **implementation, simulation and comparison of Model Predictive Control (MPC) and Reinforcement Learning (RL) controllers**.

The framework provides a common architecture that allows different controllers and environments to be integrated and evaluated using the same simulation workflow and performance metrics.

## Main Features

- Modular and extensible architecture
- Model Predictive Control: **Linear MPC and Nonlinear MPC**
- Reinforcement Learning: **PPO, SAC, A2C, DDPG, TD3 and DQN**
- Support for **Gymnasium, custom environments and FMUs**
- RL training and model storage
- Controller simulation, evaluation and comparison
- Standardised performance metrics
- Automatic result storage

## Case Studies

The framework was validated using two main control problems:

- **Double Inverted Pendulum** — nonlinear and unstable system used to compare NMPC and RL controllers.
- **Injection Mould Thermal System** — FMU-based industrial thermal process controlled through cooling-water flow.

## Usage

The framework is configured through:

```text
configs/system/main_config.py
```

Four execution modes are available:

```text
train
simulate
evaluate
compare
```

After selecting the controller, environment and execution mode, run:

```bash
python main.py
```

Example:

```python
MAIN_CONFIG = {
    "mode": "simulate",
    "controller_type": "nonlinear_mpc",
    "environment_type": "double_inverted_pendulum",
    "render": True,
    "max_steps": 500,
}
```

## Main Dependencies

```text
Python
NumPy
Gymnasium
Stable-Baselines3
CVXPY
do-mpc
CasADi
FMPy
Matplotlib
```

## Project Context

This framework was developed as part of a **Master's Dissertation in Industrial Engineering and Management**, with the objective of providing a modular and consistent platform for analysing and comparing MPC and RL control strategies.

**Author:** Bruno Vilela