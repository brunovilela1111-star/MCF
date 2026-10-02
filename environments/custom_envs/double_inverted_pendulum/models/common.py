#configurações comuns reutilizadas pelos vários modelos NMPC.
from __future__ import annotations
from typing import Any, Dict

def get_base_constraints() -> Dict[str, Any]:
    return {
        "u_min": -4.0,
        "u_max": 4.0,
    }

def get_base_horizons() -> Dict[str, int]:
    return {
        "prediction_horizon": 20,
        "control_horizon": 20,
    }

def get_base_solver_settings() -> Dict[str, Any]:
    return {
        "n_robust": 0,
        "open_loop": 0,
        "state_discretization": "collocation",
        "collocation_type": "radau",
        "collocation_deg": 3,
        "collocation_ni": 1,
        "store_full_solution": True,
    }

def get_base_cost_weights() -> Dict[str, float]:
    return {
        "theta1": 10.0,
        "theta2": 10.0,
        "position": 2.0,
        "dpos": 0.2,
        "dtheta1": 0.1,
        "dtheta2": 0.1,
        "rterm_force": 0.05,
    }

def get_base_tvp() -> Dict[str, Any]:
    return {
        "pos_set": 0.0,
    }

def get_base_builders() -> Dict[str, str]:
    return {
        "module_path": "environments.custom_envs.double_inverted_pendulum.mpc_cost",
        "objective_builder": "build_objective",
        "bounds_builder": "apply_bounds",
        "uncertainty_builder": "apply_uncertainty",
        "tvp_builder": "build_tvp_fun",
        "rterm_builder": "apply_rterm",
    }