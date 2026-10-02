#versão otimizada/tunada do NMPC. usa base + altera pesos
from __future__ import annotations
from typing import Any, Dict
from environments.custom_envs.double_inverted_pendulum.models.common import (
    get_base_builders,
    get_base_constraints,
    get_base_cost_weights,
    get_base_horizons,
    get_base_solver_settings,
    get_base_tvp,
)

def get_nonlinear_double_inverted_pendulum_optimized_config() -> Dict[str, Any]:
    """ Configuração NMPC otimizada com base no tuning. """
    horizons = get_base_horizons()
    cost_weights = get_base_cost_weights()
    cost_weights.update(
        {
            "theta1": 8.0,
            "theta2": 8.0,
            "position": 1.5,
            "rterm_force": 0.02,
        }
    )

    return {
        "use_environment_model": True,
        "reference": None,
        "constraints": get_base_constraints(),
        "prediction_horizon": horizons["prediction_horizon"],
        "control_horizon": horizons["control_horizon"],
        "t_step": 0.04,
        "solver_settings": get_base_solver_settings(),
        "cost_weights": cost_weights,
        "tvp": get_base_tvp(),
        "builders": get_base_builders(),
    }