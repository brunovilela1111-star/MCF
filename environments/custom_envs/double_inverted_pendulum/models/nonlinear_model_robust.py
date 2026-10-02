#configuração robusta do NMPC usa base + altera robustez e horizontes
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

def get_nonlinear_double_inverted_pendulum_robust_config() -> Dict[str, Any]:
    """ Configuração do NMPC robusto. """
    horizons = get_base_horizons()
    solver_settings = get_base_solver_settings()
    solver_settings.update(
        {
            "n_robust": 2,
        }
    )
    return {
        "use_environment_model": True,
        "reference": None,
        "constraints": get_base_constraints(),
        "prediction_horizon": 10,
        "control_horizon": 10,
        "t_step": 0.04,
        "solver_settings": solver_settings,
        "cost_weights": get_base_cost_weights(),
        "tvp": get_base_tvp(),
        "builders": get_base_builders(),
    }