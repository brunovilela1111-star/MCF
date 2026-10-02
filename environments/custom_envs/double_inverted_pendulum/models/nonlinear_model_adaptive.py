# usa base + adiciona parâmetros adaptativos
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

def get_nonlinear_double_inverted_pendulum_adaptive_config() -> Dict[str, Any]:
    """ Configuração inicial do Adaptive MPC.  """
    horizons = get_base_horizons()
    return {
        "use_environment_model": True,
        "reference": None,

        "constraints": get_base_constraints(),

        "prediction_horizon": horizons["prediction_horizon"],
        "control_horizon": horizons["control_horizon"],

        "t_step": 0.04,

        "solver_settings": get_base_solver_settings(),

        "cost_weights": get_base_cost_weights(),

        "tvp": get_base_tvp(),

        "adaptive_parameters": {
            "m1": 0.2,
            "m2": 0.2,
            "enabled": False,
        },

        "builders": get_base_builders(),
    }