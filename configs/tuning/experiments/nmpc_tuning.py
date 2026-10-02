# testar automaticamente diferentes pesos do NMPC e escolher a melhor configuração para duplo pendulo invertido.

from __future__ import annotations
from typing import Any, Dict
from environments.custom_envs.double_inverted_pendulum.models.nonlinear_model import (get_nonlinear_double_inverted_pendulum_config,)
from factories.controller_factory import create_controller
from factories.environment_factory import create_environment
from simulation.simulation_runner import SimulationRunner
from configs.tuning.core import get_best_result, run_grid_search

def build_cost_weights(parameters: Dict[str, Any]) -> Dict[str, float]:
    """ Pega na configuração base do modelo não linear do pêndulo duplo e altera apenas os pesos definidos no tuning."""
    base_config = get_nonlinear_double_inverted_pendulum_config()
    cost_weights = base_config["cost_weights"].copy()
    cost_weights.update(parameters)
    return cost_weights

def evaluate_nmpc_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    env = create_environment(
        "double_inverted_pendulum",
        overrides={},
    )

    controller = create_controller(
        "nonlinear_mpc",
        env=env,
        overrides={
            "cost_weights": build_cost_weights(parameters),
        },
    )

    controller.initialize()

    runner = SimulationRunner(
        controller=controller,
        environment=env,
    )

    result = runner.run_episode(
        max_steps=500,
        render=False,
        render_delay=0.0,
    )

    env.close()

    total_reward = result["total_reward"]
    num_steps = result["num_steps"]

    return {
        "total_reward": total_reward,
        "average_reward": total_reward / num_steps if num_steps > 0 else -999999.0,
        "num_steps": num_steps,
        "terminated": result["terminated"],
        "truncated": result["truncated"],
    }

def main() -> None:
    search_space = {
        "theta1": [8.0, 10.0],
        "theta2": [8.0, 10.0],
        "position": [1.5, 2.0, 2.5],
        "rterm_force": [0.02, 0.05, 0.08],
    }

    results = run_grid_search(
        search_space=search_space,
        evaluate_function=evaluate_nmpc_parameters,
        score_key="average_reward",
        metadata={
            "controller": "nonlinear_mpc",
            "environment": "double_inverted_pendulum",
            "max_steps": 500,
        },
    )

    print("\n=== RESULTADOS DO TUNING NMPC ===")
    for result in results:
        print(result.to_dict())

    best = get_best_result(results, maximize=True)

    print("\n=== MELHOR RESULTADO ===")
    print(best.to_dict())

if __name__ == "__main__":
    main()