from __future__ import annotations
from typing import Any, Dict, List
from simulation.runtime_builder import build_runtime
from evaluation.evaluator import Evaluator
from simulation.results import ComparisonResult
from evaluation.evaluation_runner import run_evaluation_episodes

class ControllerComparator:
    """ Comparador de controladores. Executa vários controladores no mesmo ambiente,usando a mesma configuração. """
    def __init__(
        self,
        environment_type: str,
        controller_types: List[str],
        num_episodes: int = 3,
        max_steps: int = 500,
        render: bool = False,
        env_overrides: Dict[str, Any] | None = None,
        controller_overrides: Dict[str, Dict[str, Any]] | None = None,
        use_trained_models: Dict[str, bool] | None = None,
        trained_model_names: Dict[str, str | None] | None = None,
    ) -> None:
        self.environment_type = environment_type
        self.controller_types = controller_types
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.render = render
        self.env_overrides = env_overrides or {}
        self.controller_overrides = controller_overrides or {}
        self.use_trained_models = use_trained_models or {}
        self.trained_model_names = trained_model_names or {}

    def _build_config(self, controller_type: str) -> Dict[str, Any]:
        """ O comparador transforma cada controlador numa execução de avaliação normal."""
        return {
            "mode": "evaluate",
            "controller_type": controller_type,
            "environment_type": self.environment_type,
            "render": self.render,
            "max_steps": self.max_steps,
            "num_episodes": self.num_episodes,
            "use_trained_model": self.use_trained_models.get(controller_type, False),
            "trained_model_name": self.trained_model_names.get(controller_type),
            "env_overrides": self.env_overrides,
            "controller_overrides": self.controller_overrides.get(
                controller_type, {}
            ),
        }
    def evaluate_controller(self, controller_type: str) -> Dict[str, Any]:
        config = self._build_config(controller_type)

        return run_evaluation_episodes(
            config=config,
            num_episodes=self.num_episodes,
            max_steps=self.max_steps,
            render=self.render,
            render_delay=0.0,
        )
    
    def run(self) -> ComparisonResult:
        comparison_results = {}

        for controller_type in self.controller_types:
            print(f"\n=== A avaliar controlador: {controller_type} ===")
            comparison_results[controller_type] = self.evaluate_controller(
                controller_type
            )
        return ComparisonResult(
            environment_name=self.environment_type,
            controllers=self.controller_types,
            num_episodes=self.num_episodes,
            max_steps=self.max_steps,
            results=comparison_results,
        )