from __future__ import annotations
from importlib import import_module
from typing import Any, Callable, Dict, Optional
import do_mpc
import numpy as np
from controllers.mpc.base_mpc_controller import BaseMPCController

class NonlinearMPCController(BaseMPCController):
    """ Controlador NMPC genérico baseado em do-mpc. """
    def __init__(
        self,
        name: str,
        mpc_type: str,
        config: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None,
        env: Optional[Any] = None,
        use_environment_model: bool = True,
        prediction_horizon: int = 20,
        control_horizon: Optional[int] = None,
        reference: Optional[Any] = None,
        constraints: Optional[Dict[str, Any]] = None,
        solver: Optional[str] = None,
        t_step: float = 0.04,
        builders: Optional[Dict[str, Any]] = None,
        solver_settings: Optional[Dict[str, Any]] = None,
        cost_weights: Optional[Dict[str, float]] = None,
        tvp: Optional[Dict[str, Any]] = None,
        adaptive_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if use_environment_model:
            if env is None:
                raise ValueError(
                    f"[{name}] Env é necessário quando use_environment_model=True."
                )

            if not hasattr(env, "model"):
                raise ValueError(
                    f"[{name}] O ambiente fornecido não tem atributo 'model'."
                )

            model = env.model

        super().__init__(
            name=name,
            mpc_type=mpc_type,
            config=config,
            model=model,
            prediction_horizon=prediction_horizon,
            control_horizon=control_horizon,
            reference=reference,
            constraints=constraints,
            solver=solver,
        )

        self.env = env
        self.use_environment_model = use_environment_model
        self.t_step = float(t_step)

        self.builders_config = builders or {}
        self.solver_settings = solver_settings or {}
        self.cost_weights = cost_weights or {}
        self.tvp_config = tvp or {}
        self.adaptive_parameters = adaptive_parameters or {}

        self.mpc = None
        self.estimator = None
        self._initialized = False

        if self.model is not None:
            self.mpc = self._build_mpc()

    def _get_solver_settings(self) -> Dict[str, Any]:
        default_settings = {
            "n_robust": 0,
            "open_loop": 0,
            "state_discretization": "collocation",
            "collocation_type": "radau",
            "collocation_deg": 3,
            "collocation_ni": 1,
            "store_full_solution": True,
        }

        default_settings.update(self.solver_settings)
        return default_settings

    def _load_builder(self, builder_key: str) -> Callable:
        if not self.builders_config:
            raise ValueError(
                f"[{self.name}] Nenhuma configuração de builders foi definida."
            )

        module_path = self.builders_config.get("module_path")
        callable_name = self.builders_config.get(builder_key)

        if module_path is None:
            raise ValueError(
                f"[{self.name}] 'module_path' não definido em builders_config."
            )

        if callable_name is None:
            raise ValueError(
                f"[{self.name}] Builder '{builder_key}' não definido."
            )

        module = import_module(module_path)

        if not hasattr(module, callable_name):
            raise AttributeError(
                f"[{self.name}] Builder '{callable_name}' não existe em {module_path}."
            )

        return getattr(module, callable_name)

    def _apply_solver_settings(self, mpc: Any) -> None:
        solver_settings = self._get_solver_settings()

        mpc.settings.n_horizon = self.prediction_horizon
        mpc.settings.t_step = self.t_step

        mpc.settings.n_robust = solver_settings["n_robust"]
        mpc.settings.open_loop = solver_settings["open_loop"]
        mpc.settings.state_discretization = solver_settings["state_discretization"]
        mpc.settings.collocation_type = solver_settings["collocation_type"]
        mpc.settings.collocation_deg = solver_settings["collocation_deg"]
        mpc.settings.collocation_ni = solver_settings["collocation_ni"]
        mpc.settings.store_full_solution = solver_settings["store_full_solution"]

    def _build_mpc(self):
        mpc = do_mpc.controller.MPC(self.model)

        self._apply_solver_settings(mpc)

        objective_builder = self._load_builder("objective_builder")
        rterm_builder = self._load_builder("rterm_builder")
        bounds_builder = self._load_builder("bounds_builder")
        uncertainty_builder = self._load_builder("uncertainty_builder")
        tvp_builder = self._load_builder("tvp_builder")

        mterm, lterm = objective_builder(
            model=self.model,
            cost_weights=self.cost_weights,
        )

        mpc.set_objective(mterm=mterm, lterm=lterm)

        rterm_builder(
            mpc=mpc,
            cost_weights=self.cost_weights,
        )

        bounds_builder(
            mpc=mpc,
            constraints=self.constraints,
        )

        uncertainty_builder(mpc=mpc)

        tvp_fun = tvp_builder(
            mpc=mpc,
            tvp_config=self.tvp_config,
        )

        mpc.set_tvp_fun(tvp_fun)

        mpc.setup()
        return mpc

    def solve_mpc(self, observation: Any) -> Any:
        if self.mpc is None:
            raise RuntimeError(f"[{self.name}] NMPC não inicializado.")

        x0 = np.asarray(observation, dtype=float).reshape(-1, 1)

        if not self._initialized:
            self.mpc.x0 = x0
            self.mpc.set_initial_guess()
            self._initialized = True

        u = self.mpc.make_step(x0)

        return np.asarray(u).reshape(-1)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "has_nonlinear_model": self.model is not None,
                "has_mpc_object": self.mpc is not None,
                "has_estimator": self.estimator is not None,
                "use_environment_model": self.use_environment_model,
                "has_env": self.env is not None,
                "t_step": self.t_step,
                "solver_settings": self._get_solver_settings(),
                "builders_config": self.builders_config,
                "has_adaptive_parameters": bool(self.adaptive_parameters),
            }
        )
        return info
    def reset(self) -> None:
        """ Reinicia o estado interno do NMPC entre episódios."""
        super().reset()
        self._initialized = False

        if self.mpc is not None:
            self.mpc.reset_history()