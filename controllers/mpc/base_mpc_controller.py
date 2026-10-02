from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, Optional
from controllers.base_controller import BaseController

class BaseMPCController(BaseController):
    """ Classe base comum a todos os controladores de Model Predictive Control (MPC). """
    def __init__(
        self,
        name: str,
        mpc_type: str,
        config: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None,
        prediction_horizon: int = 10,
        control_horizon: Optional[int] = None, # Se none assume o valor igual do prediction_horizon
        reference: Optional[Any] = None,
        constraints: Optional[Dict[str, Any]] = None,
        solver: Optional[Any] = None,
    ) -> None:
        """Inicializa o controlador base de MPC. """
        super().__init__(name=name, config=config)
        self.mpc_type = mpc_type
        self.model = model
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon or prediction_horizon
        self.reference = reference
        self.constraints = constraints or {}
        self.solver = solver

    @abstractmethod
    def solve_mpc(self, observation: Any) -> Any:
        """ Resolve o problema de otimização MPC para a observação/estado atual. """
        pass

    def compute_action(self, observation: Any) -> Any:
        """ Implementação da interface comum de controlo. -> Em controladores MPC, compute_action delega para solve_mpc(). """
        return self.solve_mpc(observation)

    def set_reference(self, reference: Any) -> None:
        """ Atualiza a referência do controlador. """
        self.reference = reference

    def set_constraints(self, constraints: Dict[str, Any]) -> None:
        """ Atualiza as restrições. """
        self.constraints = constraints

    def get_info(self) -> Dict[str, Any]:
        """ Devolve informação detalhada."""
        info = super().get_info()
        info.update(
            {
                "mpc_type": self.mpc_type,
                "prediction_horizon": self.prediction_horizon,
                "control_horizon": self.control_horizon,
                "has_model": self.model is not None,
                "has_reference": self.reference is not None,
                "has_constraints": bool(self.constraints),
                "has_solver": self.solver is not None,
            }
        )
        return info