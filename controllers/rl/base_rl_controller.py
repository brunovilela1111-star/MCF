from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, Optional
from controllers.base_controller import BaseController

class BaseRLController(BaseController):
    """ Classe base comum a todos os controladores de Reinforcement Learning. """
    def __init__(
        self,
        name: str,
        algorithm_name: str,
        config: Optional[Dict[str, Any]] = None,
        policy: Optional[Any] = None,
        device: str = "cpu",
    ) -> None:
        """ Inicializa o controlador base de RL. """
        super().__init__(name=name, config=config)
        self.algorithm_name = algorithm_name
        self.policy = policy
        self.device = device
        self.training_mode = True

    @abstractmethod
    def predict(self, observation: Any, deterministic: bool = True) -> Any: 
        """ Prediz a ação com base na observação atual."""
        pass

    def compute_action(self, observation: Any) -> Any:
        """ Implementação da interface comum de controlo -> Em controladores RL, compute_action delega para predict(). """
        deterministic = not self.training_mode
        return self.predict(observation, deterministic=deterministic)

    @abstractmethod
    def train_step(self, *args, **kwargs) -> None:
        """ Executa um passo de treino do algoritmo RL cada um pela sua lógica."""
        pass

    def update(self, *args, **kwargs) -> None:
        """ Mantém compatibilidade com a interface genérica. ->  Em RL, update() corresponde ao passo de treino. """
        self.train_step(*args, **kwargs)

    def train_mode(self) -> None:
        """ Coloca o controlador em modo de treino."""
        self.training_mode = True

    def eval_mode(self) -> None:
        """Coloca o controlador em modo de avaliação."""
        self.training_mode = False

    def get_info(self) -> Dict[str, Any]:
        """Devolve informação detalhada do controlador RL. """
        info = super().get_info()
        info.update(
            {
                "algorithm_name": self.algorithm_name,
                "device": self.device,
                "training_mode": self.training_mode,
                "has_policy": self.policy is not None,
            }
        )
        return info