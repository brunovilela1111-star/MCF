from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

class BaseEnvironment(ABC):
    """ Classe base comum a qualquer ambiente. """
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """ Inicializa o ambiente base."""
        self.name = name
        self.config = config or {}
        self.current_state = None
        self.current_observation = None
        self.current_reference = None
        self.done = False
    @abstractmethod
    def reset(self) -> Any:
        """Reinicia o ambiente para o estado inicial."""
        pass
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """ Executa um passo de simulação no ambiente."""
        pass
    def get_state(self) -> Any:
        """ Devolve o estado atual do sistema.""" #importante para MPC
        return self.current_state
    def get_observation(self) -> Any:
        """ Devolve a observação atual disponível para o controlador. """ #importante para MPC
        return self.current_observation
    def get_reference(self) -> Any:
        """ Devolve a referência atual do ambiente, se existir. """
        return self.current_reference
    def is_done(self) -> bool:
        """ Indica se o episódio/simulação terminou. """
        return self.done
    def render(self) -> None:
        """ Renderiza o ambiente, se aplicável. """
        return None
    def close(self) -> None:
        """  Fecha o ambiente. """
        return None
    def get_info(self) -> Dict[str, Any]:
        """ Devolve informação básica do ambiente. """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
            "has_state": self.current_state is not None,
            "has_observation": self.current_observation is not None,
            "has_reference": self.current_reference is not None,
            "done": self.done,
        }