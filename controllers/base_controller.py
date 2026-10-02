from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseController(ABC):
    """ Classe base abstrata comum para qualquer tipo de controlador."""
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """ Inicialização """
        self.name = name
        self.config = config or {}
        self.is_initialized = False

    def initialize(self) -> None:
        """ Inicializa o controlador para utilização. RL - criar modelo // MPC - Preparar solver"""
        self.is_initialized = True
        print(f"[{self.name}] Controlador inicializado.")

    @abstractmethod
    def compute_action(self, observation: Any) -> Any:
        """ Calcula a ação de controlo com base na observação/estado atual."""
        pass

    def reset(self) -> None:
        """  Reinicia o estado interno do controlador."""
        print(f"[{self.name}] Controlador reiniciado.")

    def update(self, *args, **kwargs) -> None:
        """ Atualiza o controlador, se aplicável. RL - treino // MPC - atualizar modelo"""
        return None

    def get_info(self) -> Dict[str, Any]:
        """  Devolve informação básica do controlador. """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "initialized": self.is_initialized,
            "config": self.config,
        }