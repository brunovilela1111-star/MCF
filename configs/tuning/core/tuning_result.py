from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class TuningResult:
    """ Guarda resultados de uma configuração testada no tuning. """
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        "Converte o resultado para dicionário, útil para imprimir ou guardar em JSON."
        return {
            "parameters": self.parameters,
            "metrics": self.metrics,
            "score": self.score,
            "metadata": self.metadata,
        }