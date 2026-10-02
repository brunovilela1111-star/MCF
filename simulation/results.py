# definir estruturas padronizadas de resultado usando dataclass.
# Isto torna os resultados mais organizados e fáceis de guardar/exportar.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class SimulationResult:
    """ Resultado estruturado de uma simulação de um único episódio."""
    controller_name: str
    environment_name: str
    episode_result: Dict[str, Any]
    metrics: Dict[str, Any]

@dataclass
class EvaluationResult:
    """Resultado estruturado de uma avaliação com múltiplos episódios."""
    controller_name: str
    environment_name: str
    episode_results: List[Dict[str, Any]]
    metrics_list: List[Dict[str, Any]]
    summary: Dict[str, Any]

@dataclass
class TrainingResult:
    """ Resultado estruturado de um processo de treino."""
    controller_name: str
    environment_name: str
    total_timesteps: int
    model_path: str

@dataclass
class ComparisonResult:
    """ Resultado estruturado de uma comparação entre vários controladores."""
    environment_name: str
    controllers: List[str]
    num_episodes: int
    max_steps: int
    results: Dict[str, Any]