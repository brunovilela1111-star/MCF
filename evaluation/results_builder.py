# organizar a saída final da avaliação num formato comum e limpo.
from __future__ import annotations
from typing import Any, Dict, List

def build_evaluation_result(
    episode_metrics: List[Dict[str, Any]],
    summary_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constrói uma estrutura final padronizada para resultados de avaliação.
    Esta função não calcula métricas.
    Apenas organiza as métricas já calculadas.
    """

    # Garante que existe pelo menos um episódio avaliado.
    if not episode_metrics:
        raise ValueError("episode_metrics não pode estar vazio.")

    # Devolve um dicionário organizado com:
    # - identificação do controlador;
    # - identificação do ambiente;
    # - número de episódios;
    # - métricas por episódio;
    # - resumo agregado.
    return {
        "controller_name": summary_metrics.get("controller_name"),
        "controller_type": summary_metrics.get("controller_type"),
        "controller_family": summary_metrics.get("controller_family"),
        "environment_name": summary_metrics.get("environment_name"),
        "environment_type": summary_metrics.get("environment_type"),
        "num_episodes": summary_metrics.get(
            "num_episodes",
            len(episode_metrics),
        ),
        "episode_metrics": episode_metrics,
        "summary_metrics": summary_metrics,
    }