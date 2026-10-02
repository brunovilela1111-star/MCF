# correr vários episódios de avaliação e devolver resultados, métricas por episódio e resumo final.
from __future__ import annotations
from typing import Any, Dict, List
from evaluation.evaluator import Evaluator
from simulation.runtime_builder import build_runtime

def run_evaluation_episodes(
    config: Dict[str, Any],
    num_episodes: int,
    max_steps: int,
    render: bool = False,
    render_delay: float = 0.0,
) -> Dict[str, Any]:
    """ Executa vários episódios de avaliação."""
    # Lista onde serão guardados os resultados brutos de cada episódio.
    episode_results = []
    # Lista onde serão guardadas as métricas calculadas para cada episódio.
    metrics_list = []
    # Repete a avaliação pelo número de episódios definido.
    for _ in range(num_episodes):
        # Cria um runtime novo para cada episódio.
        # O runtime contém:
        # - ambiente;
        # - controlador;
        # - runner;
        # - evaluator.
        runtime = build_runtime(config)

        try:
            # Corre um episódio completo usando o SimulationRunner.
            episode_result = runtime["runner"].run_episode(
                max_steps=max_steps,
                render=render,
                render_delay=render_delay,
            )

            # Calcula as métricas desse episódio.
            metrics = runtime["evaluator"].evaluate_episode(episode_result)

            # Guarda o resultado bruto.
            episode_results.append(episode_result)

            # Guarda as métricas calculadas.
            metrics_list.append(metrics)

        finally:
            # Garante que o ambiente é fechado no fim do episódio,
            # mesmo que ocorra algum erro.
            if "env" in runtime:
                try:
                    runtime["env"].close()
                except Exception:
                    pass

    # Cria um resumo agregado de todos os episódios.
    summary = Evaluator().summarize_evaluation(metrics_list)

    # Devolve a estrutura final da avaliação.
    return {
        "episode_results": episode_results,
        "metrics_list": metrics_list,
        "summary": summary,
    }