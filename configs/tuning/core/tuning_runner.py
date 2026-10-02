from __future__ import annotations
from typing import Any, Callable, Dict, List
from configs.tuning.core.parameter_grid import generate_parameter_grid
from configs.tuning.core.tuning_result import TuningResult

def run_grid_search(
    search_space: Dict[str, List[Any]],
    evaluate_function: Callable[[Dict[str, Any]], Dict[str, Any]],
    score_key: str,
    metadata: Dict[str, Any] | None = None,    
) -> List[TuningResult]:
    """ Executa um grid search simples."""
    parameter_grid = generate_parameter_grid(search_space)
    results: List[TuningResult] = []

    for parameters in parameter_grid:
        metrics = evaluate_function(parameters)

        if score_key not in metrics:
            raise KeyError(
                f"A métrica '{score_key}' não existe nas métricas devolvidas: "
                f"{list(metrics.keys())}"
            )

        results.append(
            TuningResult(
                parameters=parameters,
                metrics=metrics,
                score=float(metrics[score_key]),
                metadata=metadata or {},
            )
        )

    return results

def get_best_result(
    results: List[TuningResult],
    maximize: bool = True,
) -> TuningResult:
    """ Devolve o melhor resultado de tuning. Pode maximizar ou minimizar """
    if not results:
        raise ValueError("A lista de resultados está vazia.")

    if maximize:
        return max(results, key=lambda result: result.score)

    return min(results, key=lambda result: result.score)