# results/io.py
from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_RESULTS_DIR = Path("results")

def _timestamp() -> str:
    """ Devolve timestamp compacto para nome de ficheiro. """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(path: Path) -> None:
    """ Garante que a diretoria existe. """
    path.mkdir(parents=True, exist_ok=True)

def _make_filename(
    prefix: str,
    controller_name: str,
    environment_name: str,
    suffix: str = "json",
) -> str:
    """ Cria um nome de ficheiro consistente para resultados. """
    safe_controller = str(controller_name).replace(" ", "_")
    safe_environment = str(environment_name).replace(" ", "_")

    return f"{prefix}_{safe_controller}_{safe_environment}_{_timestamp()}.{suffix}"


def _to_serializable(obj: Any) -> Any:
    """ Converte objetos para um formato serializável em JSON. """
    if is_dataclass(obj):
        return _to_serializable(asdict(obj))

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    return str(obj)


def _save_json(data: Any, output_path: Path) -> Path:
    """ Guarda dados em JSON e devolve o caminho final. """
    serializable_data = _to_serializable(data)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=4, ensure_ascii=False)

    return output_path


def save_training_result(training_result: Any) -> Path:
    """ Guarda um TrainingResult em JSON. """
    output_dir = BASE_RESULTS_DIR / "training"
    _ensure_dir(output_dir)

    filename = _make_filename(
        prefix="training",
        controller_name=training_result.controller_name,
        environment_name=training_result.environment_name,
    )

    output_path = output_dir / filename
    return _save_json(training_result, output_path)


def save_simulation_result(simulation_result: Any) -> Path:
    """ Guarda um SimulationResult em JSON. """
    output_dir = BASE_RESULTS_DIR / "simulation"
    _ensure_dir(output_dir)

    filename = _make_filename(
        prefix="simulation",
        controller_name=simulation_result.controller_name,
        environment_name=simulation_result.environment_name,
    )

    output_path = output_dir / filename
    return _save_json(simulation_result, output_path)


def save_evaluation_result(evaluation_result: Any) -> Path:
    """ Guarda um EvaluationResult em JSON. """
    output_dir = BASE_RESULTS_DIR / "evaluation"
    _ensure_dir(output_dir)

    filename = _make_filename(
        prefix="evaluation",
        controller_name=evaluation_result.controller_name,
        environment_name=evaluation_result.environment_name,
    )

    output_path = output_dir / filename
    return _save_json(evaluation_result, output_path)

def save_comparison_result(comparison_result: Any) -> Path:
    """ Guarda um ComparisonResult em JSON. """
    output_dir = BASE_RESULTS_DIR / "comparison"
    _ensure_dir(output_dir)

    filename = (
        f"comparison_"
        f"{comparison_result.environment_name}_"
        f"{_timestamp()}.json"
    )

    output_path = output_dir / filename
    return _save_json(comparison_result, output_path)