# calcular métricas genéricas que servem para qualquer controlador e qualquer ambiente. 
# Este ficheiro contém funções para reward, ações, tracking, tempos e agregação.
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math

# Funções auxiliares
def _safe_mean(values: List[float]) -> Optional[float]:
    # Calcula média apenas se existirem valores.
    if not values:
        return None
    return float(sum(values) / len(values))

def _safe_max(values: List[float]) -> Optional[float]:
    # Calcula máximo apenas se existirem valores.
    if not values:
        return None
    return float(max(values))

def _safe_min(values: List[float]) -> Optional[float]:
    # Calcula mínimo apenas se existirem valores.
    if not values:
        return None
    return float(min(values))

def _safe_std(values: List[float]) -> Optional[float]:
    # Calcula desvio padrão de forma segura.
    if not values:
        return None

    # Se só existir um valor, o desvio padrão é 0.
    if len(values) == 1:
        return 0.0

    mean_value = sum(values) / len(values)
    variance = sum((v - mean_value) ** 2 for v in values) / len(values)

    return float(math.sqrt(variance))

def _to_scalar(value: Any) -> Optional[float]:
    """ Converte diferentes formatos de ação/erro para um número. """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, (list, tuple)):
        flat = []

        for v in value:
            scalar = _to_scalar(v)
            if scalar is not None:
                flat.append(scalar)

        if not flat:
            return None

        # Soma valores absolutos para obter uma magnitude escalar.
        return float(sum(abs(v) for v in flat))

    if hasattr(value, "flatten"):
        try:
            flat = value.flatten().tolist()
            return _to_scalar(flat)
        except Exception:
            return None

    return None

# Extração de dados
def _extract_rewards(episode_result: Dict[str, Any]) -> List[float]:
    # Extrai a lista de rewards do episódio.
    return [float(r) for r in episode_result.get("rewards", [])]

def _extract_actions(episode_result: Dict[str, Any]) -> List[float]:
    # Extrai ações e converte cada uma para magnitude escalar.
    actions = []

    for action in episode_result.get("actions", []):
        scalar = _to_scalar(action)
        if scalar is not None:
            actions.append(float(abs(scalar)))

    return actions

def _extract_action_deltas(episode_result: Dict[str, Any]) -> List[float]:
    # Calcula variação entre ações consecutivas.
    scalar_actions = []

    for action in episode_result.get("actions", []):
        scalar = _to_scalar(action)
        if scalar is not None:
            scalar_actions.append(float(scalar))

    if len(scalar_actions) < 2:
        return []

    return [
        float(abs(scalar_actions[k] - scalar_actions[k - 1]))
        for k in range(1, len(scalar_actions))
    ]

def _extract_infos(episode_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Extrai apenas entradas info que sejam dicionários.
    return [
        info for info in episode_result.get("infos", [])
        if isinstance(info, dict)
    ]

def _extract_info_metric(
    episode_result: Dict[str, Any],
    metric_name: str,
) -> List[float]:
    # Extrai uma métrica numérica guardada dentro dos infos.
    values = []

    for info in _extract_infos(episode_result):
        scalar = _to_scalar(info.get(metric_name))
        if scalar is not None:
            values.append(float(abs(scalar)))

    return values

def _extract_bool_info_metric(
    episode_result: Dict[str, Any],
    metric_name: str,
) -> List[bool]:
    # Extrai uma métrica booleana guardada dentro dos infos.
    values = []

    for info in _extract_infos(episode_result):
        if metric_name in info:
            values.append(bool(info[metric_name]))

    return values

def _extract_tracking_errors(episode_result: Dict[str, Any]) -> List[float]:
    # Extrai erros de tracking, se o ambiente os fornecer no info.
    errors = []

    for info in _extract_infos(episode_result):
        raw_error = info.get("state_error", info.get("tracking_error"))
        scalar_error = _to_scalar(raw_error)

        if scalar_error is not None:
            errors.append(float(abs(scalar_error)))

    return errors

def _extract_step_times(episode_result: Dict[str, Any]) -> List[float]:
    # Extrai tempos de execução de cada passo.
    return [float(t) for t in episode_result.get("step_times", [])]

# Métricas de reward
def compute_total_reward(episode_result: Dict[str, Any]) -> float:
    # Soma todas as rewards do episódio.
    return float(sum(_extract_rewards(episode_result)))

def compute_num_steps(episode_result: Dict[str, Any]) -> int:
    # Número de passos executados no episódio.
    return int(episode_result.get("num_steps", 0))

def compute_average_reward(episode_result: Dict[str, Any]) -> float:
    # Reward média por passo.
    rewards = _extract_rewards(episode_result)

    if not rewards:
        return 0.0

    return float(sum(rewards) / len(rewards))

def compute_max_reward(episode_result: Dict[str, Any]) -> float:
    # Melhor reward obtida num passo.
    rewards = _extract_rewards(episode_result)

    if not rewards:
        return 0.0

    return float(max(rewards))

def compute_min_reward(episode_result: Dict[str, Any]) -> float:
    # Pior reward obtida num passo.
    rewards = _extract_rewards(episode_result)

    if not rewards:
        return 0.0

    return float(min(rewards))

# Métricas de ação/controlo
def compute_mean_action_magnitude(
    episode_result: Dict[str, Any],
) -> Optional[float]:
    # Magnitude média das ações.
    return _safe_mean(_extract_actions(episode_result))

def compute_max_action_magnitude(
    episode_result: Dict[str, Any],
) -> Optional[float]:
    # Maior ação aplicada.
    return _safe_max(_extract_actions(episode_result))

def compute_action_smoothness(
    episode_result: Dict[str, Any],
) -> Optional[float]:
    # Mede a suavidade da ação.
    # Quanto menor, menos bruscas são as mudanças de ação.
    return _safe_mean(_extract_action_deltas(episode_result))

def compute_control_energy(
    episode_result: Dict[str, Any],
) -> Optional[float]:
    # Soma dos quadrados das ações.
    # Representa esforço/energia de controlo.
    actions = _extract_actions(episode_result)

    if not actions:
        return None

    return float(sum(a**2 for a in actions))

# Métricas de tracking
def compute_mean_tracking_error(
    episode_result: Dict[str, Any],
) -> Optional[float]:
    # Erro médio de seguimento.
    return _safe_mean(_extract_tracking_errors(episode_result))

def compute_max_tracking_error(
    episode_result: Dict[str, Any],
) -> Optional[float]:
    # Maior erro de seguimento.
    return _safe_max(_extract_tracking_errors(episode_result))

def compute_final_tracking_error(
    episode_result: Dict[str, Any],
) -> Optional[float]:
    # Último erro de tracking registado.
    errors = _extract_tracking_errors(episode_result)

    if not errors:
        return None

    return float(errors[-1])

# Métricas genéricas info
def compute_mean_info_metric(
    episode_result: Dict[str, Any],
    metric_name: str,
) -> Optional[float]:
    # Média de uma métrica existente no info.
    return _safe_mean(_extract_info_metric(episode_result, metric_name))

def compute_max_info_metric(
    episode_result: Dict[str, Any],
    metric_name: str,
) -> Optional[float]:
    # Máximo de uma métrica existente no info.
    return _safe_max(_extract_info_metric(episode_result, metric_name))

def compute_final_info_metric(
    episode_result: Dict[str, Any],
    metric_name: str,
) -> Optional[float]:
    # Valor final de uma métrica existente no info.
    values = _extract_info_metric(episode_result, metric_name)

    if not values:
        return None

    return float(values[-1])

def compute_success_rate_from_info(
    episode_result: Dict[str, Any],
    metric_name: str,
) -> Optional[float]:
    # Percentagem de passos onde uma condição booleana foi verdadeira.
    values = _extract_bool_info_metric(episode_result, metric_name)

    if not values:
        return None

    return float(sum(values) / len(values))

def compute_final_success_from_info(
    episode_result: Dict[str, Any],
    metric_name: str,
) -> Optional[bool]:
    # Valor booleano final de uma condição guardada no info.
    values = _extract_bool_info_metric(episode_result, metric_name)

    if not values:
        return None

    return bool(values[-1])

# Métricas de tempo
def compute_mean_step_time(
    episode_result: Dict[str, Any],
) -> Optional[float]:
    # Tempo médio por passo.
    return _safe_mean(_extract_step_times(episode_result))


def compute_total_step_time(
    episode_result: Dict[str, Any],
) -> Optional[float]:
    # Tempo total gasto nos passos do episódio.
    step_times = _extract_step_times(episode_result)

    if not step_times:
        return None

    return float(sum(step_times))

# Resumo de um episódio

def summarize_episode_metrics(
    episode_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calcula todas as métricas genéricas de um episódio.
    """

    return {
        "controller_name": episode_result.get("controller_name"),
        "controller_type": episode_result.get("controller_type"),
        "controller_family": episode_result.get("controller_family"),
        "environment_name": episode_result.get("environment_name"),
        "environment_type": episode_result.get("environment_type"),
        "total_reward": compute_total_reward(episode_result),
        "num_steps": compute_num_steps(episode_result),
        "average_reward": compute_average_reward(episode_result),
        "max_reward": compute_max_reward(episode_result),
        "min_reward": compute_min_reward(episode_result),
        "mean_action_magnitude": compute_mean_action_magnitude(episode_result),
        "max_action_magnitude": compute_max_action_magnitude(episode_result),
        "action_smoothness": compute_action_smoothness(episode_result),
        "control_energy": compute_control_energy(episode_result),
        "mean_tracking_error": compute_mean_tracking_error(episode_result),
        "max_tracking_error": compute_max_tracking_error(episode_result),
        "final_tracking_error": compute_final_tracking_error(episode_result),
        "mean_step_time": compute_mean_step_time(episode_result),
        "total_step_time": compute_total_step_time(episode_result),
        "terminated": episode_result.get("terminated", False),
        "truncated": episode_result.get("truncated", False),
    }

def summarize_multiple_episodes(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Calcula métricas para vários episódios.
    return [summarize_episode_metrics(result) for result in results]

# Agregação

def aggregate_metric(
    metrics_list: List[Dict[str, Any]],
    metric_name: str,
) -> Dict[str, Optional[float]]:
    """
    Agrega uma métrica ao longo de vários episódios.
    """

    values = [
        m[metric_name]
        for m in metrics_list
        if metric_name in m and m[metric_name] is not None
    ]

    return {
        "mean": _safe_mean(values),
        "std": _safe_std(values),
        "min": _safe_min(values),
        "max": _safe_max(values),
    }