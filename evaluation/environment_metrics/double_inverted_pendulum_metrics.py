# métricas especificas de duplo pendulo invertido
from __future__ import annotations
from typing import Any, Dict, List, Optional

def _safe_mean(values: List[float]) -> Optional[float]:
    # Calcula média apenas se houver valores.
    if not values:
        return None
    return float(sum(values) / len(values))

def _safe_max(values: List[float]) -> Optional[float]:
    # Calcula máximo apenas se houver valores.
    if not values:
        return None
    return float(max(values))

def _safe_final(values: List[float]) -> Optional[float]:
    # Devolve o último valor da lista.
    if not values:
        return None
    return float(values[-1])

def _extract_info_values(
    episode_result: Dict[str, Any],
    key: str,
) -> List[float]:
    """
    Extrai valores numéricos do campo info.
    Exemplo:
    key = "theta1_upright_error"
    """
    values: List[float] = []
    # Percorre todos os infos registados durante o episódio.
    for info in episode_result.get("infos", []):

        # Verifica se info é dicionário e se contém a métrica pedida.
        if isinstance(info, dict) and key in info and info[key] is not None:
            values.append(float(info[key]))
    return values

def _extract_info_bools(
    episode_result: Dict[str, Any],
    key: str,
) -> List[bool]:
    """
    Extrai valores booleanos do campo info.
    Exemplo:
    key = "is_upright_stabilized"
    """
    values: List[bool] = []
    for info in episode_result.get("infos", []):
        if isinstance(info, dict) and key in info:
            values.append(bool(info[key]))

    return values

def _bool_rate(values: List[bool]) -> Optional[float]:
    """
    Calcula a percentagem de valores True.
    Exemplo:
    [True, True, False] = 0.666...
    """
    if not values:
        return None

    return float(sum(values) / len(values))

def compute_double_inverted_pendulum_metrics(
    episode_result: Dict[str, Any],
) -> Dict[str, Any]:
    """ Calcula métricas específicas do ambiente double inverted pendulum. """
    # Extrai erros angulares do primeiro pêndulo.
    theta1_errors = _extract_info_values(
        episode_result,
        "theta1_upright_error",
    )
    # Extrai erros angulares do segundo pêndulo.
    theta2_errors = _extract_info_values(
        episode_result,
        "theta2_upright_error",
    )
    # Extrai erro médio de verticalidade.
    upright_errors = _extract_info_values(
        episode_result,
        "mean_upright_error",
    )
    # Extrai erro da posição do carrinho.
    cart_errors = _extract_info_values(
        episode_result,
        "cart_position_error",
    )
    # Extrai flags que indicam se o sistema esteve perto da vertical.
    near_upright_values = _extract_info_bools(
        episode_result,
        "is_near_upright",
    )
    # Extrai flags que indicam se o sistema esteve estabilizado.
    stabilized_values = _extract_info_bools(
        episode_result,
        "is_upright_stabilized",
    )
    # Devolve métricas específicas organizadas.
    return {
        # Erros médios dos pêndulos.
        "mean_theta1_upright_error": _safe_mean(theta1_errors),
        "mean_theta2_upright_error": _safe_mean(theta2_errors),
        "mean_upright_error": _safe_mean(upright_errors),
        # Erros finais dos pêndulos.
        "final_theta1_upright_error": _safe_final(theta1_errors),
        "final_theta2_upright_error": _safe_final(theta2_errors),
        "final_upright_error": _safe_final(upright_errors),
        # Erros da posição do carrinho.
        "mean_cart_position_error": _safe_mean(cart_errors),
        "max_cart_position_error": _safe_max(cart_errors),
        "final_cart_position_error": _safe_final(cart_errors),
        # Percentagem de tempo perto da vertical.
        "near_upright_rate": _bool_rate(near_upright_values),
        # Percentagem de tempo estabilizado.
        "stabilized_rate": _bool_rate(stabilized_values),
        # Estado final: ficou estabilizado ou não?
        "final_is_upright_stabilized": (
            stabilized_values[-1] if stabilized_values else None
        ),
    }