# Métricas específicas do ambiente moldes_fmu
from __future__ import annotations
from typing import Any, Dict, List, Optional

def _safe_mean(values: List[float]) -> Optional[float]:
    """Calcula a média apenas se existirem valores."""
    if not values:
        return None
    return float(sum(values) / len(values))

def _safe_max(values: List[float]) -> Optional[float]:
    """Calcula o máximo apenas se existirem valores."""
    if not values:
        return None
    return float(max(values))

def _safe_min(values: List[float]) -> Optional[float]:
    """Calcula o mínimo apenas se existirem valores."""
    if not values:
        return None
    return float(min(values))

def _safe_final(values: List[float]) -> Optional[float]:
    """Devolve o último valor da lista."""
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
        key = "T1_celsius"
        key = "mAgua"
        key = "temperature_error_abs"
    """

    values: List[float] = []

    for info in episode_result.get("infos", []):
        if isinstance(info, dict) and key in info and info[key] is not None:
            values.append(float(info[key]))

    return values


def _within_band_rate(
    errors: List[float],
    tolerance: float,
) -> Optional[float]:
    """
    Calcula a percentagem de tempo em que a temperatura esteve
    dentro da banda aceitável.

    Exemplo:
        erro <= 2 K
    """

    if not errors:
        return None

    inside_band = [
        abs(error) <= tolerance
        for error in errors
    ]

    return float(sum(inside_band) / len(inside_band))

def _settling_time(
    times: List[float],
    errors: List[float],
    tolerance: float,
) -> Optional[float]:
    """
    Calcula o tempo de estabilização.

    Considera-se que o sistema estabilizou quando entra na banda
    de tolerância e permanece dentro dela até ao fim do episódio.

    Exemplo:
        se tolerance = 2 K,
        procura o primeiro instante a partir do qual:
        |erro| <= 2 K até ao final.
    """

    if not times or not errors or len(times) != len(errors):
        return None

    for i in range(len(errors)):
        remaining_errors = errors[i:]

        if all(abs(error) <= tolerance for error in remaining_errors):
            return float(times[i])

    return None

def _overshoot_celsius(
    temperatures_celsius: List[float],
    reference_celsius: Optional[float],
) -> Optional[float]:
    """
    Calcula o overshoot térmico em ºC.
    Overshoot = temperatura máxima acima da referência.
    Se a temperatura nunca ultrapassar a referência, devolve 0.
    """

    if not temperatures_celsius or reference_celsius is None:
        return None

    max_temperature = max(temperatures_celsius)
    overshoot = max_temperature - reference_celsius

    return float(max(0.0, overshoot))

def compute_moldes_metrics(
    episode_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calcula métricas específicas do ambiente moldes_fmu.

    Estas métricas são mais interpretáveis para um sistema térmico
    de moldes de injeção do que apenas reward ou erro genérico.

    Métricas analisadas:
        - temperatura média, máxima, mínima e final;
        - erro térmico médio, máximo e final;
        - percentagem de tempo dentro da banda de tolerância;
        - tempo de estabilização;
        - overshoot térmico;
        - caudal médio e máximo;
        - consumo total aproximado de água.
    """
    # Tempo de simulação em cada passo.
    times = _extract_info_values(
        episode_result,
        "time",
    )

    # Temperatura do Molde 1 em ºC.
    temperatures_celsius = _extract_info_values(
        episode_result,
        "T1_celsius",
    )

    # Erro absoluto em Kelvin.
    # Como diferenças em Kelvin e ºC têm a mesma magnitude,
    # este valor também pode ser interpretado como erro em ºC.
    temperature_errors_abs = _extract_info_values(
        episode_result,
        "temperature_error_abs",
    )

    # Erro com sinal.
    temperature_errors = _extract_info_values(
        episode_result,
        "temperature_error",
    )

    # Caudal de água aplicado pelo controlador.
    water_flows = _extract_info_values(
        episode_result,
        "mAgua",
    )

    # Referência em ºC.
    reference_values_celsius = _extract_info_values(
        episode_result,
        "T1_ref_celsius",
    )

    reference_celsius = (
        reference_values_celsius[0]
        if reference_values_celsius
        else None
    )

    # Tolerância definida no ambiente.
    # Caso não esteja disponível no info, assume-se 2 K.
    tolerance = 2.0

    # Consumo total aproximado de água.
    # Como mAgua está em kg/s e dt ≈ 1 s,
    # a soma aproxima a massa total de água usada em kg.
    total_water_consumption = (
        float(sum(water_flows))
        if water_flows
        else None
    )

    return {
       
        # Métricas de temperatura

        "mean_temperature_celsius": _safe_mean(temperatures_celsius),
        "max_temperature_celsius": _safe_max(temperatures_celsius),
        "min_temperature_celsius": _safe_min(temperatures_celsius),
        "final_temperature_celsius": _safe_final(temperatures_celsius),

        # Métricas de erro térmico

        "mean_temperature_error": _safe_mean(temperature_errors_abs),
        "max_temperature_error": _safe_max(temperature_errors_abs),
        "final_temperature_error": _safe_final(temperature_errors_abs),

        # Métricas de estabilidade

        "temperature_within_band_rate": _within_band_rate(
            errors=temperature_errors_abs,
            tolerance=tolerance,
        ),

        "settling_time": _settling_time(
            times=times,
            errors=temperature_errors_abs,
            tolerance=tolerance,
        ),

        "overshoot_celsius": _overshoot_celsius(
            temperatures_celsius=temperatures_celsius,
            reference_celsius=reference_celsius,
        ),

        # Métricas de utilização de água

        "mean_water_flow": _safe_mean(water_flows),
        "max_water_flow": _safe_max(water_flows),
        "final_water_flow": _safe_final(water_flows),
        "total_water_consumption": total_water_consumption,
    }