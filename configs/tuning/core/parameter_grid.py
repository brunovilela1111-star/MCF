from __future__ import annotations
from itertools import product
from typing import Any, Dict, List

def generate_parameter_grid(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """  Gera todas as combinações possíveis de parâmetros a partir de um search space. """
    if not search_space:
        return [{}]

    keys = list(search_space.keys())
    values = list(search_space.values())

    combinations = []

    for combination in product(*values):
        config = dict(zip(keys, combination))
        combinations.append(config)

    return combinations