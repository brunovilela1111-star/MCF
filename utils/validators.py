# utils/validators.py
from __future__ import annotations
from registries import CONTROLLER_REGISTRY, ENVIRONMENT_REGISTRY

def validate_main_choices(
    mode: str,
    controller_type: str,
) -> None:
    """ Valida as escolhas principais feitas pelo utilizador."""
    valid_modes = {"train", "simulate", "evaluate", "compare"}

    if mode not in valid_modes:
        raise ValueError(
            "mode deve ser 'train', 'simulate', 'evaluate' ou 'compare'."
        )

    if controller_type not in CONTROLLER_REGISTRY:
        raise ValueError(f"Controlador desconhecido: '{controller_type}'.")

    controller_family = CONTROLLER_REGISTRY[controller_type]["family"]

    if mode == "train" and controller_family != "rl":
        raise ValueError(
            "Nesta fase, o modo 'train' está disponível apenas para controladores RL."
        )

def validate_controller_environment_compatibility(
    controller_type: str,
    environment_type: str,
) -> None:
    """ Valida compatibilidade entre controlador e ambiente com base em metadata dos registries."""
    if controller_type not in CONTROLLER_REGISTRY:
        raise ValueError(f"Controlador desconhecido: '{controller_type}'.")

    if environment_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Ambiente desconhecido: '{environment_type}'.")

    controller_spec = CONTROLLER_REGISTRY[controller_type]
    environment_spec = ENVIRONMENT_REGISTRY[environment_type]

    supported_action_spaces = controller_spec.get("supported_action_spaces", [])
    environment_action_space = environment_spec.get("action_space_type")

    if environment_action_space is None:
        return

    if supported_action_spaces and environment_action_space not in supported_action_spaces:
        raise ValueError(
            f"O controlador '{controller_type}' não é compatível com o ambiente "
            f"'{environment_type}'. "
            f"Action space do ambiente: '{environment_action_space}'. "
            f"Action spaces suportados: {supported_action_spaces}."
        )

def validate_execution_metadata(
    mode: str,
    controller_type: str,
    environment_type: str,
    render: bool,
) -> None:
    """ Valida regras adicionais de execução com base em metadata dos registries."""
    if controller_type not in CONTROLLER_REGISTRY:
        raise ValueError(f"Controlador desconhecido: '{controller_type}'.")

    if environment_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Ambiente desconhecido: '{environment_type}'.")

    controller_spec = CONTROLLER_REGISTRY[controller_type]
    environment_spec = ENVIRONMENT_REGISTRY[environment_type]

    if mode == "train":
        if not controller_spec.get("supports_training", False):
            raise ValueError(
                f"O controlador '{controller_type}' não suporta modo de treino."
            )

        if not environment_spec.get("supports_training", False):
            raise ValueError(
                f"O ambiente '{environment_type}' não suporta modo de treino."
            )

    if render and not environment_spec.get("supports_render_mode", False):
        print(
            f"[AVISO] O ambiente '{environment_type}' não suporta render_mode "
            "na criação. A execução continuará sem esse suporte específico."
        )