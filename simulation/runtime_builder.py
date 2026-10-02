# constrói todos os objetos necessários para executar a arquitetura: ambiente, controlador, runner e evaluator.
# valida compatibilidade e carrega modelos treinados se e quando necessário.

from __future__ import annotations
from pathlib import Path
from registries import ENVIRONMENT_REGISTRY
from factories import create_environment, create_controller
from simulation.simulation_runner import SimulationRunner
from evaluation.evaluator import Evaluator
from utils.validators import (validate_controller_environment_compatibility,validate_execution_metadata,)

def build_trained_model_path(
    controller_type: str,
    environment_type: str,
    model_name: str | None = None,
) -> Path:
    """
    Constrói o caminho onde está guardado um modelo RL treinado.

    Exemplo:
    models/rl/ppo/cartpole/ppo_cartpole
    """
    # Pasta base dos modelos RL treinados.
    base_dir = Path("models") / "rl" / controller_type / environment_type
    # Se o utilizador não indicar nome, usa nome padrão.
    final_name = model_name or f"{controller_type}_{environment_type}"
    # Devolve caminho final.
    return base_dir / final_name

def build_controller(
    controller_type: str,
    env,
    overrides: dict | None = None,
):
    """
    Cria o controlador através da controller_factory.
    A factory junta:
    defaults + overrides + classe correta.
    """
    return create_controller(
        controller_type,
        env=env,
        overrides=overrides or {},
    )

def prepare_controller(
    controller,
    controller_type: str,
    environment_type: str,
    use_trained_model: bool,
    trained_model_name: str | None,
) -> None:
    """ Inicializa o controlador e, se for RL, carrega modelo treinado quando pedido.  """
    # Inicializa o controlador.
    # Em RL cria o modelo.
    # Em MPC prepara o solver/modelo interno.
    controller.initialize()
    # Verifica se o controlador é RL ou MPC.
    controller_family = getattr(controller, "family", None)
    # Se for RL e o utilizador pediu modelo treinado, carrega o modelo.
    if controller_family == "rl" and use_trained_model:
        trained_model_path = build_trained_model_path(
            controller_type=controller_type,
            environment_type=environment_type,
            model_name=trained_model_name,
        )
        # Carrega modelo treinado.
        controller.load(str(trained_model_path))
        print(f"[INFO] Modelo treinado carregado de: {trained_model_path}")

def _build_environment_overrides(config: dict) -> dict:
    """ Constrói os overrides finais do ambiente.
    Também ativa render_mode='human' quando:
    - render=True;
    - o ambiente suporta render_mode."""
    # Vai buscar a especificação do ambiente ao registry.
    environment_spec = ENVIRONMENT_REGISTRY[config["environment_type"]]
    final_env_overrides = {}
    # Se render estiver ativo e o ambiente suportar render_mode,
    # adiciona render_mode='human'.
    if config["render"] and environment_spec.get("supports_render_mode", False):
        final_env_overrides["render_mode"] = "human"
    # Junta overrides definidos pelo utilizador no main_config.
    final_env_overrides.update(config.get("env_overrides", {}))
    
    # FMU: o tempo de simulação passa a ser controlado
    # diretamente pelo max_steps definido no main_config.
    if config["environment_type"].endswith("_fmu"):
        final_env_overrides["max_time"] = config["max_steps"]

    return final_env_overrides

def build_runtime(config: dict) -> dict:
    """ Constrói o runtime completo da arquitetura.
    Runtime = conjunto de objetos necessários para correr:
    - env;
    - controller;
    - runner;
    - evaluator.
    """
    # Valida se o controlador pode ser usado no ambiente escolhido.
    validate_controller_environment_compatibility(
        controller_type=config["controller_type"],
        environment_type=config["environment_type"],
    )
    # Valida metadados da execução.
    validate_execution_metadata(
        mode=config["mode"],
        controller_type=config["controller_type"],
        environment_type=config["environment_type"],
        render=config["render"],
    )
    # Cria overrides finais do ambiente.
    final_env_overrides = _build_environment_overrides(config)
    # Cria o ambiente através da environment_factory.
    env = create_environment(
        config["environment_type"],
        overrides=final_env_overrides,
    )
    # Cria o controlador através da controller_factory.
    controller = build_controller(
        config["controller_type"],
        env,
        overrides=config.get("controller_overrides", {}),
    )
    # Cria o avaliador.
    evaluator = Evaluator()
    # Cria o runner que liga controlador e ambiente.
    runner = SimulationRunner(
        controller=controller,
        environment=env,
    )
    # Inicializa controlador e carrega modelo treinado se necessário.
    prepare_controller(
        controller=controller,
        controller_type=config["controller_type"],
        environment_type=config["environment_type"],
        use_trained_model=config["use_trained_model"],
        trained_model_name=config["trained_model_name"],
    )
    # Devolve tudo organizado num dicionário.
    return {
        "env": env,
        "controller": controller,
        "runner": runner,
        "evaluator": evaluator,
    }