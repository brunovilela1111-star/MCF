#criar ambientes automaticamente a partir do ENVIRONMENT_REGISTRY, da config local do ambiente e dos overrides.
#cria mabiente certo automaticamente
from __future__ import annotations
from importlib import import_module
from typing import Any, Dict, Optional
from registries import ENVIRONMENT_REGISTRY

def _load_environment_defaults(
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """ Carrega os defaults/configuração base de um ambiente.
    A informação sobre onde está a config vem do ENVIRONMENT_REGISTRY.
    Exemplo:
    config_module = "environments.custom_envs.double_inverted_pendulum.config"
    config_name = "DOUBLE_INVERTED_PENDULUM_CONFIG"
    """
    # Caminho do módulo onde está a configuração do ambiente.
    config_module_path = spec.get("config_module")
    # Nome da variável de configuração dentro desse módulo.
    config_name = spec.get("config_name")
    # Se o ambiente não tiver config associada, devolve vazio.
    if not config_module_path or not config_name:
        return {}
    # Importa o módulo da configuração.
    module = import_module(config_module_path)
    # Verifica se a variável existe dentro do módulo.
    if not hasattr(module, config_name):
        raise AttributeError(
            f"Config '{config_name}' não encontrada em '{config_module_path}'."
        )
    # Devolve uma cópia da configuração.
    # Usar copy evita alterar diretamente o dicionário original.
    return getattr(module, config_name).copy()

def create_environment(
    env_type: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """ Cria um ambiente a partir do registry e da config local do próprio ambiente."""
    # Verifica se o ambiente pedido está registado.
    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Ambiente desconhecido: {env_type}")
    # Obtém especificação do ambiente.
    spec = ENVIRONMENT_REGISTRY[env_type]
    # Carrega defaults locais do ambiente.
    defaults = _load_environment_defaults(spec)
    # Junta defaults + overrides.
    # Overrides têm prioridade.
    final_config = {
        **defaults,
        **(overrides or {}),
    }
    # Verifica se o ambiente suporta render_mode na criação.
    supports_render_mode = spec.get("supports_render_mode", False)
    # Lê valor de render_mode, se existir.
    render_mode_value = final_config.get("render_mode", None)
    # Se o ambiente não suporta render_mode, remove esse campo.
    if not supports_render_mode and "render_mode" in final_config:
        final_config.pop("render_mode")
        # Se o utilizador tentou usar render_mode, imprime aviso.
        if render_mode_value is not None:
            print(
                f"[AVISO] O ambiente '{env_type}' não suporta render_mode na criação. "
                "A execução continuará sem esse suporte específico."
            )
    # Importa dinamicamente o módulo onde está a classe do ambiente.
    module = import_module(spec["module_path"])
    # Obtém a classe pelo nome.
    cls = getattr(module, spec["class_name"])
    # Cria o ambiente passando a config final.
    env = cls(
        config=final_config,
    )
    # Adiciona metadados úteis ao ambiente.
    env.env_type_key = env_type
    env.supports_render_mode = supports_render_mode
    env.backend = spec.get("backend", None)
    env.action_space_type = spec.get("action_space_type", None)
    return env