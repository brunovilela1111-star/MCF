# criar controladores automaticamente a partir do CONTROLLER_REGISTRY, dos defaults e dos overrides. 
# Também carrega configurações MPC específicas do ambiente quando necessário.
# cria o controlador certo automaticamente
from __future__ import annotations
from importlib import import_module
from typing import Any, Dict, Optional
from configs.controllers import RL_DEFAULTS, MPC_DEFAULTS
from registries import CONTROLLER_REGISTRY, MPC_MODEL_REGISTRY

def _load_mpc_model_config(
    controller_type: str,
    env: Any,
) -> Dict[str, Any]:
    """  Carrega automaticamente a configuração do modelo MPC associada a um controlador e a um ambiente.
    Exemplo:
    controller_type = "nonlinear_mpc"
    env.name = "double_inverted_pendulum"
    Vai procurar no MPC_MODEL_REGISTRY a chave:
    ("nonlinear_mpc", "double_inverted_pendulum") """
    # Se não existir ambiente, não há modelo MPC específico a carregar.
    if env is None:
        return {}
    # Obtém o nome do ambiente.
    env_name = getattr(env, "name", None)
    # Se o ambiente não tiver nome, não há como procurar no registry.
    if env_name is None:
        return {}
    # Cria a chave de procura no registry.
    key = (controller_type, env_name)
    # Se não existir configuração MPC para esta combinação,
    # devolve dicionário vazio.
    if key not in MPC_MODEL_REGISTRY:
        return {}
    # Obtém a especificação do modelo MPC.
    spec = MPC_MODEL_REGISTRY[key]
    # Importa dinamicamente o módulo onde está a função construtora.
    module = import_module(spec["module_path"])
    # Vai buscar a função pelo nome.
    builder = getattr(module, spec["callable_name"])
    # Executa a função e devolve a configuração MPC.
    return builder()

def _attach_controller_metadata(
    controller: Any,
    family: str,
    controller_type: str,
) -> Any:
    """ Anexa metadados ao controlador criado.Estes metadados são usados depois pelo runner, evaluator e validators."""
    # Define se o controlador é "rl" ou "mpc".
    controller.family = family
    # Guarda a chave usada no registry.
    # Exemplo: "ppo", "sac", "nonlinear_mpc".
    controller.controller_type_key = controller_type
    return controller

def _load_controller_defaults(
    controller_type: str,
    family: str,
    defaults_key: str,
    env: Any,
) -> Dict[str, Any]:
    """ Carrega os defaults do controlador com base na sua família.
    Se for RL:
    - vai buscar a RL_DEFAULTS.
    Se for MPC:
    - vai buscar a MPC_DEFAULTS;
    - junta também a configuração do modelo MPC associada ao ambiente. """
    # Controladores RL usam os defaults de RL.
    if family == "rl":
        return RL_DEFAULTS.get(defaults_key, {})
    # Controladores MPC usam defaults MPC + configuração específica do modelo.
    if family == "mpc":
        defaults = MPC_DEFAULTS.get(defaults_key, {})
        # Carrega configuração específica do modelo MPC.
        # Exemplo: modelo NMPC do double inverted pendulum.
        model_config = _load_mpc_model_config(controller_type, env)
        # Junta defaults gerais + configuração do modelo.
        # Se houver chaves iguais, model_config sobrepõe defaults.
        return {
            **defaults,
            **model_config,
        }
    # Se aparecer uma família desconhecida, lança erro.
    raise ValueError(f"Família de controlador desconhecida: {family}")

def create_controller(
    controller_type: str,
    env: Any = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """ Cria um controlador a partir do registry e dos defaults. """
    # Verifica se o controlador pedido está registado.
    if controller_type not in CONTROLLER_REGISTRY:
        raise ValueError(f"Controlador desconhecido: {controller_type}")
    # Obtém informação do controlador no registry.
    spec = CONTROLLER_REGISTRY[controller_type]
    # Família do controlador: "rl" ou "mpc".
    family = spec["family"]
    # Chave usada para ir buscar defaults.
    defaults_key = spec["defaults_key"]
    # Carrega os defaults adequados.
    defaults = _load_controller_defaults(
        controller_type=controller_type,
        family=family,
        defaults_key=defaults_key,
        env=env,
    )
    # Junta defaults + overrides do utilizador.
    # Os overrides têm prioridade sobre os defaults.
    final_config = {
        **defaults,
        **(overrides or {}),
    }
    # Importa dinamicamente o módulo onde está a classe do controlador.
    module = import_module(spec["module_path"])
    # Obtém a classe pelo nome.
    cls = getattr(module, spec["class_name"])
    # Alguns controladores precisam receber o ambiente.
    # Exemplo: RL precisa de env para treinar.
    if spec.get("requires_env", False):
        controller = cls(
            env=env,
            **final_config,
        )
    # Outros não precisam receber env diretamente.
    # Exemplo: alguns MPC recebem apenas configs/modelo.
    else:
        controller = cls(**final_config)
    # Adiciona metadados e devolve o controlador.
    return _attach_controller_metadata(
        controller=controller,
        family=family,
        controller_type=controller_type,
    )