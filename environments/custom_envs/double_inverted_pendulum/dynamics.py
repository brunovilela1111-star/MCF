# dinâmica de ambiente: estado atual + ação aplicada → próximo estado

from __future__ import annotations
from typing import Any, Optional
import numpy as np

def build_initial_double_pendulum_state(
    simulator: Any,
    initial_theta_factor: float,
) -> np.ndarray:
    """ Define e devolve o estado inicial do double inverted pendulum. Estado: [pos, theta1, theta2, dpos, dtheta1, dtheta2] """
    # carrinho começa no centro
    simulator.x0["pos"] = 0.0 
    # Os dois pêndulos começam perto da posição definida por initial_theta_factor.
    simulator.x0["theta"] = np.array([[initial_theta_factor * np.pi],[initial_theta_factor * np.pi],])
    # começa tudo sem velocidade
    simulator.x0["dpos"] = 0.0
    simulator.x0["dtheta"] = np.array([[0.0],[0.0],])
    # converte o estado inicial do do-mpc para um vetor numpy.
    x0 = simulator.x0.cat.full().flatten()
    simulator.init_algebraic_variables()
    return x0.astype(np.float32)

def simulate_double_pendulum_step(
    simulator: Any,
    action: Any,
) -> np.ndarray:
    """ Executa um passo no simulador do-mpc e devolve o próximo estado. A ação é convertida para o formato esperado:shape = (1, 1) """
    # converte ação para formato certo
    u = process_double_pendulum_action(action)
    #simulador calcula próximo estado
    x_next = simulator.make_step(u).flatten()
    return x_next.astype(np.float32)

def process_double_pendulum_action(action: Any) -> np.ndarray:
    """ Converte a ação para o formato interno usado pelo simulador do-mpc. """
    return np.array(action, dtype=float).reshape(1, 1)

def get_double_pendulum_observation(
    state: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """ Neste ambiente, a observação é igual ao estado. """
    if state is None:
        return None
    return state.copy()

def extract_double_pendulum_state_components(
    state: np.ndarray,
) -> dict:
    """ Extrai os componentes principais do estado. Útil para reward, métricas, debugging e futuras análises. """
    return {
        "pos": float(state[0]),
        "theta1": float(state[1]),
        "theta2": float(state[2]),
        "dpos": float(state[3]),
        "dtheta1": float(state[4]),
        "dtheta2": float(state[5]),
    }