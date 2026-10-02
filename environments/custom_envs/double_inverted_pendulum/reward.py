#reward function para RL e métricas específicas do ambiente

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

# calcula a recompensa em cada passo.
def compute_double_pendulum_reward(
    state: np.ndarray,
    action: np.ndarray,
    config: Dict[str, Any],
) -> float:
    pos = state[0]
    theta1 = state[1]
    theta2 = state[2]
    dpos = state[3]
    dtheta1 = state[4]
    dtheta2 = state[5]
    # quanto mais perto da vertical, maior o valor.
    upright_reward = config["angle_weight"]*(np.cos(theta1) + np.cos(theta2))
    position_penalty = config["position_weight"] * pos**2 # Penaliza o carrinho afastado do centro.
    velocity_penalty = config["velocity_weight"] * (dpos**2 + dtheta1**2 + dtheta2**2 )# Penaliza velocidades elevadas.
    action_penalty = config["action_weight"] * float(action[0]) ** 2 # Penaliza força exagerada.
    # reward function
    reward = (upright_reward-position_penalty-velocity_penalty-action_penalty)
    # # Bónus perto da vertical
    # if (
    #     abs(theta1) < config["near_upright_angle_threshold"]
    #     and abs(theta2) < config["near_upright_angle_threshold"]
    # ):
    #     reward += config["near_upright_bonus"]
    # # Bónus de estabilização
    # if (
    #     abs(theta1) < config["stabilized_angle_threshold"]
    #     and abs(theta2) < config["stabilized_angle_threshold"]
    #     and abs(dtheta1) < config["stabilized_velocity_threshold"]
    #     and abs(dtheta2) < config["stabilized_velocity_threshold"]
    # ):
    #     reward += config["stabilized_bonus"]
    return float(reward)

# Verifica se o episódio deve terminar.
def check_double_pendulum_termination(
    state: np.ndarray,
    config: Dict[str, Any],
) -> bool:
    pos = state[0]
    theta1 = state[1]
    theta2 = state[2]

    if abs(pos) > config["position_threshold"]: # carrinho saiu dos limites.
        return True

    # if (
    #     abs(theta1) > config["angle_threshold"]
    #     or abs(theta2) > config["angle_threshold"]
    # ): # Algum pêndulo passou o limite angular.
    #     return True

    return False

#Calcula métricas específicas:
def compute_double_pendulum_specific_metrics(
    state: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    pos = float(state[0])
    theta1 = float(state[1])
    theta2 = float(state[2])
    dtheta1 = float(state[4])
    dtheta2 = float(state[5])
    theta1_upright_error = abs(theta1)
    theta2_upright_error = abs(theta2)
    mean_upright_error = 0.5 * (
        theta1_upright_error + theta2_upright_error
    )
    is_near_upright = (
        theta1_upright_error < config["near_upright_angle_threshold"]
        and theta2_upright_error < config["near_upright_angle_threshold"]
    )
    is_upright_stabilized = (
        theta1_upright_error < config["stabilized_angle_threshold"]
        and theta2_upright_error < config["stabilized_angle_threshold"]
        and abs(dtheta1) < config["stabilized_velocity_threshold"]
        and abs(dtheta2) < config["stabilized_velocity_threshold"]
    )
    return {
        "theta1_upright_error": theta1_upright_error,
        "theta2_upright_error": theta2_upright_error,
        "mean_upright_error": float(mean_upright_error),
        "cart_position_error": abs(pos),
        "is_near_upright": bool(is_near_upright),
        "is_upright_stabilized": bool(is_upright_stabilized),
    }

def build_double_pendulum_info(
    action: np.ndarray,
    state: Optional[np.ndarray],
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "action": action.flatten().copy(),
        "state": state.copy() if state is not None else None,
    }

    if state is not None:
        info.update(
            compute_double_pendulum_specific_metrics(
                state=state,
                config=config,
            )
        )

    if extra:
        info.update(extra)

    return info