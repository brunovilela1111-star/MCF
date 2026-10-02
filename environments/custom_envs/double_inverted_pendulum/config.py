from __future__ import annotations
import numpy as np

DOUBLE_INVERTED_PENDULUM_CONFIG = {
    # Core
    "initial_theta_factor": 0.9, # posição inicial dos pêndulos.
    "action_limit": 4.0, # limite máximo de froça aplicada ao carrinho

    # Physical parameters
    "dt": 0.04, # tempo de amostragem
    "m0": 0.6, # massa carrinho
    "m1": 0.2, # massa pendulo 1
    "m2": 0.2, # massa pendulo 2
    "g": 9.80665, # valor de gravidade
    "L1": 0.5, # comprimento de pendulo 1
    "L2": 0.5, #comprimento de pendulo 2

    # Simulator - Definem como o do-mpc integra as equações do sistema.
    "simulator_integration_tool": "idas",
    "simulator_abstol": 1e-8,
    "simulator_reltol": 1e-8,

    # Reward weights - pesos influenciam a reward usada pelo RL.
    "position_weight": 1.0, # penaliza afastamento do carrinho
    "angle_weight": 1.0, # peso relacionado com ângulos
    "velocity_weight": 0.0, # penaliza velocidades elevadas
    "action_weight": 0.0, # penaliza força excessiva

    # Termination thresholds - definem limites de quando ep deve terminar
    "position_threshold": 2.0,
    # "angle_threshold": 2.0 * np.pi,

    # Reward bonuses - recompensa extra quando o pêndulo está perto da vertical ou estabilizado.
    "near_upright_angle_threshold": 0.25,
    "near_upright_bonus": 5.0,

    "stabilized_angle_threshold": 0.15,
    "stabilized_velocity_threshold": 0.5,
    "stabilized_bonus": 15.0,

    "termination_penalty": 700.0,
}