# função custo + restrições + referência do NMPC
from __future__ import annotations
from typing import Any, Dict
import numpy as np

# Custo que NMPC precisa minimizar
def build_objective(model: Any, cost_weights: Dict[str, float]):
    pos = model.x["pos"]
    theta = model.x["theta"]
    dpos = model.x["dpos"]
    dtheta = model.x["dtheta"]

    theta1 = theta[0]
    theta2 = theta[1]
    dtheta1 = dtheta[0]
    dtheta2 = dtheta[1]

    pos_set = model.tvp["pos_set"]
# Custo de verticalidade - se theta = 0 -> custo minimo
    upright_cost = (
        cost_weights["theta1"] * (1.0 - np.cos(theta1))
        + cost_weights["theta2"] * (1.0 - np.cos(theta2))
    )
# Custo de posição
    position_cost = cost_weights["position"] * (pos - pos_set) ** 2
# Custo de velocidade
    velocity_cost = (
        cost_weights["dpos"] * dpos**2
        + cost_weights["dtheta1"] * dtheta1**2
        + cost_weights["dtheta2"] * dtheta2**2
    )
# Custo total
    lterm = upright_cost + position_cost + velocity_cost
    mterm = lterm
    return mterm, lterm

# Penaliza o esforço de controlo.
def apply_rterm(mpc: Any, cost_weights: Dict[str, float]) -> None:
    mpc.set_rterm(force=cost_weights["rterm_force"]) # quanto maior rterm_force, menos agressivo fica o MPC
    
# Aplica limites à força.
def apply_bounds(mpc: Any, constraints: Dict[str, Any]) -> None:
    u_min = constraints.get("u_min", None)
    u_max = constraints.get("u_max", None)

    if u_min is not None:
        mpc.bounds["lower", "_u", "force"] = u_min

    if u_max is not None:
        mpc.bounds["upper", "_u", "force"] = u_max

# Define incerteza nos parâmetros se quiser utilizar nmpc_robust
def apply_uncertainty(mpc: Any) -> None:
    mpc.set_uncertainty_values(
        m1=np.array([0.2]),
        m2=np.array([0.2]),
    )
# define o valor desejado da posição do carrinho ao longo do horizonte.
def build_tvp_fun(mpc: Any, tvp_config: Dict[str, Any]):
    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_ind):
        tvp_template["_tvp", :, "pos_set"] = tvp_config["pos_set"]
        return tvp_template
    return tvp_fun