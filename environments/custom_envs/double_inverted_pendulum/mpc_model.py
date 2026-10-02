# modelo físico/matemático usado pelo NMPC

from __future__ import annotations
import do_mpc
from casadi import cos, sin, vertcat

def build_double_pendulum_model(m0: float,m1_value: float,m2_value: float,g: float,L1: float,L2: float,):
    model = do_mpc.model.Model("continuous") #modleo usado para NMPC
    # Centro de massa de cada pêndulo.
    l1 = L1 / 2.0
    l2 = L2 / 2.0
    # Momentos de Inércia
    J1 = (m1_value * l1**2) / 3.0
    J2 = (m2_value * l2**2) / 3.0
    # massas dos pendulos
    m1 = model.set_variable("_p", "m1")
    m2 = model.set_variable("_p", "m2")
    # constantes auxiliares para facilitar equações dinâmicas
    h1 = m0 + m1 + m2
    h2 = m1 * l1 + m2 * L1
    h3 = m2 * l2
    h4 = m1 * l1**2 + m2 * L1**2 + J1
    h5 = m2 * l2 * L1
    h6 = m2 * l2**2 + J2
    h7 = (m1 * l1 + m2 * L1) * g
    h8 = m2 * l2 * g
    # Referencia de posição de carrinho
    pos_set = model.set_variable("_tvp", "pos_set")
    # Estados
    pos = model.set_variable("_x", "pos")
    theta = model.set_variable("_x", "theta", (2, 1))
    dpos = model.set_variable("_x", "dpos")
    dtheta = model.set_variable("_x", "dtheta", (2, 1))
    ddpos = model.set_variable("_z", "ddpos")
    ddtheta = model.set_variable("_z", "ddtheta", (2, 1))
    # Força de controlo
    force = model.set_variable("_u", "force")
    # Equações diferenciais
    model.set_rhs("pos", dpos)
    model.set_rhs("theta", dtheta)
    model.set_rhs("dpos", ddpos)
    model.set_rhs("dtheta", ddtheta)
    # física do pêndulo duplo através das equações de euler
    euler_lagrange = vertcat(
        h1 * ddpos
        + h2 * ddtheta[0] * cos(theta[0])
        + h3 * ddtheta[1] * cos(theta[1])
        - (
            h2 * dtheta[0] ** 2 * sin(theta[0])
            + h3 * dtheta[1] ** 2 * sin(theta[1])
            + force
        ),
        h2 * cos(theta[0]) * ddpos
        + h4 * ddtheta[0]
        + h5 * cos(theta[0] - theta[1]) * ddtheta[1]
        - (
            h7 * sin(theta[0])
            - h5 * dtheta[1] ** 2 * sin(theta[0] - theta[1])
        ),
        h3 * cos(theta[1]) * ddpos
        + h5 * cos(theta[0] - theta[1]) * ddtheta[0]
        + h6 * ddtheta[1]
        - (
            h5 * dtheta[0] ** 2 * sin(theta[0] - theta[1])
            + h8 * sin(theta[1])
        ),
    )

    model.set_alg("euler_lagrange", euler_lagrange)
    # Energia cinética
    E_kin_cart = 0.5 * m0 * dpos**2
    E_kin_p1 = (
        0.5
        * m1
        * (
            (dpos + l1 * dtheta[0] * cos(theta[0])) ** 2
            + (l1 * dtheta[0] * sin(theta[0])) ** 2
        )
        + 0.5 * J1 * dtheta[0] ** 2
    )
    E_kin_p2 = (
        0.5
        * m2
        * (
            (
                dpos
                + L1 * dtheta[0] * cos(theta[0])
                + l2 * dtheta[1] * cos(theta[1])
            )
            ** 2
            + (
                L1 * dtheta[0] * sin(theta[0])
                + l2 * dtheta[1] * sin(theta[1])
            )
            ** 2
        )
        + 0.5 * J2 * dtheta[0] ** 2
    )
    E_kin = E_kin_cart + E_kin_p1 + E_kin_p2
    # Energia potencial
    E_pot = m1 * g * l1 * cos(theta[0]) + m2 * g * (
        L1 * cos(theta[0]) + l2 * cos(theta[1])
    )
    model.set_expression("E_kin", E_kin)
    model.set_expression("E_pot", E_pot)
    model.set_expression("tvp", pos_set)
    model.setup()
    return model

def build_double_pendulum_simulator(
    model,
    dt: float,
    integration_tool: str,
    abstol: float,
    reltol: float,
    m1_value: float,
    m2_value: float,
):
    simulator = do_mpc.simulator.Simulator(model)
# criar simulador do modelo
    simulator.set_param(
        integration_tool=integration_tool,
        abstol=abstol,
        reltol=reltol,
        t_step=dt,
    )
    p_num = simulator.get_p_template()
    p_num["m1"] = m1_value
    p_num["m2"] = m2_value
# Fornece os parâmetros ao simulador.
    def p_fun(t_now):
        return p_num
    simulator.set_p_fun(p_fun)
    tvp_template = simulator.get_tvp_template()
# Fornece os parâmetros variantes no tempo.
    def tvp_fun(t_ind):
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)
    simulator.setup()
    return simulator