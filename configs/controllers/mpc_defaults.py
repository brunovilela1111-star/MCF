MPC_BASE = {
    "constraints": {},
    "reference": None,
}

MPC_DEFAULTS = {
    "linear_mpc": {
        **MPC_BASE,
        "name": "linear_mpc_controller",
        "mpc_type": "LinearMPC",
        "prediction_horizon": 5,
        "control_horizon": 5,
    },

    "nonlinear_mpc": {
        **MPC_BASE,
        "name": "nonlinear_mpc_controller",
        "mpc_type": "NonLinearMPC",
        "prediction_horizon": 20,
        "control_horizon": 20,
    },
}