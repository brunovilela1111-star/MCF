RL_BASE = {
    "family": "rl",
    "supported_action_spaces": ["box", "discrete"],
    "supports_training": True,
    "requires_training": True,
    "requires_env": True,
}
CONTINUOUS_RL_BASE = {
    **RL_BASE,
    "supported_action_spaces": ["box"],
}
MPC_BASE = {
    "family": "mpc",
    "supported_action_spaces": ["box"],
    "supports_training": False,
    "requires_training": False,
}
NONLINEAR_MPC_BASE = {
    **MPC_BASE,
    "class_name": "NonlinearMPCController",
    "module_path": "controllers.mpc.nonlinear_mpc_controller",
    "defaults_key": "nonlinear_mpc",
    "requires_env": True,
}
# ========================================================================================= #
CONTROLLER_REGISTRY = {
    "ppo": {
        **RL_BASE,
        "class_name": "PPOController",
        "module_path": "controllers.rl.ppo_controller",
        "defaults_key": "ppo",
    },

    "a2c": {
        **RL_BASE,
        "class_name": "A2CController",
        "module_path": "controllers.rl.a2c_controller",
        "defaults_key": "a2c",
    },

    "td3": {
        **CONTINUOUS_RL_BASE,
        "class_name": "TD3Controller",
        "module_path": "controllers.rl.td3_controller",
        "defaults_key": "td3",
    },

    "dqn": {
        **RL_BASE,
        "class_name": "DQNController",
        "module_path": "controllers.rl.dqn_controller",
        "defaults_key": "dqn",
        "supported_action_spaces": ["discrete"],
    },

    "ddpg": {
        **CONTINUOUS_RL_BASE,
        "class_name": "DDPGController",
        "module_path": "controllers.rl.ddpg_controller",
        "defaults_key": "ddpg",
    },

    "sac": {
        **CONTINUOUS_RL_BASE,
        "class_name": "SACController",
        "module_path": "controllers.rl.sac_controller",
        "defaults_key": "sac",
    },

    "linear_mpc": {
        **MPC_BASE,
        "class_name": "LinearMPCController",
        "module_path": "controllers.mpc.linear_mpc_controller",
        "defaults_key": "linear_mpc",
        "requires_env": False,
    },

    "nonlinear_mpc": {
        **NONLINEAR_MPC_BASE,
    },

    "nonlinear_mpc_optimized": {
        **NONLINEAR_MPC_BASE,
    },

    "nonlinear_mpc_robust": {
        **NONLINEAR_MPC_BASE,
    },

    "nonlinear_mpc_adaptive": {
        **NONLINEAR_MPC_BASE,
    },
}