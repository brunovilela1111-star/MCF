MPC_MODEL_REGISTRY = {
    ("nonlinear_mpc", "double_inverted_pendulum"): {
        "module_path": "environments.custom_envs.double_inverted_pendulum.models.nonlinear_model",
        "callable_name": "get_nonlinear_double_inverted_pendulum_config",
    },
    ("nonlinear_mpc_optimized", "double_inverted_pendulum"): {
        "module_path": "environments.custom_envs.double_inverted_pendulum.models.nonlinear_model_optimized",
        "callable_name": "get_nonlinear_double_inverted_pendulum_optimized_config",
    },
    ("nonlinear_mpc_robust", "double_inverted_pendulum"): {
        "module_path": "environments.custom_envs.double_inverted_pendulum.models.nonlinear_model_robust",
        "callable_name": "get_nonlinear_double_inverted_pendulum_robust_config",
    },
    ("nonlinear_mpc_adaptive", "double_inverted_pendulum"): {
        "module_path": "environments.custom_envs.double_inverted_pendulum.models.nonlinear_model_adaptive",
        "callable_name": "get_nonlinear_double_inverted_pendulum_adaptive_config",
    },
    ("linear_mpc", "massa_amort_fmu"): {
        "module_path": ("environments.fmu_envs.massa_amort_env.models.linear_model"),
        "callable_name": "get_linear_model",
    },
    ("linear_mpc", "sistema_termico_fmu"): {
        "module_path": "environments.fmu_envs.sistema_termico_env.models.linear_model",
        "callable_name": "get_linear_model",
    },
    ("linear_mpc", "moldes_fmu"): {
        "module_path": "environments.fmu_envs.moldes_env.models.linear_model",
        "callable_name": "get_linear_model",
    },
    ("nonlinear_mpc", "moldes_fmu"): {
        "module_path": "environments.fmu_envs.moldes_env.models.nonlinear_model",
        "callable_name": "get_nonlinear_model",
    },
}