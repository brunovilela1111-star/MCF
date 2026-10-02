ENVIRONMENT_REGISTRY = {
# gym env
    "cartpole": {
        "class_name": "CartPoleEnvironment",
        "module_path": "environments.gymnasium_envs.cartpole_environment",
        "config_module": None,
        "config_name": None,
        "action_space_type": "discrete",
        "backend": "gymnasium",
        "supports_training": True,
        "supports_render_mode": True,
    },
    "inverted_pendulum": {
        "class_name": "InvertedPendulumEnvironment",
        "module_path": "environments.gymnasium_envs.inverted_pendulum_environment",
        "config_module": None,
        "config_name": None,
        "action_space_type": "box",
        "backend": "gymnasium",
        "supports_training": True,
        "supports_render_mode": True,
    },
    "inverted_double_pendulum": {
        "class_name": "InvertedDoublePendulumEnvironment",
        "module_path": "environments.gymnasium_envs.inverted_double_pendulum_environment",
        "config_module": None,
        "config_name": None,
        "action_space_type": "box",
        "backend": "gymnasium",
        "supports_training": True,
        "supports_render_mode": True,
    },
# custom env
    "double_inverted_pendulum": {
        "class_name": "DoubleInvertedPendulumEnvironment",
        "module_path": "environments.custom_envs.double_inverted_pendulum.environment",
        "config_module": "environments.custom_envs.double_inverted_pendulum.config",
        "config_name": "DOUBLE_INVERTED_PENDULUM_CONFIG",
        "action_space_type": "box",
        "backend": "custom",
        "supports_training": True,
        "supports_render_mode": False,
        "metrics": {
            "module_path": "evaluation.environment_metrics.double_inverted_pendulum_metrics",
            "callable_name": "compute_double_inverted_pendulum_metrics",
        },
    },
#fmu env
   
    "moldes_fmu": {
        "class_name": "MoldesFMUEnvironment",
        "module_path": "environments.fmu_envs.moldes_env.moldes_fmu_environment",
        "config_module": "environments.fmu_envs.moldes_env.moldes_config",
        "config_name": "MOLDES_CONFIG",
        "backend": "fmu",
        "supports_training": True,
        "supports_render_mode": True,
        "metrics": {
            "module_path": "evaluation.environment_metrics.moldes_fmu_metrics",
            "callable_name": "compute_moldes_metrics",
        },
    },
}