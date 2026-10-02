from __future__ import annotations

MAIN_CONFIG = {
    # mode: "train", "simulate", "evaluate", "compare"
    "mode": "simulate",
    # controller
    "controller_type": "sac",
    # environment
    "environment_type": "moldes_fmu",
    # execution
    "render": True, 
    "max_steps": 300,
    "num_episodes": 1,
    # training (only to RL)
    "total_timesteps": 500_000,
    "use_trained_model": True,
    "trained_model_name":"SAC_500K_FMU",
    # optional overrides to pontual alterations
    "env_overrides": {},
    "controller_overrides": {},
    # comparison
    "comparison_controllers": [],
    "comparison_use_trained_models": {}, 
    "comparison_trained_model_names": {},
    "comparison_controller_overrides": {}, 
    }