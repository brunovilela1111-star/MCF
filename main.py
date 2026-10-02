from configs.system.main_config import MAIN_CONFIG
from simulation.execution import (run_train_mode,run_simulate_mode,run_evaluate_mode,run_compare_mode,)
from simulation.runtime_builder import build_runtime
from utils.validators import validate_main_choices

def print_main_header(config: dict) -> None:
    print("=== MAIN DA ARQUITETURA ===")
    print(f"Mode: {config['mode']}")
    print(f"Controller type: {config['controller_type']}")
    print(f"Environment: {config['environment_type']}")

def main() -> None:
    config = MAIN_CONFIG

    validate_main_choices(
        mode=config["mode"],
        controller_type=config["controller_type"],
    )

    print_main_header(config)

    if config["mode"] == "train":
        run_train_mode(config)
        return

    if config["mode"] == "compare":
        run_compare_mode(config)
        return

    if config["mode"] == "evaluate":
        run_evaluate_mode(config)
        return

    if config["mode"] == "simulate":
        runtime = build_runtime(config)

        try:
            run_simulate_mode(config, runtime)
        finally:
            runtime["env"].close()

        return
    raise ValueError(f"Modo desconhecido: {config['mode']}")

if __name__ == "__main__":
    main()