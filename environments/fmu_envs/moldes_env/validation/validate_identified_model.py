from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Permite correr o ficheiro diretamente a partir da raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from factories.environment_factory import create_environment
from environments.fmu_envs.moldes_env.models.linear_model import get_linear_model


def choose_water_flow(k: int) -> float:
    block_size = 60
    flow_levels = [0.0, 5.0, 10.0, 15.0, 20.0]
    return flow_levels[(k // block_size) % len(flow_levels)]


def validate_identified_model():
    env = create_environment(
        "moldes_fmu",
        overrides={
            "output_names": ["T1_out", "s_out"],
            "max_time": 300.0,
            "render_mode": None,
        },
    )

    model = get_linear_model()

    A = float(model["A"][0, 0])
    B = float(model["B"][0, 0])
    C = float(model["C"][0, 0])
    d = float(model["d"][0])
    dt = float(model["dt"])

    obs, _ = env.reset()
    T1_linear = float(obs[0])

    times = []
    T1_fmu_values = []
    T1_linear_values = []
    water_flows = []

    n_steps = 300

    for k in range(n_steps):
        time = k * dt

        T1_fmu = float(obs[0])
        s = float(obs[1])
        u = choose_water_flow(k)

        times.append(time)
        T1_fmu_values.append(T1_fmu - 273.15)
        T1_linear_values.append(T1_linear - 273.15)
        water_flows.append(u)

        next_obs, _, terminated, truncated, _ = env.step(u)

        T1_linear = A * T1_linear + B * u + C * s + d

        obs = next_obs

        if terminated or truncated:
            break

    env.close()

    T1_fmu_array = np.array(T1_fmu_values)
    T1_linear_array = np.array(T1_linear_values)

    error = T1_fmu_array - T1_linear_array
    rmse = float(np.sqrt(np.mean(error**2)))

    print("\n=== VALIDAÇÃO DO MODELO IDENTIFICADO ===")
    print(f"RMSE = {rmse:.4f} ºC")

    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "fmu_vs_linear_model.png"

    plt.figure(figsize=(10, 5))

    plt.plot(
        times,
        T1_fmu_values,
        label="FMU Model",
        linewidth=2,
    )

    plt.plot(
        times,
        T1_linear_values,
        linestyle="--",
        label="Identified Linear Model",
        linewidth=2,
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Temperature of Mold 1 [°C]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nFigure saved in:\n{output_path}")


if __name__ == "__main__":
    validate_identified_model()