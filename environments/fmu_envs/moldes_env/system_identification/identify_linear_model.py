from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Permite correr este ficheiro diretamente a partir da raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from factories.environment_factory import create_environment


def collect_data():
    """
    Recolhe dados da FMU para identificar um modelo linear aumentado.

    Modelo identificado:

        T1(k+1) = A*T1(k) + B*u(k) + C*s(k) + d

    onde:
        T1 -> temperatura do Molde 1 [ºC]
        u  -> caudal de água [kg/s]
        s  -> estado dos moldes
              0 = aberto
              1 = fechado
        d  -> termo constante
    """

    env = create_environment(
        "moldes_fmu",
        overrides={
            "output_names": ["T1_out", "s_out"],
            "max_time": 300.0,
        },
    )

    X = []
    Y = []

    rng = np.random.default_rng(seed=42)

    obs, _ = env.reset()

    for _ in range(3000):
        T1_k = float(obs[0])
        s_k = float(obs[1])

        block_size = 60

        flow_levels = [
            0.0,
            5.0,
            10.0,
            15.0,
            20.0,
        ]

        action = flow_levels[
            (_ // block_size) % len(flow_levels)
        ]
        next_obs, _, terminated, truncated, _ = env.step(action)

        T1_k1 = float(next_obs[0])

        X.append([
            T1_k,
            action,
            s_k,
            1.0,
        ])

        Y.append(T1_k1)

        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
    X_array = np.array(X)
    Y_array = np.array(Y)
    s_values = X_array[:, 2]
    print(f"s=0 count: {np.sum(s_values == 0)}")
    print(f"s=1 count: {np.sum(s_values == 1)}")
    env.close()
    return X_array, Y_array

def identify_model():
    """
    Identifica os coeficientes do modelo linear aumentado.

    Resolve por mínimos quadrados:

        Y = X * theta

    com:

        theta = [A, B, C, d]
    """

    X, Y = collect_data()

    theta, residuals, _, _ = np.linalg.lstsq(
        X,
        Y,
        rcond=None,
    )

    A = float(theta[0])
    B = float(theta[1])
    C = float(theta[2])
    d = float(theta[3])

    print("\n=== MODELO LINEAR IDENTIFICADO A PARTIR DA FMU ===")
    print(f"A = {A:.8f}")
    print(f"B = {B:.8f}")
    print(f"C = {C:.8f}")
    print(f"d = {d:.8f}")

    print("\nModelo:")
    print(
        f"T1(k+1) = {A:.6f}*T1(k) "
        f"+ {B:.6f}*u(k) "
        f"+ {C:.6f}*s(k) "
        f"+ {d:.6f}"
    )

    if residuals.size > 0:
        rmse = float(np.sqrt(residuals[0] / len(Y)))
        print(f"\nRMSE = {rmse:.6f}")
    # Diretório models do ambiente.
    models_dir = (
        Path(__file__).resolve().parents[1]
        / "models"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_file = models_dir / "identified_linear_model.npz"

    np.savez(
        model_file,
        A=A,
        B=B,
        C=C,
        d=d,
    )

    print(
        f"\nModelo guardado em:\n{model_file}"
    )
    return A, B, C, d


if __name__ == "__main__":
    identify_model()