from pathlib import Path
import numpy as np
from environments.fmu_envs.moldes_env.moldes_config import MOLDES_CONFIG

def get_linear_model():
    """
    Carrega automaticamente o modelo linear/afim identificado
    a partir da FMU.

    O ficheiro identified_linear_model.npz é criado pelo módulo:
        system_identification/identify_linear_model.py

    Modelo identificado:
        T1(k+1) = A*T1(k) + B*u(k) + C*s(k) + d
    """

    config = MOLDES_CONFIG

    model_file = (
        Path(__file__).resolve().parent
        / "identified_linear_model.npz"
    )

    # Segurança caso o utilizador ainda não tenha corrido a identificação.
    if not model_file.exists():
        raise FileNotFoundError(
            f"Modelo identificado não encontrado:\n{model_file}\n\n"
            "Execute primeiro o ficheiro:\n"
            "environments/fmu_envs/moldes_env/system_identification/"
            "identify_linear_model.py"
        )

    # Carrega os coeficientes identificados a partir da FMU.
    data = np.load(model_file)

    A_identified = float(data["A"])
    B_identified = float(data["B"])
    C_identified = float(data["C"])
    d_identified = float(data["d"])

    print("\n=== MODELO IDENTIFICADO CARREGADO ===")
    print(f"A = {A_identified:.8f}")
    print(f"B = {B_identified:.8f}")
    print(f"C = {C_identified:.8f}")
    print(f"d = {d_identified:.8f}")

    # Matrizes do modelo usado pelo Linear MPC.
    A = np.array([[A_identified]])
    B = np.array([[B_identified]])
    C = np.array([[C_identified]])
    d = np.array([d_identified])

    # Pesos do custo MPC.
    Q = np.array([[float(config["linear_Q"])]])
    R = np.array([[float(config["linear_R"])]])

    # Referência de temperatura do Molde 1.
    reference = np.array([float(config["T1_ref"])])

    # Limites físicos do caudal de água.
    constraints = {
        "u_min": float(config["u_min"]),
        "u_max": float(config["u_max"]),
    }
 
    return {
        "A": A,
        "B": B,
        "C": C,
        "d": d,
        "Q": Q,
        "R": R,
        "reference": reference,
        "constraints": constraints,
        "prediction_horizon": int(config["linear_prediction_horizon"]),
        "control_horizon": int(config["linear_control_horizon"]),
        "dt": float(config["dt"]),
        "cycle_period": float(config["cycle_period"]),
        "cycle_open_time": float(config["cycle_open_time"]),
    }