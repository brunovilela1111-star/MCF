from pathlib import Path

# CONFIGURAÇÃO DO AMBIENTE FMU - MOLDES DE INJEÇÃO
CURRENT_DIR = Path(__file__).resolve().parent
MOLDES_CONFIG = {
    # Identificação
    "name": "moldes_fmu",
    "fmu_path": str(CURRENT_DIR / "fmus" / "Moldes.fmu"),

    # Variáveis da FMU
    "input_name": "mAgua",
    "output_names": ["T1_out", "s_out"],

    # Tempo de simulação
    "dt": 1.0,
    "start_time": 0.0,
    
    # Ciclo dos moldes
    # Deve coincidir com o Pulse definido no OpenModelica.
    "cycle_period": 50.0,
    "cycle_open_time": 10.0,
    # s(t) = 0  -> moldes abertos
    # s(t) = 1  -> moldes fechados
    #0-10 s      -> aberto
    # 10-50 s     -> fechado
    # 50-60 s     -> aberto
    # 60-100 s    -> fechado
    # ...
    
    # Limites do caudal de água [kg/s]
    "u_min": 0.0,
    "u_max": 75.0,

    # Referência de controlo
    "T1_ref": 303.15,       # 30 ºC 303.15
    "T1_initial": 296.15,   # 23 ºC 296.15

    # Reward / custo
    "temperature_weight": 1.0,
    "action_weight": 0.001,
    "temperature_tolerance": 2.0,
    "terminate_on_target": False,

    # Parâmetros físicos
    "m1": 7000.0, # Kg
    "c1": 460.0, # J

    "Tinj": 513.15,    #240 ºC 513.15
    "Tpl": 513.15,       #240 ºC 513.15
    "Tamb": 296.15,     #23 ºC 296.15
    "Tagua":293.15,    # 20 ºC 293.15
    "T2":308.15,       # 35 ºC 308.15

    "kinj": 5000.0, # W
    "kpl": 8000.0, # W
    "k21": 20000.0, # W
    "kamb": 300.0, # W
    "cAgua": 4180.0, # J 

    # Parâmetros lineares MPC
    "linear_Q": 100.0,
    "linear_R": 0.001,
    "linear_prediction_horizon": 20,
    "linear_control_horizon": 5,
}