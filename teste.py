from configs.system.main_config import MAIN_CONFIG
from simulation.runtime_builder import build_runtime

config = MAIN_CONFIG.copy()
config["mode"] = "simulate"
config["controller_type"] = "linear_mpc"
config["environment_type"] = "moldes_fmu"
config["max_steps"] = 300
config["render"] = False

runtime = build_runtime(config)

env = runtime["env"]
controller = runtime["controller"]

obs, _ = env.reset()
controller.reset()

for step in range(300):
    action = controller.compute_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 25 == 0 or step == 299:
        print(
            f"step={step:03d} | "
            f"t={info['time']:.0f}s | "
            f"T1={info['T1_celsius']:.2f} ºC | "
            f"erro={info['tracking_error']:.2f} K | "
            f"mAgua={info['mAgua']:.2f} kg/s"
        )

    if terminated or truncated:
        break

env.close()