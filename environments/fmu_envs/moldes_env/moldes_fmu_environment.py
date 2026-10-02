from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from environments.fmu_envs.fmu_environment import FMUEnvironment
from environments.fmu_envs.moldes_env.moldes_config import MOLDES_CONFIG
from environments.fmu_envs.moldes_env.graphics.moldes_live_plot import MoldesLivePlot
 
class MoldesFMUEnvironment(FMUEnvironment):
    """
    Ambiente FMU para o sistema térmico simplificado dos moldes de injeção.

    A FMU representa a planta real.
    O controlador atua apenas no caudal de água mAgua.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Inicializa o ambiente dos moldes.

        A configuração base vem de MOLDES_CONFIG.
        Os overrides podem substituir valores específicos.
        """

        merged_config = MOLDES_CONFIG.copy()

        if config is not None:
            merged_config.update(config)

        super().__init__(config=merged_config)

        self.T1_ref = float(self.config["T1_ref"])
        self.T1_initial = float(self.config["T1_initial"])

        self.temperature_weight = float(self.config["temperature_weight"])
        self.action_weight = float(self.config["action_weight"])
        self.temperature_tolerance = float(self.config["temperature_tolerance"])
        self.terminate_on_target = bool(self.config["terminate_on_target"])

        self.reference = np.array([self.T1_ref], dtype=np.float32)
        # Visualização em tempo real.
        self.live_plot = None

        if self.config.get("render_mode") == "human":
            self.live_plot = MoldesLivePlot()

    def compute_reward(self, observation: np.ndarray, action: float) -> float:
        """
        Calcula a reward.

        Reward = negativo do custo.

        O custo penaliza:
        - erro de temperatura;
        - esforço de controlo.
        """

        T1 = float(observation[0])
        temperature_error = T1 - self.T1_ref

        cost = (
            self.temperature_weight * temperature_error**2
            + self.action_weight * action**2
        )

        return -float(cost)

    def check_terminated(self, observation: np.ndarray) -> bool:
        """
        Define se o episódio termina antes do max_time.

        Por defeito, não termina ao atingir a referência, porque interessa
        avaliar a evolução completa do sistema.
        """

        if not self.terminate_on_target:
            return False

        T1 = float(observation[0])
        temperature_error = abs(T1 - self.T1_ref)

        return bool(temperature_error <= self.temperature_tolerance)

    def build_info(
        self,
        observation: np.ndarray,
        action: float,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        """
        Informação adicional usada por métricas, debugging e análise.
        """
        T1 = float(observation[0])
        temperature_error = T1 - self.T1_ref

        if self.live_plot is not None:
            mold_state = 0.0
            try:
                if len(observation) > 1:
                    mold_state = float(observation[1])
            except Exception:
                pass

            self.live_plot.update(
                time=self.time,
                temperature_celsius=T1 - 273.15,
                reference_celsius=self.T1_ref - 273.15,
                water_flow=float(action),
                mold_state=mold_state,
            )
            print(
                f"time={self.time:.0f} "
                f"T1={T1:.2f} "
                f"mAgua={action:.2f}"
                )

        return {
            "time": self.time,
            "mAgua": action,

            "T1": T1,
            "T1_celsius": T1- 273.15,

            "T1_ref": self.T1_ref,
            "T1_ref_celsius": self.T1_ref- 273.15,

            "temperature_error": temperature_error,
            "temperature_error_abs": abs(temperature_error),

            "tracking_error": abs(temperature_error),
            "reference": self.reference.copy(),

            "terminated": terminated,
            "truncated": truncated,
        }

    def get_reference(self) -> np.ndarray:
        """
        Devolve a referência usada por controladores e métricas.
        """

        return self.reference.copy()