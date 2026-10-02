from __future__ import annotations
import shutil
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
from environments.base_environment import BaseEnvironment


class FMUEnvironment(BaseEnvironment, gym.Env):
    """
    Ambiente genérico para modelos FMU.

    Esta classe trata apenas da parte comum a qualquer FMU:
    - carregar o ficheiro .fmu;
    - inicializar a simulação;
    - enviar ação para uma variável de entrada;
    - avançar a simulação;
    - ler variáveis de saída;
    - fechar/libertar a instância FMU.

    A reward, referência e lógica específica devem ser definidas
    nas subclasses, por exemplo MassaAmortFMUEnvironment.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}

        BaseEnvironment.__init__(
            self,
            name=config.get("name", "fmu_environment"),
            config=config,
        )

        self.config = config

        # Caminho para o ficheiro .fmu
        self.fmu_path = config["fmu_path"]

        # Nome da variável de entrada da FMU, por exemplo "u"
        self.input_name = config["input_name"]

        # Lista de variáveis de saída, por exemplo ["x", "v"]
        self.output_names = config["output_names"]

        # Tempo de simulação
        self.dt = float(config.get("dt", 0.01))
        self.start_time = float(config.get("start_time", 0.0))
        self.max_time = float(config.get("max_time", 10.0))
        self.time = self.start_time

        # Limites da ação
        self.u_min = float(config.get("u_min", -10.0))
        self.u_max = float(config.get("u_max", 10.0))

        # Objetos internos da FMU
        self.model_description = None
        self.unzipdir = None
        self.fmu = None

        # Dicionário: nome_variável -> valueReference
        self.variables = {}

        self.input_ref = None
        self.output_refs = None

        # Espaço de ação contínuo: u
        self.action_space = spaces.Box(
            low=np.array([self.u_min], dtype=np.float32),
            high=np.array([self.u_max], dtype=np.float32),
            dtype=np.float32,
        )

        # Espaço de observação genérico.
        # O tamanho é igual ao número de outputs da FMU.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.output_names),),
            dtype=np.float32,
        )

        self.current_state = None
        self.current_observation = None
        self.done = False

    def _load_fmu(self) -> None:
        """
        Carrega e inicializa a FMU.
        """

        self.model_description = read_model_description(self.fmu_path)
        self.unzipdir = extract(self.fmu_path)

        model_identifier = self.model_description.coSimulation.modelIdentifier
        guid = self.model_description.guid

        self.fmu = FMU2Slave(
            guid=guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=model_identifier,
            instanceName=f"{self.name}_instance",
        )

        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        self.variables = {
            variable.name: variable.valueReference
            for variable in self.model_description.modelVariables
        }

        self.input_ref = self.variables[self.input_name]
        self.output_refs = [
            self.variables[name]
            for name in self.output_names
        ]

    def _close_fmu(self) -> None:
        """
        Fecha e limpa a instância da FMU.
        """

        if self.fmu is not None:
            try:
                self.fmu.terminate()
            except Exception:
                pass

            try:
                self.fmu.freeInstance()
            except Exception:
                pass

            self.fmu = None

        if self.unzipdir is not None:
            shutil.rmtree(self.unzipdir, ignore_errors=True)
            self.unzipdir = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Reinicia o ambiente.

        Formato compatível com Gymnasium:
        observation, info = env.reset()
        """

        if seed is not None:
            np.random.seed(seed)

        self._close_fmu()
        self._load_fmu()

        self.time = self.start_time
        self.done = False

        observation = np.asarray(
            self.fmu.getReal(self.output_refs),
            dtype=np.float32,
        )

        self.current_observation = observation
        self.current_state = observation

        info = {
            "time": self.time,
        }

        return observation, info

    def step(self, action: Any):
        """
        Executa um passo da simulação FMU.

        Recebe:
        - action: força/comando aplicado à FMU.

        Devolve:
        - observation
        - reward
        - terminated
        - truncated
        - info
        """

        if self.fmu is None:
            raise RuntimeError("A FMU ainda não foi inicializada. Chama reset() primeiro.")

        action_array = np.asarray(action, dtype=float).reshape(-1)
        u = float(action_array[0])
        u = float(np.clip(u, self.u_min, self.u_max))

        self.fmu.setReal([self.input_ref], [u])

        self.fmu.doStep(
            currentCommunicationPoint=self.time,
            communicationStepSize=self.dt,
        )

        self.time += self.dt

        observation = np.asarray(
            self.fmu.getReal(self.output_refs),
            dtype=np.float32,
        )
        
        reward = self.compute_reward(observation=observation, action=u)

        terminated = self.check_terminated(observation=observation)
        truncated = bool(self.time >= self.max_time)

        self.done = bool(terminated or truncated)
        self.current_observation = observation
        self.current_state = observation

        info = self.build_info(
            observation=observation,
            action=u,
            terminated=terminated,
            truncated=truncated,
        )

        return observation, reward, terminated, truncated, info

    def compute_reward(self, observation: np.ndarray, action: float) -> float:
        """Reward genérica.Deve ser sobrescrita nas subclasses. """
        return 0.0

    def check_terminated(self, observation: np.ndarray) -> bool:
        """  Condição de término. Por defeito, não termina antes de max_time. """
        return False

    def build_info(
        self,
        observation: np.ndarray,
        action: float,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        """
        Informação auxiliar do passo.
        """
        return {
            "time": self.time,
            "action": action,
            "terminated": terminated,
            "truncated": truncated,
        }

    def get_state(self):
        return self.current_state

    def get_observation(self):
        return self.current_observation

    def render(self):
        return None

    def close(self) -> None:
        """
        Fecha corretamente a FMU e a janela gráfica.
        """

        if self.live_plot is not None:
            self.live_plot.close()

        super().close()