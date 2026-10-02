from __future__ import annotations
from typing import Any, Dict, Optional
from stable_baselines3 import DDPG
from controllers.rl.base_rl_controller import BaseRLController

class DDPGController(BaseRLController):
    def __init__(
        self,
        name: str,
        algorithm_name: str,
        env: Any,
        config: Optional[Dict[str, Any]] = None,
        policy: str = "MlpPolicy",
        learning_rate: float = 1e-3,
        buffer_size: int = 100000,
        batch_size: int = 256,
        gamma: float = 0.99,
        device: str = "cpu",
        verbose: int = 1,
        model: Optional[Any] = None,
    ) -> None:
        super().__init__(
            name=name,
            algorithm_name=algorithm_name,
            config=config,
            policy=None,
            device=device,
        )

        self.env = env
        self.policy_name = policy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose
        self.model = model

    def initialize(self) -> None:
        if self.model is None:
            self.model = DDPG(
                policy=self.policy_name,
                env=self.env.env if hasattr(self.env, "env") else self.env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                device=self.device,
                verbose=self.verbose,
            )

        self.policy = self.model.policy
        self.is_initialized = True
        print(f"[{self.name}] Controlador DDPG inicializado.")

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        if self.model is None:
            raise RuntimeError(f"[{self.name}] O modelo DDPG ainda não foi inicializado.")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def train_step(self, total_timesteps: int, *args, **kwargs) -> None:
        if self.model is None:
            raise RuntimeError(f"[{self.name}] O modelo DDPG ainda não foi inicializado.")

        self.model.learn(total_timesteps=total_timesteps, *args, **kwargs)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError(f"[{self.name}] Não existe modelo DDPG para guardar.")

        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = DDPG.load(path)
        self.policy = self.model.policy
        self.is_initialized = True

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "policy_name": self.policy_name,
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "has_model": self.model is not None,
            }
        )
        return info