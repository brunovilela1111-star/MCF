from __future__ import annotations
from typing import Any, Dict, Optional
from stable_baselines3 import A2C
from controllers.rl.base_rl_controller import BaseRLController

class A2CController(BaseRLController):
    def __init__(
        self,
        name: str,
        algorithm_name: str,
        env: Any,
        config: Optional[Dict[str, Any]] = None,
        policy: str = "MlpPolicy",
        learning_rate: float = 7e-4,
        n_steps: int = 5,
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
        self.n_steps = n_steps
        self.gamma = gamma
        self.verbose = verbose
        self.model = model

    def initialize(self) -> None:
        if self.model is None:
            self.model = A2C(
                policy=self.policy_name,
                env=self.env.env if hasattr(self.env, "env") else self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                gamma=self.gamma,
                device=self.device,
                verbose=self.verbose,
            )

        self.policy = self.model.policy
        self.is_initialized = True
        print(f"[{self.name}] Controlador A2C inicializado.")

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        if self.model is None:
            raise RuntimeError(f"[{self.name}] O modelo A2C ainda não foi inicializado.")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def train_step(self, total_timesteps: int, *args, **kwargs) -> None:
        if self.model is None:
            raise RuntimeError(f"[{self.name}] O modelo A2C ainda não foi inicializado.")

        self.model.learn(total_timesteps=total_timesteps, *args, **kwargs)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError(f"[{self.name}] Não existe modelo A2C para guardar.")

        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = A2C.load(path)
        self.policy = self.model.policy
        self.is_initialized = True

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "policy_name": self.policy_name,
                "learning_rate": self.learning_rate,
                "n_steps": self.n_steps,
                "gamma": self.gamma,
                "has_model": self.model is not None,
            }
        )
        return info