from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from environments.base_environment import BaseEnvironment

class GymEnvironment(BaseEnvironment):
    """ Wrapper genérico para ambientes Gymnasium.Adapta a interface Gymnasium para a interface BaseEnvironment. """
    def __init__(
        self,
        name: str,
        env_id: str,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """ Inicializa o ambiente Gymnasium. """
        super().__init__(name=name, config=config)

        self.env_id = env_id
        self.render_mode = render_mode

        self.env = gym.make(env_id, render_mode=render_mode)

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """ Reinicia o ambiente Gymnasium."""
        observation, info = self.env.reset()

        self.current_observation = observation
        self.current_state = observation  # por agora assumimos igual
        self.current_reference = None
        self.done = False

        return observation, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """ Executa um passo no ambiente Gymnasium. """
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.current_observation = observation
        self.current_state = observation
        self.done = bool(terminated or truncated)

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> None:
        """ Renderiza o ambiente."""
        if self.render_mode is not None:
            self.env.render()

    def close(self) -> None:
        """ Fecha o ambiente. """
        self.env.close()

    def get_info(self) -> Dict[str, Any]:
        """ Informação adicional do ambiente. """
        info = super().get_info()
        info.update(
            {
                "env_id": self.env_id,
                "render_mode": self.render_mode,
            }
        )
        return info