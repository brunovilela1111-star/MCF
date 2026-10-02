from __future__ import annotations
from typing import Any, Dict, Optional
from environments.gymnasium_envs.gym_environment import GymEnvironment

class InvertedPendulumEnvironment(GymEnvironment):
    """ Ambiente específico Inverted Pendulum (Gymnasium MuJoCo). """
    def __init__(
        self,
        name: str = "inverted_pendulum",
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name,
            env_id="InvertedPendulum-v5",
            config=config,
            render_mode=render_mode,
        )