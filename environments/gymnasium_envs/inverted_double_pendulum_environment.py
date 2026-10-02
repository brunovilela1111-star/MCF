from __future__ import annotations
from typing import Any, Dict, Optional
from environments.gymnasium_envs.gym_environment import GymEnvironment

class InvertedDoublePendulumEnvironment(GymEnvironment):
    """ Wrapper do ambiente Gymnasium MuJoCo Inverted Double Pendulum. """
    def __init__(
        self,
        name: str = "inverted_double_pendulum",
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name,
            env_id="InvertedDoublePendulum-v5",
            config=config,
            render_mode=render_mode,
        )