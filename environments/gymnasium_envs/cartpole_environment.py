from __future__ import annotations
from typing import Any, Dict, Optional
from environments.gymnasium_envs.gym_environment import GymEnvironment

class CartPoleEnvironment(GymEnvironment):
    """ Ambiente específico CartPole. """
    def __init__(
        self,
        name: str = "cartpole",
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name,
            env_id="CartPole-v1",#importe de ambiente gym
            config=config,
            render_mode=render_mode,
        )