#configuração + dinâmica + reward + renderização + modelo MPC + interface Gym

from __future__ import annotations
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from environments.base_environment import BaseEnvironment
from environments.custom_envs.double_inverted_pendulum.config import (DOUBLE_INVERTED_PENDULUM_CONFIG,)
from environments.custom_envs.double_inverted_pendulum.dynamics import (build_initial_double_pendulum_state, get_double_pendulum_observation,process_double_pendulum_action,simulate_double_pendulum_step,)
from environments.custom_envs.double_inverted_pendulum.graphics import (close_double_pendulum_renderer,initialize_double_pendulum_renderer,render_double_pendulum,)
from environments.custom_envs.double_inverted_pendulum.mpc_model import (build_double_pendulum_model,build_double_pendulum_simulator,)
from environments.custom_envs.double_inverted_pendulum.reward import (build_double_pendulum_info,check_double_pendulum_termination,compute_double_pendulum_reward,)

class DoubleInvertedPendulumEnvironment(BaseEnvironment, gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        name: str = "double_inverted_pendulum",
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        BaseEnvironment.__init__(self, name=name, config=config)
        gym.Env.__init__(self)

        cfg = {
            **DOUBLE_INVERTED_PENDULUM_CONFIG,
            **(config or {}),
            **kwargs,
        }

        if render_mode is not None:
            cfg["render_mode"] = render_mode

        self.render_mode = cfg.get("render_mode", render_mode)
        self.config = cfg

        self._load_core_parameters(cfg)
        self._load_physical_parameters(cfg)
        self._load_simulator_parameters(cfg)
        self._load_reward_parameters(cfg)
        self._init_render_objects()

        self.model = build_double_pendulum_model(
            m0=self.m0,
            m1_value=self.m1_value,
            m2_value=self.m2_value,
            g=self.g,
            L1=self.L1,
            L2=self.L2,
        )

        self.simulator = build_double_pendulum_simulator(
            model=self.model,
            dt=self.dt,
            integration_tool=self.simulator_integration_tool,
            abstol=self.simulator_abstol,
            reltol=self.simulator_reltol,
            m1_value=self.m1_value,
            m2_value=self.m2_value,
        )

        self._build_spaces()
        self._reset_internal_state()

    def _load_core_parameters(self, params: Dict[str, Any]) -> None: 
        self.max_steps = int(params.get("max_steps", 500))
        self.elapsed_steps = 0
        self.initial_theta_factor = float(params.get("initial_theta_factor", 0.9))
        self.action_limit = float(params.get("action_limit", 4.0))

    def _load_physical_parameters(self, params: Dict[str, Any]) -> None:
        self.dt = float(params.get("dt", 0.04))
        self.m0 = float(params.get("m0", 0.6))
        self.m1_value = float(params.get("m1", 0.2))
        self.m2_value = float(params.get("m2", 0.2))
        self.g = float(params.get("g", 9.80665))
        self.L1 = float(params.get("L1", 0.5))
        self.L2 = float(params.get("L2", 0.5))

    def _load_simulator_parameters(self, params: Dict[str, Any]) -> None:
        self.simulator_integration_tool = params.get(
            "simulator_integration_tool",
            "idas",
        )
        self.simulator_abstol = float(params.get("simulator_abstol", 1e-8))
        self.simulator_reltol = float(params.get("simulator_reltol", 1e-8))

    def _load_reward_parameters(self, params: Dict[str, Any]) -> None:
        self.reward_config = {
            "position_weight": float(params.get("position_weight", 1.0)),
            "angle_weight": float(params.get("angle_weight", 0.5)),
            "velocity_weight": float(params.get("velocity_weight", 0.1)),
            "action_weight": float(params.get("action_weight", 0.01)),
            "position_threshold": float(params.get("position_threshold", 2.0)),
            "angle_threshold": float(params.get("angle_threshold", 2.0 * np.pi)),
            "near_upright_angle_threshold": float(
                params.get("near_upright_angle_threshold", 0.25)
            ),
            "near_upright_bonus": float(params.get("near_upright_bonus", 5.0)),
            "stabilized_angle_threshold": float(
                params.get("stabilized_angle_threshold", 0.15)
            ),
            "stabilized_velocity_threshold": float(
                params.get("stabilized_velocity_threshold", 0.5)
            ),
            "stabilized_bonus": float(params.get("stabilized_bonus", 15.0)),
            "termination_penalty": float(params.get("termination_penalty", 50.0)),
        }

        self.termination_penalty = self.reward_config["termination_penalty"]

    def _init_render_objects(self) -> None:
        self.fig = None
        self.ax = None
        self.bar1 = None
        self.bar2 = None

    def _build_spaces(self) -> None:
        self.action_space = gym.spaces.Box(
            low=np.array([-self.action_limit], dtype=np.float32),
            high=np.array([self.action_limit], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

    def _reset_internal_state(self) -> None:
        self.current_state = None
        self.current_observation = None
        self.current_reference = None
        self.done = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset()

        if seed is not None:
            np.random.seed(seed)

        self.current_state = build_initial_double_pendulum_state(
            simulator=self.simulator,
            initial_theta_factor=self.initial_theta_factor,
        )

        self.current_observation = get_double_pendulum_observation(
            self.current_state
        )

        self.current_reference = None
        self.done = False
        self.elapsed_steps = 0

        info = {}

        return self.current_observation.copy(), info

    def step(self, action: Any):
        if self.done:
            raise RuntimeError(
                f"[{self.name}] O episódio já terminou. "
                "Chama reset() antes de voltar a usar step()."
            )

        u = process_double_pendulum_action(action)

        try:
            x_next = simulate_double_pendulum_step(
                simulator=self.simulator,
                action=u,
            )

        except Exception as error:
            self.done = True

            reward = -100.0
            terminated = True
            truncated = False

            info = build_double_pendulum_info(
                action=u,
                state=self.current_state,
                config=self.reward_config,
                extra={"simulation_error": str(error)},
            )

            return (
                self.current_observation.copy(),
                float(reward),
                bool(terminated),
                bool(truncated),
                info,
            )

        self.elapsed_steps += 1

        self.current_state = x_next
        self.current_observation = get_double_pendulum_observation(
            self.current_state
        )

        reward = compute_double_pendulum_reward(
            state=self.current_state,
            action=u.flatten(),
            config=self.reward_config,
        )

        terminated = check_double_pendulum_termination(
            state=self.current_state,
            config=self.reward_config,
        )

        truncated = self.elapsed_steps >= self.max_steps

        if terminated:
            reward -= self.termination_penalty

        info = build_double_pendulum_info(
            action=u,
            state=self.current_state,
            config=self.reward_config,
        )

        self.done = bool(terminated or truncated)

        return (
            self.current_observation.copy(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> None:
        if self.current_state is None:
            return

        if self.fig is None:
            (
                self.fig,
                self.ax,
                self.bar1,
                self.bar2,
            ) = initialize_double_pendulum_renderer(name=self.name)

        render_double_pendulum(
            fig=self.fig,
            ax=self.ax,
            bar1=self.bar1,
            bar2=self.bar2,
            state=self.current_state,
            L1=self.L1,
            L2=self.L2,
        )

    def close(self) -> None:
        close_double_pendulum_renderer(self.fig)

        self.fig = None
        self.ax = None
        self.bar1 = None
        self.bar2 = None

    def get_state(self):
        if self.current_state is None:
            return None

        return self.current_state.copy()