# training/rl/train_rl.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from factories import create_controller, create_environment
from registries import CONTROLLER_REGISTRY


class TrainingMetricsCallback(BaseCallback):
    """
    Callback simples para recolher métricas de aprendizagem durante o treino RL.

    Métricas calculadas:
    - reward acumulada por episódio;
    - evolução da reward por média móvel;
    - reward final, média e máxima;
    - estabilidade do treino;
    - episódios até convergência;
    - taxa de convergência.
    """

    def __init__(
        self,
        window_size: int = 10,
        convergence_threshold: float = 0.05,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)

        self.window_size = window_size
        self.convergence_threshold = convergence_threshold

        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is None or dones is None:
            return True

        reward = float(np.asarray(rewards).flatten()[0])
        done = bool(np.asarray(dones).flatten()[0])

        self.current_episode_reward += reward

        if done:
            self.episode_rewards.append(float(self.current_episode_reward))
            self.current_episode_reward = 0.0

        return True

    def get_training_metrics(self) -> Dict[str, Any]:
        rewards = self.episode_rewards

        if not rewards:
            return {
                "num_training_episodes": 0,
                "episode_rewards": [],
                "moving_average_reward": [],
                "final_cumulative_reward": None,
                "best_cumulative_reward": None,
                "mean_cumulative_reward": None,
                "training_stability": None,
                "episodes_to_convergence": None,
                "convergence_rate": None,
            }

        moving_average = [
            float(np.mean(rewards[max(0, i - self.window_size + 1): i + 1]))
            for i in range(len(rewards))
        ]

        recent_rewards = rewards[-self.window_size:]
        training_stability = (
            float(np.std(recent_rewards))
            if len(recent_rewards) >= 2
            else 0.0
        )

        episodes_to_convergence = self._estimate_episodes_to_convergence(
            moving_average
        )

        convergence_rate = None
        if episodes_to_convergence is not None and episodes_to_convergence > 0:
            initial_reward = rewards[0]
            converged_reward = rewards[episodes_to_convergence - 1]
            convergence_rate = float(
                (converged_reward - initial_reward) / episodes_to_convergence
            )

        return {
            "num_training_episodes": len(rewards),
            "episode_rewards": rewards,
            "moving_average_reward": moving_average,
            "final_cumulative_reward": float(rewards[-1]),
            "best_cumulative_reward": float(max(rewards)),
            "mean_cumulative_reward": float(np.mean(rewards)),
            "training_stability": training_stability,
            "episodes_to_convergence": episodes_to_convergence,
            "convergence_rate": convergence_rate,
        }

    def _estimate_episodes_to_convergence(
        self,
        moving_average: list[float],
    ) -> Optional[int]:
        if len(moving_average) < 2 * self.window_size:
            return None

        for i in range(self.window_size, len(moving_average)):
            previous = moving_average[i - self.window_size]
            current = moving_average[i]

            scale = max(abs(previous), 1e-8)
            relative_change = abs(current - previous) / scale

            if relative_change <= self.convergence_threshold:
                return i + 1

        return None


def validate_rl_controller(controller_type: str) -> None:
    """Valida se o controlador pedido existe no registry e se pertence à família RL."""
    if controller_type not in CONTROLLER_REGISTRY:
        raise ValueError(f"Controlador desconhecido: '{controller_type}'.")

    spec = CONTROLLER_REGISTRY[controller_type]

    if spec["family"] != "rl":
        raise ValueError(
            f"O controlador '{controller_type}' não pertence à família RL."
        )


def build_model_path(
    controller_type: str,
    env_type: str,
    model_name: Optional[str] = None,
) -> Path:
    """Constrói o caminho onde o modelo treinado será guardado."""
    base_dir = Path("models") / "rl" / controller_type / env_type
    base_dir.mkdir(parents=True, exist_ok=True)

    final_name = model_name or f"{controller_type}_{env_type}"

    return base_dir / final_name


def train_rl(
    controller_type: str,
    env_type: str,
    total_timesteps: int,
    env_overrides: Optional[Dict[str, Any]] = None,
    controller_overrides: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> Path:
    """
    Treina um controlador RL.

    Fluxo:
    1. Valida o controlador no registry.
    2. Cria o ambiente pela environment_factory.
    3. Cria o controlador pela controller_factory.
    4. Inicializa o controlador.
    5. Treina o modelo com recolha de métricas de aprendizagem.
    6. Guarda o modelo treinado em models/.
    7. Fecha o ambiente no fim, mesmo que ocorra erro.
    """

    validate_rl_controller(controller_type)

    print("\n========== TREINO RL ==========")
    print(f"Controlador: {controller_type}")
    print(f"Ambiente: {env_type}")
    print(f"Timesteps de treino: {total_timesteps}")

    env = create_environment(
        env_type=env_type,
        overrides=env_overrides,
    )

    try:
        controller = create_controller(
            controller_type=controller_type,
            env=env,
            overrides=controller_overrides,
        )
        controller.initialize()

        print("\n[INFO] Configuração final do controlador:")
        for key, value in controller.get_info().items():
            print(f"  - {key}: {value}")

        callback = TrainingMetricsCallback(
            window_size=10,
            convergence_threshold=0.05,
        )

        print("\n[INFO] A iniciar treino...")
        controller.train_step(
            total_timesteps=total_timesteps,
            callback=callback,
        )
        print("[INFO] Treino concluído com sucesso.")

        training_metrics = callback.get_training_metrics()

        print("\n[INFO] Métricas de aprendizagem RL:")
        for key, value in training_metrics.items():
            if key in {"episode_rewards", "moving_average_reward"}:
                print(f"  - {key}: {len(value)} valores registados")
            else:
                print(f"  - {key}: {value}")

        model_path = build_model_path(
            controller_type=controller_type,
            env_type=env_type,
            model_name=model_name,
        )

        controller.save(str(model_path))
        print(f"[INFO] Modelo guardado em: {model_path}")

        return model_path

    finally:
        if hasattr(env, "close"):
            env.close()