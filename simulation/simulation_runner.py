# executar episódios de simulação. Faz a interação direta entre controlador e ambiente.
from __future__ import annotations
import time
from typing import Any, Dict, List

class SimulationRunner:
    """ Runner genérico para simular a interação entre um controlador e um ambiente.
    Este ficheiro não sabe se o controlador é RL ou MPC.
    Ele apenas faz:
    observação/estado -> controlador -> ação -> ambiente -> novo estado.
    """

    def __init__(self, controller: Any, environment: Any) -> None:
        # Guarda o controlador que será usado na simulação.
        self.controller = controller

        # Guarda o ambiente que será simulado.
        self.environment = environment

    def _get_controller_input(self, observation: Any) -> Any:
        """  Decide qual informação deve ser enviada ao controlador.
        Regra:
        - RL usa a observação devolvida pelo ambiente;
        - MPC usa o estado físico real, se o ambiente tiver get_state().
        """
        # Verifica a família do controlador: "rl" ou "mpc".
        controller_family = getattr(self.controller, "family", None)
        # Se for MPC e o ambiente tiver get_state(),
        # envia o estado real do sistema.
        if controller_family == "mpc" and hasattr(self.environment, "get_state"):
            return self.environment.get_state()
        # Caso contrário, envia a observação normal.
        return observation

    def _render_environment(self, render_delay: float) -> None:
        """ Renderiza o ambiente, se este tiver método render(). """
        # Verifica se o ambiente consegue renderizar.
        if hasattr(self.environment, "render"):
            self.environment.render()
            # Pequena pausa para a animação não ficar demasiado rápida.
            if render_delay > 0.0:
                time.sleep(render_delay)

    def run_episode(
        self,
        max_steps: int = 500,
        render: bool = False,
        render_delay: float = 0,
        reset_controller: bool = True,
    ) -> Dict[str, Any]:
        """ Executa um episódio completo.
        Um episódio é uma sequência:
        reset -> ação -> step -> ação -> step -> ...
        até terminar ou atingir max_steps.
        """
        # Reinicia o controlador no início do episódio, se pedido.
        if reset_controller:
            self.controller.reset()
        # Reinicia o ambiente.
        reset_result = self.environment.reset()
        # Alguns ambientes devolvem (observation, info).
        # Outros devolvem apenas observation.
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, reset_info = reset_result
        else:
            observation = reset_result
            reset_info = {}
        # Listas onde serão guardados os dados do episódio.
        observations: List[Any] = [observation]
        actions: List[Any] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        step_times: List[float] = []
        # Acumuladores.
        total_reward = 0.0
        steps = 0
        # Flags de término.
        terminated = False
        truncated = False
        # Loop principal do episódio.
        for _ in range(max_steps):
            # Renderização opcional.
            if render:
                self._render_environment(render_delay)
            # Escolhe o input certo para o controlador.
            # RL recebe observation.
            # MPC recebe state, se possível.
            controller_input = self._get_controller_input(observation)
            # Mede o tempo que o controlador demora a calcular a ação.
            start_time = time.perf_counter()
            # Controlador calcula a ação.
            action = self.controller.compute_action(controller_input)
            # Tempo de cálculo da ação.
            step_time = time.perf_counter() - start_time
            # Aplica a ação no ambiente.
            step_result = self.environment.step(action)
            # Formato Gymnasium:
            # observation, reward, terminated, truncated, info
            if len(step_result) == 5:
                next_observation, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)

            # Formato antigo:
            # observation, reward, done, info
            else:
                next_observation, reward, done, info = step_result
                terminated = bool(done)
                truncated = False

            # Guarda dados do passo.
            observations.append(next_observation)
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info if isinstance(info, dict) else {})
            step_times.append(step_time)

            # Atualiza acumuladores.
            total_reward += float(reward)
            steps += 1

            # Atualiza observação atual.
            observation = next_observation

            # Se o episódio terminou, sai do loop.
            if done:
                break

        # Devolve todos os dados brutos do episódio.
        return {
            "controller_name": self.controller.name,
            "controller_type": self.controller.__class__.__name__,
            "controller_family": getattr(self.controller, "family", None),
            "environment_name": self.environment.name,
            "environment_type": self.environment.__class__.__name__,
            "reset_info": reset_info,
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
            "step_times": step_times,
            "total_reward": total_reward,
            "num_steps": steps,
            "terminated": terminated,
            "truncated": truncated,
        }

    def run_episodes(
        self,
        num_episodes: int = 1,
        max_steps: int = 500,
        render: bool = False,
        render_delay: float = 0.0,
        reset_controller_each_episode: bool = True,
    ) -> List[Dict[str, Any]]:
        """ Executa vários episódios consecutivos. """
        results = []

        # Corre vários episódios chamando run_episode().
        for _ in range(num_episodes):
            episode_result = self.run_episode(
                max_steps=max_steps,
                render=render,
                render_delay=render_delay,
                reset_controller=reset_controller_each_episode,
            )
            results.append(episode_result)
        return results