# Calcula métricas genéricas e, se existirem, métricas específicas do ambiente através do ENVIRONMENT_REGISTRY.
from __future__ import annotations
from importlib import import_module
from typing import Any, Dict, List
from evaluation.metrics import (summarize_episode_metrics,aggregate_metric,)
from registries import ENVIRONMENT_REGISTRY

class Evaluator:
    """ Classe responsável por avaliar os resultados de simulação.
    recebe resultados já produzidos pelo runner e transforma esses resultados em métricas. """

    def _compute_environment_specific_metrics(
        self,
        episode_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calcula métricas específicas de um ambiente, caso existam.

        Exemplo:
        Para o double inverted pendulum, podem existir métricas como:
        - erro médio dos ângulos;
        - taxa perto da posição vertical;
        - taxa de estabilização.
        """
        # Obtém o nome do ambiente usado no episódio.
        environment_name = episode_result.get("environment_name")
        # Se não existir nome do ambiente, não há métricas específicas.
        if environment_name is None:
            return {}
        # Vai ao registry buscar informação sobre esse ambiente.
        env_spec = ENVIRONMENT_REGISTRY.get(environment_name)
        # Se o ambiente não estiver registado, não calcula métricas específicas.
        if env_spec is None:
            return {}
        # Procura no registry se esse ambiente tem métricas específicas definidas.
        metrics_spec = env_spec.get("metrics")
        if metrics_spec is None:
            return {}

        # Caminho do módulo onde está a função de métricas específicas.
        module_path = metrics_spec.get("module_path")
        # Nome da função que calcula essas métricas.
        callable_name = metrics_spec.get("callable_name")

        if not module_path or not callable_name:
            return {}
        try:
            # Importa dinamicamente o módulo das métricas específicas.
            module = import_module(module_path)
            # Obtém a função específica pelo nome.
            metrics_callable = getattr(module, callable_name)
            # Executa a função e devolve as métricas específicas.
            return metrics_callable(episode_result)
        except Exception as error:
            # Se algo falhar, a avaliação continua sem métricas específicas.
            print(
                f"[Evaluator] Aviso: falha ao calcular métricas "
                f"específicas do ambiente '{environment_name}': {error}"
            )
            return {}

    def evaluate_episode(
        self,
        episode_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """ Avalia um único episódio. """
        # Calcula métricas genéricas:
        # reward, passos, ações, energia, tempos, etc.
        metrics = summarize_episode_metrics(episode_result)
        # Calcula métricas específicas do ambiente, se existirem.
        environment_metrics = self._compute_environment_specific_metrics(
            episode_result
        )
        # Junta métricas genéricas + métricas específicas.
        metrics.update(environment_metrics)
        return metrics

    def evaluate_episodes(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """ Avalia vários episódios individualmente. """
        return [
            self.evaluate_episode(result)
            for result in results
        ]

    def summarize_evaluation(
        self,
        metrics_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Cria um resumo agregado de vários episódios.

        Para cada métrica calcula:
        - média;
        - desvio padrão;
        - mínimo;
        - máximo.
        """

        if not metrics_list:
            raise ValueError("metrics_list não pode estar vazia.")

        # Número total de episódios avaliados.
        num_episodes = len(metrics_list)
        # Percentagem de episódios que terminaram por falha/condição terminal.
        termination_rate = (
            sum(
                1
                for m in metrics_list
                if m.get("terminated", False)
            )
            / num_episodes
        )
        # Resumo das métricas principais.
        summary = {
            "controller_name": metrics_list[0].get("controller_name"),
            "controller_type": metrics_list[0].get("controller_type"),
            "controller_family": metrics_list[0].get("controller_family"),
            "environment_name": metrics_list[0].get("environment_name"),
            "environment_type": metrics_list[0].get("environment_type"),
            "num_episodes": num_episodes,
            "termination_rate": termination_rate,
            "total_reward": aggregate_metric(metrics_list, "total_reward"),
            "num_steps": aggregate_metric(metrics_list, "num_steps"),
            "average_reward": aggregate_metric(metrics_list, "average_reward"),
            "max_reward": aggregate_metric(metrics_list, "max_reward"),
            "min_reward": aggregate_metric(metrics_list, "min_reward"),
            "mean_action_magnitude": aggregate_metric(metrics_list, "mean_action_magnitude"),
            "max_action_magnitude": aggregate_metric(metrics_list, "max_action_magnitude"),
            "action_smoothness": aggregate_metric(metrics_list, "action_smoothness"),
            "control_energy": aggregate_metric(metrics_list, "control_energy"),
            "mean_tracking_error": aggregate_metric(metrics_list, "mean_tracking_error"),
            "max_tracking_error": aggregate_metric(metrics_list, "max_tracking_error"),
            "final_tracking_error": aggregate_metric(metrics_list, "final_tracking_error"),
            "mean_step_time": aggregate_metric(metrics_list, "mean_step_time"),
            "total_step_time": aggregate_metric(metrics_list, "total_step_time"),
        }

        # Identifica métricas extra que não estavam previstas no resumo base.
        # Isto permite agregar automaticamente métricas específicas do ambiente.
        ignored_keys = set(summary.keys())
        ignored_keys.update(
            {
                "controller_name",
                "controller_type",
                "controller_family",
                "environment_name",
                "environment_type",
                "terminated",
                "truncated",
            }
        )
        extra_metric_names = set()
        for metrics in metrics_list:
            for key in metrics.keys():
                if key not in ignored_keys:
                    extra_metric_names.add(key)

        # Agrega dinamicamente todas as métricas específicas encontradas.
        for metric_name in sorted(extra_metric_names):
            summary[metric_name] = aggregate_metric(
                metrics_list,
                metric_name,
            )
        return summary