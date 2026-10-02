from __future__ import annotations
from comparison.controller_comparator import ControllerComparator
from evaluation.evaluation_runner import run_evaluation_episodes
from evaluation.results_builder import build_evaluation_result
from results import (save_training_result,save_simulation_result,save_evaluation_result,save_comparison_result,)
from simulation.results import (TrainingResult,SimulationResult,EvaluationResult,ComparisonResult,)
from training.rl import train_rl

# =========================
# Funções de impressão
# =========================

def print_training_result(training_result: TrainingResult) -> None:
    """ Imprime no terminal o resultado final do treino."""
    print("\n=== TREINO CONCLUÍDO ===")
    print(f"Controller: {training_result.controller_name}")
    print(f"Environment: {training_result.environment_name}")
    print(f"Total timesteps: {training_result.total_timesteps}")
    print(f"Modelo guardado em: {training_result.model_path}")

def print_simulation_result(metrics: dict) -> None:
    """ Imprime métricas de uma simulação de episódio único. """
    print("=== MÉTRICAS DA SIMULAÇÃO ===")
    print(f"Controller: {metrics['controller_name']}")
    print(f"Environment: {metrics['environment_name']}")
    print(f"Total reward: {metrics['total_reward']}")
    print(f"Num steps: {metrics['num_steps']}")
    print(f"Average reward: {metrics['average_reward']}")
    print(f"Max reward: {metrics['max_reward']}")
    print(f"Min reward: {metrics['min_reward']}")
    print(f"Mean action magnitude: {metrics['mean_action_magnitude']}")
    print(f"Action smoothness: {metrics['action_smoothness']}")
    print(f"Mean tracking error: {metrics['mean_tracking_error']}")
    print(f"Mean step time: {metrics['mean_step_time']}")
    print(f"Total step time: {metrics['total_step_time']}")
    print(f"Terminated: {metrics['terminated']}")

def print_evaluation_result(
    metrics_list: list[dict],
    summary: dict,
) -> None:
    """Imprime métricas de vários episódios e o resumo agregado."""
    print("\n=== MÉTRICAS POR EPISÓDIO ===")
    # Mostra resultados individuais de cada episódio.
    for i, metrics in enumerate(metrics_list, start=1):
        print(
            f"Episódio {i}: "
            f"total_reward={metrics['total_reward']}, "
            f"num_steps={metrics['num_steps']}, "
            f"average_reward={metrics['average_reward']}, "
            f"terminated={metrics['terminated']}"
        )
    print("\n=== RESUMO DA AVALIAÇÃO ===")
    print(f"Controller: {summary['controller_name']}")
    print(f"Environment: {summary['environment_name']}")
    print(f"Num episodes: {summary['num_episodes']}")
    print(f"Termination rate: {summary['termination_rate']}")
    print(f"Average total reward: {summary['total_reward']['mean']}")
    print(f"Total reward std: {summary['total_reward']['std']}")
    print(f"Average num steps: {summary['num_steps']['mean']}")
    print(f"Average reward per step: {summary['average_reward']['mean']}")
    print(f"Mean action magnitude: {summary['mean_action_magnitude']['mean']}")
    print(f"Action smoothness: {summary['action_smoothness']['mean']}")
    print(f"Mean tracking error: {summary['mean_tracking_error']['mean']}")
    print(f"Final tracking error: {summary['final_tracking_error']['mean']}")
    print(f"Mean step time: {summary['mean_step_time']['mean']}")
    print(f"Total step time: {summary['total_step_time']['mean']}")

def print_environment_specific_metrics(summary: dict) -> None:
    """
    Imprime automaticamente as métricas específicas do ambiente.

    As métricas genéricas já apresentadas no resumo são ignoradas.
    """
    generic_keys = {
        "controller_name",
        "controller_type",
        "controller_family",
        "environment_name",
        "environment_type",
        "num_episodes",
        "termination_rate",
        "total_reward",
        "num_steps",
        "average_reward",
        "max_reward",
        "min_reward",
        "mean_action_magnitude",
        "max_action_magnitude",
        "action_smoothness",
        "control_energy",
        "mean_tracking_error",
        "max_tracking_error",
        "final_tracking_error",
        "mean_step_time",
        "total_step_time",
    }

    specific_metrics = {
        key: value
        for key, value in summary.items()
        if key not in generic_keys
    }

    if not specific_metrics:
        return

    print("  Environment-specific metrics:")

    for metric_name, metric_data in specific_metrics.items():
        if isinstance(metric_data, dict):
            mean_value = metric_data.get("mean")

            if mean_value is not None:
                print(f"    {metric_name}: {mean_value}")
        elif metric_data is not None:
            print(f"    {metric_name}: {metric_data}")

def print_comparison_result(
    comparison_result: ComparisonResult,
    output_path,
) -> None:
    """ Imprime resumo da comparação entre controladores. """
    print("\n=== RESUMO DA COMPARAÇÃO ===")
    # Percorre cada controlador comparado e imprime o resumo.
    for controller_type, result in comparison_result.results.items():
        summary = result["summary"]
        print(f"\nController: {controller_type}")
        print(f"  Average total reward: {summary['total_reward']['mean']}")
        print(f"  Average reward per step: {summary['average_reward']['mean']}")
        print(f"  Termination rate: {summary['termination_rate']}")
        print(f"  Average num steps: {summary['num_steps']['mean']}")
        print(f"  Mean action magnitude: {summary['mean_action_magnitude']['mean']}")
        print(f"  Action smoothness: {summary['action_smoothness']['mean']}")
        print(f"  Mean step time: {summary['mean_step_time']['mean']}")
        print_environment_specific_metrics(summary)

    print("\n=== COMPARAÇÃO CONCLUÍDA ===")
    print(f"Resultado guardado em: {output_path}")

# =========================
# Modos de execução
# =========================

def run_train_mode(config: dict) -> TrainingResult:
    """ Executa o modo de treino para controladores RL. """
    # Chama a função de treino RL.
    model_path = train_rl(
        controller_type=config["controller_type"],
        env_type=config["environment_type"],
        total_timesteps=config["total_timesteps"],
        env_overrides=config["env_overrides"],
        controller_overrides=config["controller_overrides"],
        model_name=config["trained_model_name"],
    )
    # Cria objeto estruturado com o resultado do treino.
    training_result = TrainingResult(
        controller_name=config["controller_type"],
        environment_name=config["environment_type"],
        total_timesteps=config["total_timesteps"],
        model_path=str(model_path),
    )
    # Guarda resultado em ficheiro.
    save_training_result(training_result)
    # Imprime resultado no terminal.
    print_training_result(training_result)
    return training_result

def run_simulate_mode(
    config: dict,
    runtime: dict,
) -> SimulationResult:
    """ Executa o modo de simulação para um único episódio. """
    # Corre um episódio usando o runner já criado pelo runtime_builder.
    episode_result = runtime["runner"].run_episode(
        max_steps=config["max_steps"],
        render=config["render"],
        render_delay=config.get("render_delay", 0.2),
    )
    # Calcula métricas do episódio.
    metrics = runtime["evaluator"].evaluate_episode(episode_result)
    # Organiza resultado da simulação.
    simulation_result = SimulationResult(
        controller_name=metrics["controller_name"],
        environment_name=metrics["environment_name"],
        episode_result=episode_result,
        metrics=metrics,
    )
    # Guarda resultado.
    save_simulation_result(simulation_result)
    # Imprime métricas.
    print_simulation_result(metrics)
    return simulation_result

def run_evaluate_mode(config: dict) -> EvaluationResult:
    """Executa o modo de avaliação para vários episódios."""
    # Corre vários episódios de avaliação.
    evaluation_data = run_evaluation_episodes(
        config=config,
        num_episodes=config["num_episodes"],
        max_steps=config["max_steps"],
        render=config["render"],
        render_delay=config.get("render_delay", 0.2),
    )
    # Extrai resultados e métricas.
    episode_results = evaluation_data["episode_results"]
    metrics_list = evaluation_data["metrics_list"]
    summary = evaluation_data["summary"]
    # Organiza resultado num formato padronizado.
    built_result = build_evaluation_result(
        episode_metrics=metrics_list,
        summary_metrics=summary,
    )
    # Cria dataclass de resultado de avaliação.
    evaluation_result = EvaluationResult(
        controller_name=built_result["controller_name"],
        environment_name=built_result["environment_name"],
        episode_results=episode_results,
        metrics_list=built_result["episode_metrics"],
        summary=built_result["summary_metrics"],
    )
    # Guarda resultado.
    save_evaluation_result(evaluation_result)
    # Imprime resultado no terminal.
    print_evaluation_result(metrics_list, summary)
    return evaluation_result


def run_compare_mode(config: dict) -> ComparisonResult:
    """ Executa o modo de comparação entre vários controladores. """

    # Cria o comparador de controladores.
    comparator = ControllerComparator(
        environment_type=config["environment_type"],
        controller_types=config["comparison_controllers"],
        num_episodes=config["num_episodes"],
        max_steps=config["max_steps"],
        render=config["render"],
        env_overrides=config["env_overrides"],
        controller_overrides=config.get("comparison_controller_overrides", {}),
        use_trained_models=config.get("comparison_use_trained_models", {}),
        trained_model_names=config.get("comparison_trained_model_names", {}),
    )

    # Executa a comparação.
    comparison_result = comparator.run()
    # Guarda resultado da comparação.
    output_path = save_comparison_result(comparison_result)
    # Imprime resumo.
    print_comparison_result(comparison_result, output_path)
    return comparison_result