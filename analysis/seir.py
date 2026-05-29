"""Publication-ready SEIR trajectory grid for a finished experiment.

Re-creates the multi-agent SEIR comparison figure (one panel per agent, 4+3
layout) but styled for a paper: no per-panel metrics box, a single shared
legend, larger fonts, thicker lines, and a shared y-scale with abbreviated
tick labels.

Every agent's mean SEIR curves are reconstructed by re-evaluation from the
saved ``config.json`` (so all panels share the same daily resolution).
Baselines are re-run with their original per-seed eval-seed groups; RL agents
are re-evaluated from their saved best checkpoint + VecNormalize stats in
``<scenario>/weights/``. Fixed seeds reproduce the original figure's curves.

Usage:
    python -m analysis.seir <experiment_path>
    python -m analysis.seir pomdp_t3000000/default
    python -m analysis.seir pomdp_t3000000/default --agents no_action ppo_framestack
"""

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from analysis.data import EXPERIMENTS_DIR
from src.config import (
    Config,
    DQNConfig,
    PPOBaselineConfig,
    PPOFrameStackConfig,
    PPORecurrentConfig,
)
from src.results import AggregatedResult
from src.utils import display_name

SEIR_COLORS = {"S": "blue", "E": "orange", "I": "red", "R": "green"}
SEIR_LABELS = {
    "S": "Susceptible (S)",
    "E": "Exposed (E)",
    "I": "Infected (I)",
    "R": "Recovered (R)",
}

# Publication styling
TITLE_FONTSIZE = 40
TICK_FONTSIZE = 32
AXIS_LABEL_FONTSIZE = 46
LEGEND_FONTSIZE = 38
LINEWIDTH = 7.0


def _config_from_dict(base_config: dict) -> Config:
    """Reconstruct a Config from a serialized base_config dict.

    config.json is written via ``asdict`` (src/experiment.py), which flattens
    the nested agent dataclasses into plain dicts. Rebuild them before
    constructing the top-level Config.

    Args:
        base_config: The ``base_config`` mapping from config.json.

    Returns:
        Reconstructed Config instance.
    """
    nested = {
        "ppo_baseline": PPOBaselineConfig,
        "ppo_framestack": PPOFrameStackConfig,
        "ppo_recurrent": PPORecurrentConfig,
        "dqn": DQNConfig,
    }
    kwargs = dict(base_config)
    for key, cls in nested.items():
        if key in kwargs:
            kwargs[key] = cls(**kwargs[key])
    return Config(**kwargs)


def _reeval_curves(
    config: Config,
    pomdp_params: dict,
    run_dir: Path,
    training_seeds: List[int],
    eval_seeds: List[int],
    n_eval: int,
    agent: str,
) -> AggregatedResult:
    """Reconstruct mean SEIR curves for one agent by re-evaluation.

    Baselines (``no_action``, ``severe``, ``random``, ``threshold``) are re-run
    with the same per-seed eval-seed groups as the original training run (and
    the same RandomAgent re-seeding). RL agents are re-evaluated from their
    saved best checkpoint plus VecNormalize stats in ``<scenario>/weights/``.
    Fixed seeds make the curves reproduce the original figure.

    Args:
        config: Reconstructed base Config.
        pomdp_params: POMDP wrapper parameters from config.json.
        run_dir: Experiment run directory (weights live in ``run_dir.parent``).
        training_seeds: Training seeds.
        eval_seeds: Shared eval seeds for RL agents.
        n_eval: Episodes per seed group (config ``num_eval_episodes``).
        agent: Agent name.

    Returns:
        Cross-seed aggregated result with mean S/E/I/R trajectories.
    """
    # Heavy SB3/env dependencies are imported lazily at call time.
    from src.agents import RandomAgent, create_baseline_agents
    from src.evaluation import aggregate_across_seeds, evaluate_agent
    from src.train import _load_model

    n_seeds = len(training_seeds)
    baseline = create_baseline_agents(config, [agent], pomdp_params)
    per_seed = []

    if baseline:
        # Baseline: a distinct eval-seed group per training seed (matches run_evaluation).
        base_agent = baseline[0]
        seed_groups = [
            list(range(2024 + i * n_eval, 2024 + (i + 1) * n_eval)) for i in range(n_seeds)
        ]
        for group in seed_groups:
            run_agent = (
                RandomAgent(seed=group[0]) if isinstance(base_agent, RandomAgent) else base_agent
            )
            agg, _ = evaluate_agent(run_agent, config, pomdp_params, agent, group)
            per_seed.append(agg)
        eval_seeds_per_seed = seed_groups
    else:
        # RL: load each seed's best checkpoint, evaluate on the shared eval seeds.
        weights_dir = run_dir.parent / "weights"
        for seed in training_seeds:
            model_path = weights_dir / f"best_{agent}_seed{seed}" / "best_model.zip"
            vecnorm_path = weights_dir / f"{agent}_seed{seed}_vecnormalize.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"No saved model for RL agent '{agent}': {model_path}")
            model = _load_model(str(model_path), agent)
            agg, _ = evaluate_agent(
                model, config, pomdp_params, agent, eval_seeds, vecnorm_path=str(vecnorm_path)
            )
            per_seed.append(agg)
        eval_seeds_per_seed = [eval_seeds] * n_seeds

    combined, _ = aggregate_across_seeds(
        per_seed, training_seeds, agent, eval_seeds_per_seed=eval_seeds_per_seed
    )
    return combined


def _make_axes(num_agents: int):
    """Build the panel layout: one row for <=5 agents, else two rows (shorter
    bottom row centered via GridSpec). Mirrors src.utils.plot_all_results.

    Returns:
        Tuple of (figure, list of axes, set of left-edge panel indices that
        keep y-tick labels).
    """
    if num_agents <= 5:
        fig, ax_array = plt.subplots(
            1, num_agents, figsize=(7 * num_agents, 6), layout="constrained"
        )
        axes = [ax_array] if num_agents == 1 else list(ax_array)
        return fig, axes, {0}

    ncols = ceil(num_agents / 2)
    second_row_count = num_agents - ncols
    figsize = (7 * ncols, 12)

    if second_row_count == ncols:
        fig, ax_array = plt.subplots(2, ncols, figsize=figsize, layout="constrained")
        return fig, list(ax_array.flatten()), {0, ncols}

    gs_cols = 2 * ncols
    left_pad = (gs_cols - second_row_count * 2) // 2
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(2, gs_cols, figure=fig)
    axes = [fig.add_subplot(gs[0, i * 2: i * 2 + 2]) for i in range(ncols)]
    for i in range(second_row_count):
        col_start = left_pad + i * 2
        axes.append(fig.add_subplot(gs[1, col_start: col_start + 2]))
    return fig, axes, {0, ncols}


def _thousands(value: float, _pos) -> str:
    """Format a y-tick as an abbreviated count (200000 -> '200k')."""
    if value == 0:
        return "0"
    return f"{value / 1000:.0f}k"


def plot_seir_grid(
    curves: Dict[str, AggregatedResult],
    population: int,
    days: int,
    save_path_base: Path,
) -> None:
    """Render the publication SEIR grid and save it as PNG and PDF.

    Args:
        curves: Ordered mapping agent_name -> AggregatedResult (panel order preserved).
        population: Total population N, sets the shared y-axis limit.
        days: Total simulation length in days, sets the shared x-axis limit.
        save_path_base: Output path without extension; ``.png`` and ``.pdf``
            are written.
    """
    fig, axes, left_panels = _make_axes(len(curves))
    # Extra vertical gap between the two panel rows so the bottom row's titles
    # clear the top row's x-axis ticks (constrained_layout default is ~0.02).
    fig.get_layout_engine().set(hspace=0.14)

    for idx, (agent, c) in enumerate(curves.items()):
        ax = axes[idx]
        for comp in ("S", "E", "I", "R"):
            ax.plot(c.t, getattr(c, f"{comp}_mean"), color=SEIR_COLORS[comp], linewidth=LINEWIDTH)

        ax.set_title(display_name(agent), fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlim(0, days)
        ax.set_ylim(0, population * 1.02)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

        if idx in left_panels:
            ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
        else:
            ax.tick_params(labelleft=False)

    handles = [Line2D([0], [0], color=SEIR_COLORS[c], linewidth=10) for c in ("S", "E", "I", "R")]
    labels = [SEIR_LABELS[c] for c in ("S", "E", "I", "R")]

    # constrained_layout (set in _make_axes) places the shared axis labels
    # without overlapping panels. The legend sits just below the figure; the
    # bbox_inches="tight" save expands the canvas to include it.
    fig.supxlabel("Time (days)", fontsize=AXIS_LABEL_FONTSIZE)
    fig.supylabel("Number of people", fontsize=AXIS_LABEL_FONTSIZE)
    fig.legend(
        handles, labels,
        loc="upper center", ncol=4, fontsize=LEGEND_FONTSIZE,
        frameon=False, bbox_to_anchor=(0.5, -0.01),
    )

    png_path = save_path_base.with_suffix(".png")
    pdf_path = save_path_base.with_suffix(".pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved:\n  {png_path}\n  {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication-ready SEIR trajectory grid for an experiment.",
    )
    parser.add_argument(
        "experiment",
        help="Path relative to experiments/ (e.g. pomdp_t3000000/default).",
    )
    parser.add_argument(
        "--agents", nargs="+", default=None,
        help="Only plot these agents (default: all target_agents from config).",
    )
    args = parser.parse_args()

    run_dir = EXPERIMENTS_DIR / args.experiment
    if not run_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {run_dir}")

    with open(run_dir / "config.json") as f:
        config_json = json.load(f)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)

    base_config = config_json["base_config"]
    pomdp_params = config_json["pomdp_params"]
    scenario_name = config_json["scenario_name"]
    population = base_config["N"]
    days = base_config["days"]
    training_seeds = config_json["training_seeds"]
    eval_seeds = config_json["eval_seeds"]
    n_eval = config_json["num_eval_episodes"]

    agents = args.agents or config_json["target_agents"]
    config = _config_from_dict(base_config)

    curves: Dict[str, AggregatedResult] = {}
    for agent in agents:
        combined = _reeval_curves(
            config, pomdp_params, run_dir, training_seeds, eval_seeds, n_eval, agent,
        )
        curves[agent] = combined
        reward = float(np.mean(combined.seed_mean_rewards))
        ref = summary["agents"].get(agent, {}).get("cross_seed_mean_reward")
        ref_str = f" vs summary {ref:.3f}" if ref is not None else ""
        print(f"  {agent}: {combined.n_episodes} episodes, mean reward {reward:.3f}{ref_str}")

    if not curves:
        print("No agents to plot.")
        return

    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    save_path_base = output_dir / f"seir_{scenario_name}"

    plot_seir_grid(curves, population=population, days=days, save_path_base=save_path_base)


if __name__ == "__main__":
    main()
