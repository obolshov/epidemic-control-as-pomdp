"""Action profile plot: per-agent action choices across a representative episode.

For each agent, selects the training seed closest to cross-seed mean reward,
then the episode closest to that seed's mean. Plots a step function of
intervention severity over days.

Usage:
    python -m analysis.action_profile <experiment_path>
    python -m analysis.action_profile <experiment_path> --agents ppo_baseline ppo_framestack
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from analysis.data import EXPERIMENTS_DIR
from src.env import InterventionAction
from src.utils import _save_or_show

ACTION_INDEX = {a.name: i for i, a in enumerate(InterventionAction)}

DISPLAY_NAMES = {
    "ppo_baseline": "Memoryless",
    "ppo_framestack": "FrameStack",
    "ppo_recurrent": "Recurrent",
}


def find_representative_episode(
    evaluation: dict,
    summary: dict,
    agent_name: str,
) -> tuple[str, int, int]:
    """Find the episode closest to cross-seed mean reward.

    Args:
        evaluation: Parsed evaluation.json.
        summary: Parsed summary.json.
        agent_name: Agent to select for.

    Returns:
        (training_seed_str, episode_idx, eval_seed).
    """
    cross_seed_mean = summary["agents"][agent_name]["cross_seed_mean_reward"]
    seeds_data = evaluation[agent_name]["seeds"]

    best_seed = min(
        seeds_data,
        key=lambda s: abs(np.mean(seeds_data[s]["total_reward"]) - cross_seed_mean),
    )
    best_seed_mean = float(np.mean(seeds_data[best_seed]["total_reward"]))

    chosen = seeds_data[best_seed]
    rewards = np.array(chosen["total_reward"])
    episode_idx = int(np.argmin(np.abs(rewards - best_seed_mean)))
    eval_seed = chosen["eval_seeds"][episode_idx]

    return best_seed, episode_idx, eval_seed


def load_episode_log(
    run_dir: Path,
    agent_name: str,
    training_seed: str,
    eval_seed: int,
) -> dict:
    """Load a JSON episode log file.

    Tries RL path (train_{seed}/) first, falls back to baseline path.

    Args:
        run_dir: Experiment run directory.
        agent_name: Agent name.
        training_seed: Training seed string.
        eval_seed: Evaluation seed for the episode.

    Returns:
        Parsed JSON log data.
    """
    rl_path = run_dir / "logs" / agent_name / f"train_{training_seed}" / f"seed_{eval_seed}.json"
    baseline_path = run_dir / "logs" / agent_name / f"seed_{eval_seed}.json"

    log_path = rl_path if rl_path.exists() else baseline_path

    if not log_path.exists():
        raise FileNotFoundError(
            f"No log file found for {agent_name} seed={eval_seed} "
            f"(tried {rl_path} and {baseline_path})"
        )

    with open(log_path) as f:
        return json.load(f)


def plot_action_profiles(
    profiles: dict[str, tuple[list[int], list[int]]],
    save_path: Optional[str] = None,
) -> None:
    """Plot action step functions for multiple agents.

    Args:
        profiles: Dict of agent_name -> (days, action_indices).
        save_path: Path to save figure. If None, displays interactively.
    """
    n_agents = len(profiles)
    fig, axes = plt.subplots(
        n_agents, 1,
        figsize=(10, 2.0 * n_agents + 0.5),
        sharex=True,
        squeeze=False,
    )

    colors = plt.cm.tab10.colors

    for idx, (agent_name, (days, actions)) in enumerate(profiles.items()):
        ax = axes[idx, 0]
        color = colors[idx % len(colors)]

        ax.step(days, actions, where="post", color=color, linewidth=1.5)
        ax.fill_between(days, actions, step="post", color=color, alpha=0.2)

        ax.set_title(DISPLAY_NAMES.get(agent_name, agent_name), fontsize=11)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["NO", "MILD", "MODERATE", "SEVERE"], fontsize=9)
        ax.set_ylim(-0.3, 3.3)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Day")
    fig.suptitle("Action Profile (representative episode)", fontsize=11)
    plt.tight_layout()
    _save_or_show(save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-agent action profiles for a representative episode."
    )
    parser.add_argument(
        "experiment",
        help="Path relative to experiments/ (e.g. pomdp_t3000000/2025-05-20_12-00-00)",
    )
    parser.add_argument(
        "--agents", nargs="+", default=None,
        help="Agents to plot (default: all ppo_* agents)",
    )
    args = parser.parse_args()

    run_dir = EXPERIMENTS_DIR / args.experiment
    if not run_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {run_dir}")

    with open(run_dir / "evaluation.json") as f:
        evaluation = json.load(f)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)

    agent_names = args.agents or [
        name for name in evaluation if name.startswith("ppo_")
    ]

    profiles: dict[str, tuple[list[int], list[int]]] = {}

    for agent_name in agent_names:
        if agent_name not in evaluation:
            print(f"  {agent_name}: not found in evaluation.json, skipping")
            continue

        train_seed, ep_idx, eval_seed = find_representative_episode(
            evaluation, summary, agent_name,
        )
        print(
            f"  {agent_name}: train_seed={train_seed}, "
            f"episode={ep_idx}, eval_seed={eval_seed}"
        )

        try:
            log_data = load_episode_log(run_dir, agent_name, train_seed, eval_seed)
        except FileNotFoundError as e:
            print(f"    {e}")
            continue

        days = [step["day"] for step in log_data["steps"]]
        actions = [ACTION_INDEX[step["action"]] for step in log_data["steps"]]
        profiles[agent_name] = (days, actions)

    if not profiles:
        print("No action profiles to plot.")
        return

    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    save_path = str(output_dir / "action_profile.png")

    plot_action_profiles(profiles, save_path=save_path)
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    main()
