"""Training diagnostics: early stopping, best reward, and convergence step per seed.

Reads evaluations.npz files saved by EvalCallback during training
to show when each agent/seed reached its best performance.

Usage:
    python -m analysis.training_summary pomdp_t3000000/default
    python -m analysis.training_summary pomdp_t3000000/default --agents ppo_baseline ppo_recurrent
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from analysis.data import EXPERIMENTS_DIR


def _parse_eval_dir(dirname: str) -> tuple[str, int] | None:
    """Extract (agent_name, seed) from '{agent}_seed{seed}_eval'."""
    m = re.match(r"(.+)_seed(\d+)_eval$", dirname)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def load_training_stats(
    run_dir: Path,
    total_timesteps: int,
    eval_freq: int,
    agent_filter: list[str] | None = None,
) -> dict[str, dict[int, dict]]:
    """Load per-agent, per-seed training stats from evaluations.npz files.

    Args:
        run_dir: Path to experiment run directory.
        total_timesteps: Max training steps from config.
        eval_freq: Evaluation frequency from config.
        agent_filter: If given, only include these agent names.

    Returns:
        Nested dict: agent_name -> seed -> {best_reward, best_step,
        final_step, early_stopped}.
    """
    log_dir = run_dir / "logs"
    results: dict[str, dict[int, dict]] = {}

    for npz_path in sorted(log_dir.glob("*_seed*_eval/evaluations.npz")):
        parsed = _parse_eval_dir(npz_path.parent.name)
        if parsed is None:
            continue
        agent, seed = parsed

        if agent_filter and agent not in agent_filter:
            continue

        data = np.load(npz_path)
        timesteps = data["timesteps"]
        mean_rewards = np.mean(data["results"], axis=1)

        best_idx = int(np.argmax(mean_rewards))

        stats = {
            "best_reward": float(mean_rewards[best_idx]),
            "best_step": int(timesteps[best_idx]),
            "final_step": int(timesteps[-1]),
            "early_stopped": int(timesteps[-1]) < total_timesteps - eval_freq,
        }

        results.setdefault(agent, {})[seed] = stats

    return results


def print_summary(experiment_path: str, stats: dict, total_timesteps: int) -> None:
    """Print formatted training summary table."""
    print(f"\nExperiment: {experiment_path}")
    print(f"Total timesteps: {total_timesteps:_}\n")

    for agent in sorted(stats):
        seeds = stats[agent]
        print(f"Agent: {agent}")
        for seed in sorted(seeds):
            s = seeds[seed]
            stopped = "yes" if s["early_stopped"] else "no "
            print(
                f"  seed {seed:<5d}  "
                f"best_reward={s['best_reward']:>8.2f}  "
                f"best_step={s['best_step']:>10_}  "
                f"final_step={s['final_step']:>10_}  "
                f"early_stopped={stopped}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training diagnostics per agent and seed.",
    )
    parser.add_argument(
        "experiment",
        help="Path relative to experiments/ (e.g. pomdp_t3000000/default).",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Only show these agents (default: all RL agents).",
    )
    args = parser.parse_args()

    run_dir = EXPERIMENTS_DIR / args.experiment
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {run_dir}")

    with open(config_path) as f:
        config = json.load(f)

    total_timesteps = config["base_config"]["total_timesteps"]
    eval_freq = config["base_config"]["eval_freq"]

    stats = load_training_stats(run_dir, total_timesteps, eval_freq, args.agents)

    if not stats:
        print(f"No evaluations.npz found in {run_dir / 'logs'}")
        return

    print_summary(args.experiment, stats, total_timesteps)


if __name__ == "__main__":
    main()
