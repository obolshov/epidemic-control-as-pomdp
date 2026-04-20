"""
Predefined experiment scenarios for reproducibility.

This module defines standard POMDP scenarios and utilities for creating custom scenarios.
Each scenario specifies POMDP parameters. Target agents are defined globally.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config import Config


TARGET_AGENTS = [
    "no_action",
    "severe",
    "random",
    "threshold",
    "ppo_baseline",
    "ppo_framestack",
    "ppo_recurrent",
]

# Maps base agent name → list of (suffix_key, config_getter, default_value).
# Add entries here to encode new agent-level hyperparameters in the agent name.
# Non-default values are appended as "_key{value}" suffixes, allowing multiple
# variants to coexist in the same scenario's weights directory.
AGENT_VARIANT_PARAMS: Dict[str, List[Tuple[str, Callable[["Config"], Any], Any]]] = {
    "ppo_baseline": [
        ("ent", lambda cfg: cfg.ent_coef, 0.2),
    ],
    "ppo_framestack": [
        ("nstack", lambda cfg: cfg.n_stack, 20),
        ("ent", lambda cfg: cfg.ent_coef, 0.2),
    ],
    "ppo_recurrent": [
        ("lstm", lambda cfg: cfg.lstm_hidden_size, 32),
        ("ent", lambda cfg: cfg.ent_coef, 0.2),
        ("nsteps", lambda cfg: cfg.recurrent_n_steps, 256),
    ],
}


def get_agent_variant_name(agent_name: str, config: "Config") -> str:
    """Return agent name with non-default hyperparameter suffixes appended.

    Agent-level hyperparameters (n_stack, lstm_hidden_size, ent_coef) are encoded
    in the agent name rather than the scenario folder name, so multiple variants
    can coexist in the same weights directory.

    Args:
        agent_name: Base agent name (e.g. ``"ppo_framestack"``).
        config: Experiment config with current hyperparameter values.

    Returns:
        Agent name with suffixes for non-default values.
        Examples: ``"ppo_framestack_nstack5"``, ``"ppo_recurrent_lstm64"``,
        ``"ppo_baseline_ent0.05"``.
    """
    params = AGENT_VARIANT_PARAMS.get(agent_name, [])
    parts = []
    for key, getter, default in params:
        value = getter(config)
        if value != default:
            parts.append(f"{key}{value:.4g}" if isinstance(value, float) else f"{key}{value}")
    return (agent_name + "_" + "_".join(parts)) if parts else agent_name


PREDEFINED_SCENARIOS = {
    "mdp": {
        "description": "Baseline MDP (full observability)",
        "pomdp_params": {
            "include_exposed": True,
        },
    },
    "incompleteness": {
        "description": "POMDP Experiment 1: Incomplete surveillance — masked E + per-episode stochastic under-reporting (detection_rate ~ U[0.15, 0.40]) with testing-capacity saturation (~ U[0.5%, 2%]/day)",
        "pomdp_params": {
            "include_exposed": False,
            "detection_rate": (0.15, 0.40),
            "testing_capacity": (0.005, 0.02),
        },
    },
    "incompleteness_and_noise": {
        "description": "POMDP Experiment 2: Stochastic incomplete surveillance + AR(1) autocorrelated multiplicative noise (ρ=0.7)",
        "pomdp_params": {
            "include_exposed": False,
            "detection_rate": (0.15, 0.40),
            "testing_capacity": (0.005, 0.02),
            "noise_stds": [0.05, 0.30, 0.15],
            "noise_rho": 0.7,
        },
    },
    "pomdp": {
        "description": "POMDP Experiment 3: Stochastic incomplete surveillance + AR(1) noise (ρ=0.7) + temporal lag (5–14 days)",
        "pomdp_params": {
            "include_exposed": False,
            "detection_rate": (0.15, 0.40),
            "testing_capacity": (0.005, 0.02),
            "noise_stds": [0.05, 0.30, 0.15],
            "noise_rho": 0.7,
            "lag": [5, 14],
        },
    },
    # --- Isolated distortion ablation scenarios ---
    # Each applies ONE distortion group to the base MDP.
    "only_noise": {
        "description": "Ablation: observation noise only (E visible, AR(1) noise ρ=0.7)",
        "pomdp_params": {
            "include_exposed": True,
            "noise_stds": [0.05, 0.30, 0.30, 0.15],  # [S, E, I, R]; E=0.30 like I (pre-symptomatic, rarely detected directly)
            "noise_rho": 0.7,
        },
    },
    "only_temporal": {
        "description": "Ablation: temporal distortion only (E visible, lag 5–14 days)",
        "pomdp_params": {
            "include_exposed": True,
            "lag": [5, 14],
        },
    },
}


def get_scenario(name: str) -> Dict[str, Any]:
    """
    Get configuration for a predefined scenario.
    
    Args:
        name: Scenario name (e.g., "mdp", "no_exposed").
        
    Returns:
        Dictionary with scenario configuration including target agents.
        
    Raises:
        ValueError: If scenario name is not recognized.
    """
    if name not in PREDEFINED_SCENARIOS:
        available = ", ".join(PREDEFINED_SCENARIOS.keys())
        raise ValueError(
            f"Unknown scenario: '{name}'. Available scenarios: {available}"
        )
    
    scenario = PREDEFINED_SCENARIOS[name].copy()
    scenario["target_agents"] = TARGET_AGENTS.copy()
    return scenario


def list_scenarios() -> List[str]:
    """
    Get list of all predefined scenario names.
    
    Returns:
        List of scenario name strings.
    """
    return list(PREDEFINED_SCENARIOS.keys())


def get_scenario_description(name: str) -> str:
    """
    Get human-readable description of a scenario.
    
    Args:
        name: Scenario name.
        
    Returns:
        Description string.
        
    Raises:
        ValueError: If scenario name is not recognized.
    """
    if name not in PREDEFINED_SCENARIOS:
        raise ValueError(f"Unknown scenario: '{name}'")
    
    return PREDEFINED_SCENARIOS[name]["description"]
