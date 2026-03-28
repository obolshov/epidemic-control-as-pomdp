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
        ("ent", lambda cfg: cfg.ent_coef, 0.01),
    ],
    "ppo_framestack": [
        ("nstack", lambda cfg: cfg.n_stack, 10),
        ("ent", lambda cfg: cfg.ent_coef, 0.01),
    ],
    "ppo_recurrent": [
        ("lstm", lambda cfg: cfg.lstm_hidden_size, 32),
        ("ent", lambda cfg: cfg.recurrent_ent_coef, 0.05),
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
    "no_exposed": {
        "description": "POMDP Experiment 1: Masked E compartment",
        "pomdp_params": {
            "include_exposed": False,
        },
    },
    "underreporting": {
        "description": "POMDP Experiment 2: Masked E + Under-reporting (detection_rate=0.3) + testing saturation (1.5%/day)",
        "pomdp_params": {
            "include_exposed": False,
            "detection_rate": 0.3,
            "testing_capacity": 0.015,
        },
    },
    "noisy_pomdp": {
        "description": "POMDP Experiment 3: Masked E + under-reporting (k=0.3) + testing saturation (1.5%/day) + multiplicative noise",
        "pomdp_params": {
            "include_exposed": False,
            "detection_rate": 0.3,
            "testing_capacity": 0.015,
            "noise_stds": [0.05, 0.30, 0.15],
            "noise_rho": 0.7,
        },
    },
    "pomdp": {
        "description": "POMDP Experiment 4: Masked E + under-reporting (k=0.3) + testing saturation (1.5%/day) + AR(1) noise (ρ=0.7) + temporal lag (5–14 days) + action delay (5 days)",
        "pomdp_params": {
            "include_exposed": False,
            "detection_rate": 0.3,
            "testing_capacity": 0.015,
            "noise_stds": [0.05, 0.30, 0.15],
            "noise_rho": 0.7,
            "lag": [5, 14],
            "action_delay": 5,
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


def create_custom_scenario_name(
    pomdp_params: Dict[str, Any],
    total_timesteps: int = 0,
    deterministic: bool = False,
) -> str:
    """
    Generate a descriptive name for a custom scenario based on POMDP parameters.

    This creates human-readable names by concatenating non-default parameter values.
    Agent-level hyperparameters (n_stack, lstm_hidden_size, ent_coef) are NOT
    encoded here — they are encoded in agent names via ``get_agent_variant_name()``.

    Example: {"include_exposed": False, "detection_rate": 0.3}, total_timesteps=10000
        -> "custom_no_exposed_k0.3_t10000"

    Args:
        pomdp_params: Dictionary of POMDP parameters.
        total_timesteps: RL training budget. Appended as ``_t{n}`` when non-zero.
        deterministic: If True, appends ``_det`` suffix.

    Returns:
        Generated scenario name string.
    """
    parts = []

    # Check each possible parameter and add to name if non-default
    if not pomdp_params.get("include_exposed", True):
        parts.append("no_exposed")

    if pomdp_params.get("action_delay") and pomdp_params["action_delay"] > 0:
        parts.append(f"adelay{pomdp_params['action_delay']}")

    if "detection_rate" in pomdp_params and pomdp_params["detection_rate"] < 1.0:
        k = pomdp_params["detection_rate"]
        parts.append(f"k{k:.2g}")

    if pomdp_params.get("noise_stds") and any(s > 0 for s in pomdp_params["noise_stds"]):
        stds = pomdp_params["noise_stds"]
        parts.append(f"noise{'_'.join(f'{s:.2g}' for s in stds)}")

    if pomdp_params.get("noise_rho", 0.0) > 0:
        parts.append(f"rho{pomdp_params['noise_rho']:.2g}")

    if pomdp_params.get("testing_capacity"):
        parts.append(f"cap{pomdp_params['testing_capacity']}")

    if pomdp_params.get("lag"):
        min_lag, max_lag = pomdp_params["lag"]
        parts.append(f"lag{min_lag}_{max_lag}")

    base = "custom_" + "_".join(parts) if parts else "custom"

    if deterministic:
        base += "_det"
    if total_timesteps:
        base += f"_t{total_timesteps}"

    return base


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
