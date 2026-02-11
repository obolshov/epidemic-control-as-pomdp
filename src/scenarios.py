"""
Predefined experiment scenarios for reproducibility.

This module defines standard POMDP scenarios and utilities for creating custom scenarios.
Each scenario specifies POMDP parameters. Target agents are defined globally.
"""

from typing import Dict, Any, List


TARGET_AGENTS = [
    "random",
    "threshold",
    "ppo_baseline",
    "ppo_framestack",
    "ppo_recurrent",
]


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
    # Future experiments can be added here:
    # "noisy_observations": {
    #     "description": "POMDP Experiment 2: Observations with Gaussian noise",
    #     "pomdp_params": {
    #         "include_exposed": True,
    #         "noise_std": 0.1,
    #     },
    # },
    # "delayed_observations": {
    #     "description": "POMDP Experiment 3: Delayed observations",
    #     "pomdp_params": {
    #         "include_exposed": True,
    #         "delay": 5,
    #     },
    # },
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


def create_custom_scenario_name(pomdp_params: Dict[str, Any]) -> str:
    """
    Generate a descriptive name for a custom scenario based on POMDP parameters.
    
    This creates human-readable names by concatenating non-default parameter values.
    Example: {"include_exposed": False, "delay": 5} -> "no_exposed_delay5"
    
    Args:
        pomdp_params: Dictionary of POMDP parameters.
        
    Returns:
        Generated scenario name string.
    """
    parts = []
    
    # Check each possible parameter and add to name if non-default
    if not pomdp_params.get("include_exposed", True):
        parts.append("no_exposed")
    
    if "delay" in pomdp_params and pomdp_params["delay"] > 0:
        parts.append(f"delay{pomdp_params['delay']}")
    
    if "noise_std" in pomdp_params and pomdp_params["noise_std"] > 0:
        noise_val = pomdp_params["noise_std"]
        parts.append(f"noise{noise_val}")
    
    # Add more parameters as needed in the future
    # if "frame_stack" in pomdp_params and pomdp_params["frame_stack"] > 1:
    #     parts.append(f"framestack{pomdp_params['frame_stack']}")
    
    if not parts:
        # All parameters are default, use generic name
        return "custom"
    
    return "_".join(parts)


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
