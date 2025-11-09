import numpy as np
import matplotlib.pyplot as plt
from src.config import DefaultConfig
import os


def create_3d_reward_surface(w_S: float):
    """
    Creates a continuous 3D surface of the reward function.
    
    :param w_S: Stringency weight value
    :return: fig, ax for saving
    """
    # X-axis: infection penalty from 0 to 3
    infection_penalty_range = np.linspace(0, 3, 100)
    
    # Y-axis: action.value from 1.0 to 0.1
    action_value_range = np.linspace(1.0, 0.1, 100)
    
    # Create meshgrid
    INFECTION_PENALTY, ACTION_VALUE = np.meshgrid(
        infection_penalty_range, 
        action_value_range
    )
    
    # Calculate reward for each point in the grid
    REWARD = np.zeros_like(INFECTION_PENALTY)
    for i in range(len(infection_penalty_range)):
        for j in range(len(action_value_range)):
            infection_penalty_val = infection_penalty_range[i]
            action_value = action_value_range[j]
            
            # Calculate stringency_penalty for this action.value
            action_stringency = (1 - action_value) ** 2
            stringency_penalty = w_S * action_stringency
            
            # Calculate reward: reward = -(infection_penalty + stringency_penalty)
            reward = -(infection_penalty_val + stringency_penalty)
            REWARD[j, i] = reward
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot continuous surface
    surf = ax.plot_surface(
        INFECTION_PENALTY, ACTION_VALUE, REWARD,
        cmap='viridis',
        alpha=0.9,
        linewidth=0,
        antialiased=True
    )
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Reward')
    
    # Configure axes with compact labels
    ax.set_xlabel('Infection Penalty\nmax(0, log((I_t1+ε)/(I_t+ε)))', fontsize=11, labelpad=8)
    ax.set_ylabel('Action Value', fontsize=11, labelpad=8)
    ax.set_zlabel('Reward', fontsize=11, labelpad=8)
    ax.set_title(f'3D Reward Function (w_S = {w_S:.2f})', fontsize=13, fontweight='bold', pad=15)
    
    # Set view angle
    ax.view_init(elev=30, azim=45)
    
    return fig, ax


if __name__ == "__main__":
    # Create output directory
    output_dir = "results/3d_reward_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over w_S values from 0 to 4 with step 0.25
    w_S_values = np.arange(0, 4.5, 0.5)
    
    print(f"Creating 3D reward surfaces for {len(w_S_values)} w_S values...")
    
    for w_S in w_S_values:
        print(f"Creating plot for w_S = {w_S:.1f}...")
        fig, ax = create_3d_reward_surface(w_S)
        
        # Save plot with formatted filename
        filename = f"{output_dir}/reward_3d_wS_{w_S:.1f}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
        
        plt.close(fig)
    
    print(f"\nAll plots saved to: {output_dir}")

