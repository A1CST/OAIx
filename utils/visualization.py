import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import cm, patches as mpatches
from constants import *


def setup_action_plot():
    """Setup the action history plot"""
    fig_action, ax_action = plt.subplots(figsize=(8, 4))
    ax_action.set_ylim(-0.5, 5.5)  # Updated to accommodate 6 actions
    ax_action.set_yticks([0, 1, 2, 3, 4, 5])
    ax_action.set_yticklabels(['sleep', 'up', 'right', 'down', 'left', 'run'])
    ax_action.set_xlabel('Time Steps')
    ax_action.set_title('Agent Action History')
    action_line, = ax_action.plot([], [], 'b-')
    action_mapping = {'sleep': 0, 'up': 1, 'right': 2, 'down': 3, 'left': 4, 'run': 5}

    return fig_action, ax_action, action_line, action_mapping


def setup_survival_plot():
    """Setup the survival time plot"""
    fig_survival, ax_survival = plt.subplots(figsize=(8, 4))
    ax_survival.set_xlabel('Game #')
    ax_survival.set_ylabel('Survival Time (hours)')
    ax_survival.set_title('Longest Survival Times')
    survival_bars = ax_survival.bar([], [], color='green')
    ax_survival.set_ylim(0, 10)  # Start with 10 hours as max, will adjust dynamically

    return fig_survival, ax_survival


def setup_stats_plot():
    """Setup the food eaten per game plot"""
    fig_stats, ax_stats = plt.subplots(figsize=(8, 4))
    ax_stats.set_xlabel('Game #')
    ax_stats.set_ylabel('Food Eaten')
    ax_stats.set_title('Food Eaten Per Game')
    ax_stats.grid(True, linestyle='--', alpha=0.7)

    return fig_stats, ax_stats


def setup_health_plot():
    """Setup health tracking plot"""
    fig_health, ax_health = plt.subplots(figsize=(8, 4))
    ax_health.set_xlabel('Time Steps')
    ax_health.set_ylabel('Value')
    ax_health.set_title('Current Game Health & Energy')
    ax_health.set_ylim(0, MAX_HEALTH)
    ax_health.grid(True, linestyle='--', alpha=0.7)
    health_line, = ax_health.plot([], [], 'r-', label='Health')
    energy_line, = ax_health.plot([], [], 'b-', label='Energy')
    ax_health.legend()

    return fig_health, ax_health, health_line, energy_line


def update_action_plot(fig_action, ax_action, action_line, agent_actions_history, action_mapping):
    """Update action history plot"""
    if agent_actions_history:
        y_data = [action_mapping[action] for action in agent_actions_history]
        x_data = list(range(len(y_data)))

        # Update plot data
        action_line.set_data(x_data, y_data)

        # Adjust x axis limits to show the most recent data
        if len(x_data) > 100:
            ax_action.set_xlim(max(0, len(x_data) - 100), len(x_data))
        else:
            ax_action.set_xlim(0, max(100, len(x_data)))

        fig_action.canvas.draw_idle()
        plt.pause(0.001)


def update_survival_plot(fig_survival, ax_survival, survival_times_history):
    """Update survival time plot"""
    if survival_times_history:
        x_data = list(range(1, len(survival_times_history) + 1))

        # Clear previous bars and create new ones
        ax_survival.clear()
        ax_survival.set_xlabel('Game #')
        ax_survival.set_ylabel('Survival Time (hours)')
        ax_survival.set_title('Longest Survival Times')

        # Convert ticks to hours for better readability
        hours_data = [t / TICKS_PER_HOUR for t in survival_times_history]

        # Adjust y-limit based on max value
        y_max = max(hours_data) * 1.2 if hours_data else 10
        ax_survival.set_ylim(0, max(10, y_max))

        # Create new bars
        ax_survival.bar(x_data, hours_data, color='green')

        fig_survival.canvas.draw_idle()
        plt.pause(0.001)


def update_stats_plot(fig_stats, ax_stats, food_eaten_per_game):
    """Update food eaten per game plot"""
    if food_eaten_per_game:
        # Clear the plot first
        ax_stats.clear()
        ax_stats.set_xlabel('Game #')
        ax_stats.set_ylabel('Food Eaten')
        ax_stats.set_title('Food Eaten Per Game')
        ax_stats.grid(True, linestyle='--', alpha=0.7)

        # Create x-axis data (game numbers)
        x_data = list(range(1, len(food_eaten_per_game) + 1))

        # Plot new bars
        ax_stats.bar(x_data, food_eaten_per_game, color='g', alpha=0.7)

        # Adjust y-axis limits with padding
        y_max = max(max(food_eaten_per_game), 1) * 1.2  # Ensure minimum y_max of 1
        ax_stats.set_ylim(0, y_max)

        # Set x-axis limits with some padding
        ax_stats.set_xlim(0, len(food_eaten_per_game) + 0.5)

        # Add value labels on top of each bar
        for i, v in enumerate(food_eaten_per_game):
            ax_stats.text(i + 1, v, str(v), ha='center', va='bottom')
    else:
        # If no data, set default limits
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)

    # Refresh the plot
    fig_stats.canvas.draw_idle()
    plt.pause(0.001)


def update_health_plot(fig_health, ax_health, current_health_history, current_time_steps, current_energy_history=None):
    """Update health tracking plot"""
    if current_health_history:
        # Clear the plot first
        ax_health.clear()
        ax_health.set_xlabel('Time Steps')
        ax_health.set_ylabel('Value')
        ax_health.set_title('Current Game Health & Energy')
        ax_health.set_ylim(0, MAX_HEALTH)
        ax_health.grid(True, linestyle='--', alpha=0.7)

        # Plot health line
        ax_health.plot(current_time_steps, current_health_history, 'r-', label='Health')
        
        # Plot energy line if available
        if current_energy_history and len(current_energy_history) == len(current_time_steps):
            ax_health.plot(current_time_steps, current_energy_history, 'b-', label='Energy')
        
        ax_health.legend()

        # Set x-axis limits to show the most recent data
        if len(current_time_steps) > 100:
            ax_health.set_xlim(max(0, len(current_time_steps) - 100), len(current_time_steps))
        else:
            ax_health.set_xlim(0, max(100, len(current_time_steps)))

        # Refresh the plot
        fig_health.canvas.draw_idle()
        plt.pause(0.001)


def draw_flowchart():
    """Draw static flowchart for the agent's cognitive pipeline"""
    fig_flow, ax_flow = plt.subplots(figsize=(12, 6))
    boxes = {
        "Inputs (Sensory Data)": (0.1, 0.6),
        "Tokenizer": (0.25, 0.6),
        "CNN / LSTM (Encoder - Pattern Recognition)": (0.4, 0.6),
        "Central LSTM (Core Pattern Processor)": (0.55, 0.6),
        "CNN / LSTM (Decoder)": (0.7, 0.6),
        "Tokenizer (Reverse)": (0.85, 0.6),
        "Actions": (0.85, 0.4),
        "New Input + Previous Actions": (0.1, 0.4)
    }
    for label, (x, y) in boxes.items():
        ax_flow.add_patch(mpatches.FancyBboxPatch(
            (x - 0.1, y - 0.05), 0.2, 0.1,
            boxstyle="round,pad=0.02", edgecolor="black", facecolor="lightgray"
        ))
        ax_flow.text(x, y, label, ha="center", va="center", fontsize=9)

    forward_flow = [
        ("Inputs (Sensory Data)", "Tokenizer"),
        ("Tokenizer", "CNN / LSTM (Encoder - Pattern Recognition)"),
        ("CNN / LSTM (Encoder - Pattern Recognition)", "Central LSTM (Core Pattern Processor)"),
        ("Central LSTM (Core Pattern Processor)", "CNN / LSTM (Decoder)"),
        ("CNN / LSTM (Decoder)", "Tokenizer (Reverse)"),
        ("Tokenizer (Reverse)", "Actions"),
        ("Actions", "New Input + Previous Actions"),
        ("New Input + Previous Actions", "Inputs (Sensory Data)")
    ]

    for start, end in forward_flow:
        x1, y1 = boxes[start]
        x2, y2 = boxes[end]
        offset1 = 0.05 if y1 > y2 else -0.05
        offset2 = -0.05 if y1 > y2 else 0.05
        ax_flow.annotate("",
                         xy=(x2, y2 + offset2),
                         xytext=(x1, y1 + offset1),
                         arrowprops=dict(arrowstyle="->", color='black'))

    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis('off')
    plt.tight_layout()
    plt.show(block=False)

    return fig_flow


def setup_heatmap():
    """Set up the heatmap figure for visualization."""
    # Create a figure with two subplots: one for the raw hidden states and one for the game grid mapping
    fig_heatmap, (ax_raw, ax_grid) = plt.subplots(1, 2, figsize=(16, 8))

    # Set up the raw hidden states heatmap
    ax_raw.set_title('LSTM Hidden States Heatmap')
    heatmap_img_raw = ax_raw.imshow(np.zeros((10, 10)), cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(heatmap_img_raw, ax=ax_raw)

    # Set up the game grid mapping heatmap
    ax_grid.set_title('Agent Exploration Heatmap')
    grid_width = WIDTH // GRID_SIZE
    grid_height = HEIGHT // GRID_SIZE
    heatmap_img_grid = ax_grid.imshow(np.zeros((grid_height, grid_width)), cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(heatmap_img_grid, ax=ax_grid)

    # Add agent position marker
    agent_marker, = ax_grid.plot([0], [0], 'ro', markersize=10, label='Agent')

    # Add food markers
    food_markers, = ax_grid.plot([], [], 'go', markersize=8, label='Food')

    # Add enemy markers
    enemy_markers, = ax_grid.plot([], [], 'bo', markersize=8, label='Enemies')

    # Add red zone
    red_zone_x = RED_ZONE_X // GRID_SIZE
    red_zone_y = RED_ZONE_Y // GRID_SIZE
    red_zone_width = RED_ZONE_WIDTH // GRID_SIZE
    red_zone_height = RED_ZONE_HEIGHT // GRID_SIZE
    red_zone = plt.Rectangle((red_zone_x, red_zone_y), red_zone_width, red_zone_height,
                             linewidth=2, edgecolor='r', facecolor='none', alpha=0.5, label='Safe Zone')
    ax_grid.add_patch(red_zone)

    ax_grid.legend()
    plt.tight_layout()

    # Initialize exploration grid
    exploration_grid = np.zeros((grid_height, grid_width))

    return fig_heatmap, (ax_raw, ax_grid), (heatmap_img_raw, heatmap_img_grid), (
    agent_marker, food_markers, enemy_markers), exploration_grid


def update_heatmap(fig_heatmap, axes, heatmap_imgs, markers, central_lstm, pattern_lstm, agent_pos, food_positions,
                   enemies, exploration_grid):
    """Update the heatmap with the current hidden states and exploration data."""
    if central_lstm.model.last_hidden_state is None or pattern_lstm.model.last_hidden_state is None:
        return exploration_grid

    ax_raw, ax_grid = axes
    heatmap_img_raw, heatmap_img_grid = heatmap_imgs
    agent_marker, food_markers, enemy_markers = markers

    # Combine hidden states from both models
    combined_states = np.concatenate([
        central_lstm.model.last_hidden_state,
        pattern_lstm.model.last_hidden_state
    ])

    # Reshape to a square grid (assuming combined length is a perfect square)
    grid_size = int(np.ceil(np.sqrt(len(combined_states))))
    padded_states = np.zeros(grid_size * grid_size)
    padded_states[:len(combined_states)] = combined_states

    # Reshape to a 2D grid for the raw heatmap
    heatmap_data_raw = padded_states.reshape(grid_size, grid_size)

    # Update the raw heatmap
    heatmap_img_raw.set_data(heatmap_data_raw)
    heatmap_img_raw.set_clim(vmin=np.min(heatmap_data_raw), vmax=np.max(heatmap_data_raw))

    # Get agent's grid position (ensure integer indices)
    agent_cell_x = int(agent_pos[0] // GRID_SIZE)
    agent_cell_y = int(agent_pos[1] // GRID_SIZE)

    # Update exploration grid
    exploration_grid[agent_cell_y, agent_cell_x] += 0.1
    exploration_grid = np.clip(exploration_grid, 0, 1)  # Clip values between 0 and 1

    # Apply gaussian blur to create a smoother heatmap
    exploration_grid_smooth = gaussian_filter(exploration_grid, sigma=1.0)

    # Update the grid heatmap with exploration data
    heatmap_img_grid.set_data(exploration_grid_smooth)
    heatmap_img_grid.set_clim(vmin=0, vmax=1)

    # Update agent position marker
    agent_marker.set_data([agent_cell_x], [agent_cell_y])

    # Update food markers
    if food_positions:
        food_x = [int(food[0] // GRID_SIZE) for food in food_positions]
        food_y = [int(food[1] // GRID_SIZE) for food in food_positions]
        food_markers.set_data(food_x, food_y)
    else:
        food_markers.set_data([], [])

    # Update enemy markers
    if enemies:
        enemy_x = [int(enemy['pos'][0] // GRID_SIZE) for enemy in enemies]
        enemy_y = [int(enemy['pos'][1] // GRID_SIZE) for enemy in enemies]
        enemy_markers.set_data(enemy_x, enemy_y)
    else:
        enemy_markers.set_data([], [])

    # Update the colorbars
    fig_heatmap.canvas.draw_idle()
    plt.pause(0.001)

    return exploration_grid