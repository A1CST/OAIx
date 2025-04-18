#!/usr/bin/env python3
"""
Main entry point for the agent simulation with LSTMs.
Parses command-line arguments and runs the simulation.
"""

import os
# Set environment variable to fix OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import pygame
import matplotlib.pyplot as plt
import random
import torch
import atexit
import time

# Import constants
from constants import *

# Import LSTM models
from models.lstm_model import LSTMModel, RealLSTM

# Import game modules
from game.environment import (initialize_food, initialize_enemies, update_enemies,
                              is_in_red_zone, get_vision_data, get_hearing_data,
                              get_background_color)
from game.agent import (initialize_agent, process_move, get_sensory_data,
                        update_agent_state, check_food_collision, agent_pos,
                        agent_direction, agent_action, agent_actions_history)
from game.pipeline import pipeline, tokenize, encode, decode, reverse_tokenizer
from game.rendering import (setup_pygame, draw_agent, draw_food, draw_enemies,
                            draw_red_zone, draw_vision_cells, draw_stats_panel,
                            draw_sensory_panel)

# Import utility modules
from utils.visualization import (setup_action_plot, setup_survival_plot, setup_stats_plot,
                                 setup_health_plot, update_action_plot, update_survival_plot,
                                 update_stats_plot, update_health_plot, draw_flowchart,
                                 setup_heatmap, update_heatmap)
from utils.helpers import (ensure_directories, initialize_game_state, reset_agent_state,
                           log_death, handle_death, update_game_clock)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run the agent simulation')
    parser.add_argument('--render', action='store_true', help='Render the Pygame window')
    parser.add_argument('--actions', action='store_true', help='Only show the actions chart')
    parser.add_argument('--speed', type=int, default=1, help='Speed multiplier for headless mode (default: 1)')
    parser.add_argument('--enemies', action='store_true', help='Enable enemies in the game (enabled by default)')
    parser.add_argument('--maximize', action='store_true', help='Render the game in full screen mode')
    parser.add_argument('--heatmap', action='store_true', help='Show realtime heatmap of hidden states')
    parser.add_argument('--fast', action='store_true', help='Run in ultra-fast mode (no rendering, high speed)')
    parser.add_argument('--model', type=str, default='ppo', help='Model type (ppo or dqn)')
    return parser.parse_args()


def setup_game(args):
    """Setup game environment, LSTMs, and visualization"""
    # Set constants based on arguments
    if args.fast:
        # Ultra-fast mode: no rendering, maximum speed
        fps = 10000  # Very high FPS in ultra-fast mode
        ticks_per_hour = 10  # Minimal ticks per hour in ultra-fast mode
        print("Running in ULTRA-FAST mode (no rendering, maximum speed)")
    else:
        # Normal mode: adjust based on render flag
        fps = FPS if args.render else 1000 * args.speed  # Higher FPS in headless mode
        ticks_per_hour = TICKS_PER_HOUR if args.render else 60  # Reduced ticks per hour in headless
    
    enemy_count = ENEMY_COUNT if args.enemies else 0  # Number of enemies

    # Ensure directories exist
    ensure_directories()

    # Create LSTM instances and load checkpoints
    pattern_lstm = RealLSTM("Pattern LSTM", hidden_size=HIDDEN_SIZE, output_size=6, learning_rate=LEARNING_RATE)
    central_lstm = RealLSTM("Central LSTM", hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, learning_rate=LEARNING_RATE)
    pattern_lstm.load_checkpoint(PATTERN_CKPT)
    central_lstm.load_checkpoint(CENTRAL_CKPT)
    atexit.register(lambda: pattern_lstm.save_checkpoint(PATTERN_CKPT))
    atexit.register(lambda: central_lstm.save_checkpoint(CENTRAL_CKPT))

    # Initialize pygame
    screen, game_surface, is_maximized, scale_info = setup_pygame(args)
    clock = pygame.time.Clock()

    # Initialize agent
    initialize_agent()

    # Initialize game state
    game_state = initialize_game_state()

    # Initialize food and enemies
    food_positions = initialize_food(FOOD_COUNT)
    enemies = initialize_enemies(enemy_count)

    # Prepare font for HUD elements
    font = pygame.font.Font(None, 18)

    # Setup visualization plots
    plot_data = {}
    if not args.maximize or args.actions:  # Only set up plots if not in maximize-only mode
        plot_data['action_plot'] = setup_action_plot()

        # Set up survival and stats plots (only if not in actions-only mode)
        if not args.actions:
            plot_data['survival_plot'] = setup_survival_plot()
            plot_data['stats_plot'] = setup_stats_plot()

            # Set up health tracking plot (only in headless mode and not in maximize mode)
            if not args.render and not args.maximize:
                plot_data['health_plot'] = setup_health_plot()

    # Draw the static flowchart before the game starts (only if not in maximize-only mode or actions-only mode)
    if not args.maximize and not args.actions:
        plot_data['flowchart'] = draw_flowchart()

    # Initialize heatmap visualization if enabled
    if args.heatmap:
        plot_data['heatmap'] = setup_heatmap()
        plot_data['exploration_grid'] = plot_data['heatmap'][4]

    return {
        'pattern_lstm': pattern_lstm,
        'central_lstm': central_lstm,
        'screen': screen,
        'game_surface': game_surface,
        'is_maximized': is_maximized,
        'scale_info': scale_info,
        'clock': clock,
        'fps': fps,
        'font': font,
        'food_positions': food_positions,
        'enemies': enemies,
        'game_state': game_state,
        'plot_data': plot_data,
        'ticks_per_hour': ticks_per_hour
    }


def process_events():
    """Process pygame events"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            # Add escape key to exit full screen mode
            if event.key == pygame.K_ESCAPE:
                return False
    return True


def update_game(game_setup, args):
    """Update game state for one frame"""
    # Import agent variables at the beginning of the function
    from game.agent import agent_pos, agent_direction, agent_action, agent_actions_history
    
    pattern_lstm = game_setup['pattern_lstm']
    central_lstm = game_setup['central_lstm']
    screen = game_setup['screen']
    game_surface = game_setup['game_surface']
    is_maximized = game_setup['is_maximized']
    scale_info = game_setup['scale_info']
    font = game_setup['font']
    food_positions = game_setup['food_positions']
    enemies = game_setup['enemies']
    game_state = game_setup['game_state']
    plot_data = game_setup['plot_data']
    ticks_per_hour = game_setup['ticks_per_hour']

    # Update game clock with the appropriate ticks per hour
    game_state = update_game_clock(game_state, ticks_per_hour)

    # Print exploration rate and average novelty every 5 seconds (50 ticks at FPS=10)
    if game_state['game_ticks'] % 50 == 0:
        # Calculate average novelty from prediction errors
        avg_novelty = 0.0
        if central_lstm.prediction_errors:
            avg_novelty = sum(central_lstm.prediction_errors) / len(central_lstm.prediction_errors)
        print(f"Exploration Rate: {central_lstm.exploration_rate:.4f}, Average Novelty: {avg_novelty:.4f}")

        # Update heatmap visualization if enabled
        if 'heatmap' in plot_data:
            plot_data['exploration_grid'] = update_heatmap(
                plot_data['heatmap'][0],
                plot_data['heatmap'][1],
                plot_data['heatmap'][2],
                plot_data['heatmap'][3],
                central_lstm,
                pattern_lstm,
                agent_pos,
                food_positions,
                enemies,
                plot_data['exploration_grid']
            )

    # Get background color based on time of day
    bg_color = get_background_color(game_state['game_hour'])
    screen.fill(bg_color)
    if game_surface:
        game_surface.fill(bg_color)

    # Get vision data
    vision_cells, vision_range = get_vision_data(
        agent_pos,
        agent_direction,
        enemies,
        food_positions,
        game_state['game_hour']
    )

    # Get sensory data
    sensory_states, sensory_data = get_sensory_data(
        agent_pos,
        vision_cells,
        enemies,
        food_positions,
        game_state['game_hour']
    )

    # Process through the pipeline; central LSTM will output a valid command
    move = pipeline(sensory_data, pattern_lstm, central_lstm)

    # Convert move to actual position and get action type
    new_pos, action, direction = process_move(move, agent_pos)

    # Update global agent variables (making them accessible to other modules)
    pixels_moved = abs(new_pos[0] - agent_pos[0]) + abs(new_pos[1] - agent_pos[1])
    agent_pos[0], agent_pos[1] = new_pos
    agent_direction = direction
    agent_action = action
    agent_actions_history.append(action)

    # Check for food collision
    food_eaten, game_setup['food_positions'] = check_food_collision(agent_pos, food_positions)
    if food_eaten:
        game_state['food_eaten'] += 1

    # Update enemies
    update_enemies(enemies)

    # Update agent's internal state
    is_running = action == "run"
    agent_died = update_agent_state(game_state['food_eaten'], pixels_moved, is_running, enemies)

    # Check for death
    if agent_died:
        # Import the game.agent module correctly
        import game.agent as agent_module
        game_state = handle_death(game_state, agent_module, pattern_lstm, central_lstm, ticks_per_hour)

    # Update action plot
    if 'action_plot' in plot_data and not (is_maximized and not args.actions):
        update_action_plot(
            plot_data['action_plot'][0],
            plot_data['action_plot'][1],
            plot_data['action_plot'][2],
            agent_actions_history,
            plot_data['action_plot'][3]
        )

    # Update survival plot
    if 'survival_plot' in plot_data and not (is_maximized or args.actions):
        update_survival_plot(
            plot_data['survival_plot'][0],
            plot_data['survival_plot'][1],
            game_state['survival_times_history']
        )

    # Update stats plot
    if 'stats_plot' in plot_data and not (is_maximized or args.actions):
        update_stats_plot(
            plot_data['stats_plot'][0],
            plot_data['stats_plot'][1],
            game_state['food_eaten_per_game']
        )

    # Update health tracking data and plot
    if 'health_plot' in plot_data and not (is_maximized or args.actions):
        from game.agent import health, energy
        game_state['current_health_history'].append(health)
        game_state['current_energy_history'].append(energy)
        game_state['current_time_steps'].append(game_state['current_game_time'])
        update_health_plot(
            plot_data['health_plot'][0],
            plot_data['health_plot'][1],
            game_state['current_health_history'],
            game_state['current_time_steps'],
            game_state['current_energy_history']
        )

    # Rendering (if enabled)
    surface_to_draw = game_surface if is_maximized else screen

    if args.render or is_maximized:
        # Draw game elements
        draw_food(surface_to_draw, food_positions)
        draw_enemies(surface_to_draw, enemies)
        draw_agent(surface_to_draw, agent_pos, agent_direction)
        draw_vision_cells(surface_to_draw, vision_cells, agent_pos, agent_direction, vision_range)
        
        # Import agent state variables for stats panel
        from game.agent import health, energy, digestion, starvation_timer
        
        draw_stats_panel(
            surface_to_draw,
            font,
            health,
            energy,
            digestion,
            game_state['death_count'],
            game_state['food_eaten'],
            agent_action,
            starvation_timer,
            game_state['game_hour'],
            game_state['game_day']
        )
        draw_sensory_panel(
            surface_to_draw,
            font,
            sensory_states,
            health,
            digestion,
            vision_cells,
            sensory_data['hearing']
        )
        draw_red_zone(surface_to_draw, STATS_PANEL_WIDTH, font)

        # If maximized, scale and blit the game surface to the screen
        if is_maximized:
            scaled_width, scaled_height, pos_x, pos_y = scale_info
            scaled_surface = pygame.transform.scale(game_surface, (scaled_width, scaled_height))
            screen.blit(scaled_surface, (pos_x, pos_y))

        # Update the display
        pygame.display.flip()

    game_setup['game_state'] = game_state
    return game_setup


def main():
    """Main entry point for the simulation"""
    args = parse_args()

    # Setup game
    game_setup = setup_game(args)

    # Main game loop
    running = True
    while running:
        running = process_events()
        if not running:
            break

        # Update game state
        game_setup = update_game(game_setup, args)

        # Control game speed
        game_setup['clock'].tick(game_setup['fps'])

    # Cleanup
    pygame.quit()
    plt.ioff()
    plt.close('all')
    print("Game terminated gracefully.")


if __name__ == "__main__":
    main()