import os
import random
from constants import *


def ensure_directories():
    """Ensure that necessary directories exist"""
    directories = ["checkpoints", "data"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def initialize_game_state():
    """Initialize game state variables"""
    game_state = {
        # Game clock
        "game_ticks": 0,
        "game_hour": 6,  # Start at 6 AM
        "game_day": 1,  # Start at day 1

        # Game statistics
        "current_game_time": 0,  # Current game time in ticks
        "longest_game_time": 0,  # Longest game time in ticks
        "death_count": 0,  # Number of deaths
        "food_eaten": 0,  # Food eaten in current life
        "total_health_lost": 0,  # Health lost in current life

        # History tracking for plots
        "agent_actions_history": [],  # Action history for current life
        "survival_times_history": [],  # Survival times across games
        "food_eaten_history": [],  # Track food eaten over time
        "health_lost_history": [],  # Track health lost over time
        "food_eaten_per_game": [],  # Food eaten per game
        "current_health_history": [],  # Health over time in current game
        "current_energy_history": [],  # Energy over time in current game
        "current_time_steps": [],  # Time points for current game
        "time_points": []  # Time points for food/health chart
    }

    return game_state


def reset_agent_state(agent_module):
    """Reset agent state after death"""
    agent_module.health = MAX_HEALTH
    agent_module.energy = INITIAL_ENERGY
    agent_module.regen_timer = 0
    agent_module.digestion = 0.0
    agent_module.starvation_timer = 0
    agent_module.damage_cooldown = 0
    agent_module.agent_actions_history = []

    # Re-initialize agent position
    agent_module.agent_pos = [
        random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
        random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE
    ]
    agent_module.agent_direction = random.randint(0, 3)


def log_death(game_state, cause="unknown", ticks_per_hour=TICKS_PER_HOUR):
    """Log death information to file"""
    # Create the log file if it doesn't exist
    if not os.path.exists("data/survival_log.csv"):
        with open("data/survival_log.csv", "w") as f:
            f.write("day,ticks,food,death_cause\n")

    # Log the death
    with open("data/survival_log.csv", "a") as f:
        f.write(f"{game_state['game_day']},{game_state['current_game_time']},{game_state['food_eaten']},{cause}\n")

    # Print death information
    hours_alive = game_state['current_game_time'] / ticks_per_hour
    longest_hours = game_state['longest_game_time'] / ticks_per_hour
    print(f"Last game survival time: {hours_alive:.2f} hours ({game_state['current_game_time']} ticks)")
    print(f"Highest survival time: {longest_hours:.2f} hours ({game_state['longest_game_time']} ticks)")

    # Store survival time
    game_state['survival_times_history'].append(game_state['current_game_time'])
    game_state['longest_game_time'] = max(game_state['longest_game_time'], game_state['current_game_time'])

    # Store food eaten for this game
    game_state['food_eaten_per_game'].append(game_state['food_eaten'])


def handle_death(game_state, agent_module, pattern_lstm, central_lstm, ticks_per_hour=TICKS_PER_HOUR):
    """Handle agent death including model checkpointing"""
    # Determine cause of death
    death_cause = "starvation" if agent_module.starvation_timer >= STARVATION_TIME else "enemy"

    # Log the death
    log_death(game_state, death_cause, ticks_per_hour)

    # —— NEW: log current novelty & token count
    import game.pipeline as pipeline_mod
    token_count = len(pipeline_mod.token_sequence_buffer)
    if central_lstm.prediction_errors:
        avg_novelty = sum(central_lstm.prediction_errors) / len(central_lstm.prediction_errors)
    else:
        avg_novelty = 0.0
    print(f"[DEATH] Avg Novelty: {avg_novelty:.4f}, Token Buffer Length: {token_count}")

    # —— NEW: clear novelty history & token buffer
    pipeline_mod.token_sequence_buffer.clear()
    central_lstm.prediction_errors.clear()
    print("[RESET] Cleared prediction_errors and token_sequence_buffer")

    # Increment death counter
    game_state['death_count'] += 1

    # Save checkpoint every CHECKPOINT_INTERVAL deaths
    if game_state['death_count'] % CHECKPOINT_INTERVAL == 0:
        print(f"Saving checkpoint after {game_state['death_count']} deaths...")
        pattern_lstm.save_checkpoint(PATTERN_CKPT)
        central_lstm.save_checkpoint(CENTRAL_CKPT)

    # Reset agent state
    reset_agent_state(agent_module)

    # Reset food positions
    from game.environment import initialize_food
    game_state['food_positions'] = initialize_food(FOOD_COUNT)

    # Reset game state for new life
    game_state['food_eaten'] = 0
    game_state['total_health_lost'] = 0
    game_state['agent_actions_history'] = []
    game_state['current_health_history'] = []
    game_state['current_energy_history'] = []
    game_state['current_time_steps'] = []

    # Reset survival timers
    game_state['current_game_time'] = 0
    game_state['game_ticks'] = 0
    game_state['game_hour'] = DAY_START_HOUR

    return game_state


def update_game_clock(game_state, ticks_per_hour=TICKS_PER_HOUR):
    """Update the game clock (time of day and day counter)"""
    game_state['game_ticks'] += 1
    game_state['current_game_time'] += 1  # Increment current game time

    # Update game hour every ticks_per_hour
    if game_state['game_ticks'] >= ticks_per_hour:
        game_state['game_ticks'] = 0
        game_state['game_hour'] = (game_state['game_hour'] + 1) % HOURS_PER_DAY

        # Increment day counter when we reach midnight (hour 0)
        if game_state['game_hour'] == 0:
            game_state['game_day'] += 1
            print(f"Day {game_state['game_day']} has begun")

    return game_state


def update_agent_state(action, reward, done, info):
    """Update the agent's state based on the action taken and the reward received."""
    global game_state
    
    # Update health
    game_state['current_health'] = info['health']
    
    # Update energy
    game_state['current_energy'] = info['energy']
    
    # Update food eaten
    if info.get('food_eaten', False):
        game_state['food_eaten'] += 1
        game_state['food_eaten_history'].append(game_state['food_eaten'])
        game_state['time_points'].append(game_state['current_game_time'])
    
    # Update death count if agent died
    if done:
        game_state['death_count'] += 1
        game_state['survival_times_history'].append(game_state['current_game_time'])
        game_state['food_eaten_per_game'].append(game_state['food_eaten'])
        
        # Log death to survival log
        with open('data/survival_log.csv', 'a') as f:
            f.write(f"{game_state['game_day']},{game_state['current_game_time']},"
                    f"{game_state['food_eaten']},{info.get('death_reason','unknown')}\n")
        
        # Reset game state for new game
        reset_game_state(game_state)
        game_state['current_game_time'] = 0  # ⬅️ Reset survival time after death
        print("[RESET] current_game_time reset to 0 after death.")
    
    return game_state
