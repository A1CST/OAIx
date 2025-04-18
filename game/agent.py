import random
from constants import *

# Global agent state variables
agent_pos = [0, 0]
agent_direction = DIRECTION_UP  # 0=up, 1=right, 2=down, 3=left
agent_direction_vectors = AGENT_DIRECTION_VECTORS  # Up, Right, Down, Left
agent_action = "sleep"  # Initial action state
agent_actions_history = []  # Store action history for plotting
last_position = [0, 0]  # Store the last position to track actual movement
is_exhausted = False  # Flag for when agent passes out from exhaustion

# Agent state parameters
health = MAX_HEALTH
digestion = 0.0  # Start with no digestion
energy = INITIAL_ENERGY  # Start with initial energy
regen_timer = 0
starvation_timer = 0  # Track how long agent has been at 0% digestion
damage_cooldown = 0  # Track frames since last damage taken


def initialize_agent():
    """Initialize agent with random position and direction"""
    global agent_pos, agent_direction, agent_action, health, digestion, energy
    global regen_timer, starvation_timer, damage_cooldown, agent_actions_history, last_position
    global is_exhausted

    agent_pos = [
        random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
        random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE
    ]
    last_position = agent_pos.copy()  # Initialize last_position
    agent_direction = random.randint(0, 3)
    agent_action = "sleep"
    agent_actions_history = []
    health = MAX_HEALTH
    digestion = 0.0
    energy = INITIAL_ENERGY
    regen_timer = 0
    starvation_timer = 0
    damage_cooldown = 0
    is_exhausted = False  # Ensure agent starts not exhausted
    
    print(f"Agent initialized: Energy={energy}, Exhausted={is_exhausted}")


def process_move(move, agent_pos):
    """Process agent movement with collision detection"""
    global energy, agent_action, last_position, is_exhausted, digestion
    
    # Store current position to calculate actual movement later
    last_position = agent_pos.copy()
    
    # Debug output to track state
    if move[0] != 0 or move[1] != 0:
        print(f"Processing move: {move}, Energy={energy:.1f}, Exhausted={is_exhausted}, Digestion={digestion:.1f}")
    
    # Check if agent is exhausted
    if is_exhausted:
        if energy >= 20:
            # Agent has recovered enough energy to wake up
            is_exhausted = False
            print(f"Agent recovered from exhaustion, energy: {energy:.1f}%")
        else:
            agent_action = "sleep"
            return agent_pos, "sleep", agent_direction
    
    # Force rest when digestion > 10% and energy < 20%
    if digestion > 10 and energy < 20 and not is_exhausted:
        agent_action = "sleep"
        print(f"Agent is resting to recover energy: {energy:.1f}% (with digestion: {digestion:.1f}%)")
        return agent_pos, "sleep", agent_direction
    
    # Check if agent has enough energy to move
    if energy <= 0:
        # Agent passes out from exhaustion
        is_exhausted = True
        agent_action = "sleep"
        print("Agent passed out from exhaustion (0% energy)")
        return agent_pos, "sleep", agent_direction

    # Set initial values
    dx, dy = move
    is_running = False
    
    # Check if the move is in the current direction (running)
    if move == agent_direction_vectors[agent_direction] and abs(dx) + abs(dy) > GRID_SIZE:
        is_running = True
    
    # Check if agent has enough energy for the move
    energy_needed = ENERGY_COST_PER_BLOCK_RUNNING if is_running else ENERGY_COST_PER_BLOCK
    
    # Significantly increased energy cost to fix the balance issue
    energy_needed *= 5.0  # Multiplier to make movement more costly
    
    if energy < energy_needed:
        # Not enough energy for this move, force sleep
        agent_action = "sleep"
        print(f"Not enough energy for move: {energy:.1f}% < {energy_needed:.1f}% needed")
        return agent_pos, "sleep", agent_direction

    # Calculate potential new position
    new_pos_x = agent_pos[0] + move[0]
    new_pos_y = agent_pos[1] + move[1]

    # Update agent position with wall collision
    # Restrict movement at walls
    if new_pos_x < 0:
        new_pos_x = 0
    elif new_pos_x >= WIDTH:
        new_pos_x = WIDTH - GRID_SIZE

    if new_pos_y < 0:
        new_pos_y = 0
    elif new_pos_y >= HEIGHT:
        new_pos_y = HEIGHT - GRID_SIZE

    # Return processed movement and determine action
    if move[0] < 0:
        action = "left"
        direction = DIRECTION_LEFT
    elif move[0] > 0:
        action = "right"
        direction = DIRECTION_RIGHT
    elif move[1] < 0:
        action = "up"
        direction = DIRECTION_UP
    elif move[1] > 0:
        action = "down"
        direction = DIRECTION_DOWN
    else:
        action = "sleep"
        direction = agent_direction  # Keep current direction

    # If we're running, update the action
    if is_running:
        action = "run"

    return (new_pos_x, new_pos_y), action, direction


def get_sensory_data(agent_pos, vision_cells, enemies, food_positions, game_hour):
    """Gather all sensory data for the agent"""
    # Determine "smell" signal: if any food is within 1 grid cell, set to true.
    agent_cell = (agent_pos[0] // GRID_SIZE, agent_pos[1] // GRID_SIZE)
    smell_flag = any(
        abs(agent_cell[0] - (food[0] // GRID_SIZE)) <= 1 and
        abs(agent_cell[1] - (food[1] // GRID_SIZE)) <= 1
        for food in food_positions
    )

    # Determine "touch" signal: if agent is at the edge of the grid
    touch_flag = (agent_pos[0] == 0 or agent_pos[0] == WIDTH - GRID_SIZE or
                  agent_pos[1] == 0 or agent_pos[1] == HEIGHT - GRID_SIZE)

    # Process vision data
    vision_value = "none"
    if vision_cells:
        for cell in vision_cells:
            if "threat-food-wall" in cell:
                vision_value = "threat-food-wall"
                break
            elif "threat-wall" in cell and vision_value not in ["threat-food-wall"]:
                vision_value = "threat-wall"
                break
            elif "threat" in cell and vision_value not in ["threat-food-wall", "threat-wall"]:
                vision_value = "threat"
            elif "food-wall" in cell and vision_value not in ["threat-food-wall", "threat-wall", "threat"]:
                vision_value = "food-wall"
            elif "food" in cell and vision_value not in ["threat-food-wall", "threat-wall", "threat", "food-wall"]:
                vision_value = "food"
            elif "wall" in cell and vision_value not in ["threat-food-wall", "threat-wall", "threat", "food-wall",
                                                         "food"]:
                vision_value = "wall"

    # Get hearing data
    from game.environment import get_hearing_data
    hearing_value = get_hearing_data(agent_pos, enemies)

    # Gather sensory data
    sensory_data = {
        "smell": "true" if smell_flag else "false",
        "touch": "true" if touch_flag else "false",
        "vision": vision_value,
        "hearing": hearing_value,
        "digestion": digestion,
        "agent_pos": tuple(agent_pos),
        "food": food_positions,
        "health": health,
        "energy": energy,  # Add energy to sensory data
        "exhausted": is_exhausted  # Add exhaustion state to sensory data
    }

    # Return sensory state and data
    sensory_states = {
        "Smell": smell_flag,
        "Touch": touch_flag,
        "Vision": vision_value != "none",
        "Hearing": hearing_value != "none",
        "Taste": False,  # Not implemented yet
        "Exhausted": is_exhausted  # Add exhaustion state to sensory states
    }

    return sensory_states, sensory_data


def update_agent_state(food_eaten_count, pixels_moved, is_running, enemies):
    """Update agent's internal state (health, digestion, energy)"""
    global health, digestion, energy, regen_timer, starvation_timer, damage_cooldown
    global agent_action, last_position, agent_pos, is_exhausted

    # Calculate energy consumption based on actual movement from last_position to agent_pos
    actual_dx = abs(agent_pos[0] - last_position[0])
    actual_dy = abs(agent_pos[1] - last_position[1])
    actual_pixels_moved = actual_dx + actual_dy
    
    # Calculate energy consumption based on actual movement
    if actual_pixels_moved > 0:
        # Calculate blocks moved (each block is GRID_SIZE pixels)
        blocks_moved = actual_pixels_moved / GRID_SIZE

        # Consume more energy when running
        if is_running:
            energy_cost = blocks_moved * ENERGY_COST_PER_BLOCK_RUNNING
        else:
            energy_cost = blocks_moved * ENERGY_COST_PER_BLOCK

        # Significantly increased energy cost multiplier to fix balance
        energy_cost *= 5.0  # Using the same multiplier as in process_move
        
        # Ensure we're always consuming some energy when moving
        energy_cost = max(energy_cost, 1.0)  # Increased minimum energy cost per move
        
        # Deduct energy cost
        energy -= energy_cost
        if energy <= 0:
            energy = 0
            # Agent passes out from exhaustion
            is_exhausted = True
            agent_action = "sleep"
            print("Agent passed out from exhaustion (energy depleted during movement)")

    # Update health: regenerate if timer active; no longer has constant decay
    if regen_timer > 0:
        health += REGEN_RATE
        if health > MAX_HEALTH:
            health = MAX_HEALTH
        regen_timer -= 1
    elif digestion <= 0:
        # Track starvation time
        starvation_timer += 1
        if starvation_timer % 1000 == 0:
            print(f"[TICK] Starving: {starvation_timer}, Health: {health}, Digestion: {digestion}")

        # Start decreasing health after STARVATION_TIME has passed
        if starvation_timer >= STARVATION_TIME:
            health -= DECAY_RATE
    else:
        # Reset starvation timer if agent has food in digestion
        starvation_timer = 0

        # Heal if digestion is above 50% and we haven't taken damage recently
        if digestion > 50 and damage_cooldown <= 0:
            health += HEALING_RATE
            if health > MAX_HEALTH:
                health = MAX_HEALTH

    # Decrease damage cooldown timer
    if damage_cooldown > 0:
        damage_cooldown -= 1

    # Update digestion based on fixed rate (not affected by movement)
    if agent_action == "sleep":
        # Slower digestion when sleeping
        digestion_decay = BASE_DIGESTION_DECAY_RATE * 0.5

        # Add energy while sleeping if digestion is above 10%
        if digestion > 10:
            # Energy gain while sleeping
            energy_gain = 0.1
            energy += energy_gain
            if energy > MAX_ENERGY:
                energy = MAX_ENERGY
                
            # Check if agent can wake up from exhaustion
            if is_exhausted and energy >= 20:
                is_exhausted = False
                print(f"Agent recovered from exhaustion, energy: {energy:.1f}%")
    else:
        # Fixed digestion rate when awake
        digestion_decay = BASE_DIGESTION_DECAY_RATE

    digestion -= digestion_decay
    if digestion < 0:
        digestion = 0

    # Add energy based on digestion (only when not sleeping)
    if digestion > 0 and agent_action != "sleep":
        # Reduced energy gain based on digestion to fix balance
        energy_gain = (ENERGY_GAIN_PER_DIGESTION * digestion_decay) * 0.2  # Reduced by 80%
        energy += energy_gain
        if energy > MAX_ENERGY:
            energy = MAX_ENERGY

    # Check for collision with enemies
    agent_damage = check_enemy_collision(agent_pos, enemies)
    if agent_damage > 0:
        health -= agent_damage
        damage_cooldown = DAMAGE_COOLDOWN  # Reset damage cooldown when taking damage
        
    # Periodically log agent status for monitoring
    if starvation_timer % 100 == 0 or (actual_pixels_moved > 0 and starvation_timer % 20 == 0):
        exhaustion_state = "EXHAUSTED" if is_exhausted else "normal"
        print(f"Agent status: Health={health:.1f}, Energy={energy:.1f}, Digestion={digestion:.1f}, Action={agent_action}, State={exhaustion_state}")

    return health <= 0  # Return True if agent died


def check_enemy_collision(agent_pos, enemies):
    """Check if agent collided with an enemy and return damage amount"""
    damage = 0
    for enemy in enemies:
        # Round enemy position for collision detection
        enemy_pos_rounded = [round(enemy['pos'][0]), round(enemy['pos'][1])]
        if agent_pos[0] == enemy_pos_rounded[0] and agent_pos[1] == enemy_pos_rounded[1]:
            damage = ENEMY_DAMAGE
            break  # Only take damage once even if multiple enemies occupy the same cell
    return damage


def check_food_collision(agent_pos, food_positions):
    """Check if agent collided with food and eat it if possible"""
    food_eaten = False
    
    # Declare global variables at the beginning of the function
    global regen_timer, digestion

    for food in list(food_positions):
        if agent_pos[0] == food[0] and agent_pos[1] == food[1]:
            # Check if digestion is below threshold to allow eating
            if digestion <= DIGESTION_THRESHOLD:
                food_positions.remove(food)
                new_food = [random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
                            random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE]
                food_positions.append(new_food)

                regen_timer = REGEN_DURATION  # Start health regeneration timer
                food_eaten = True

                # Increase digestion level
                digestion += DIGESTION_INCREASE
                if digestion > MAX_DIGESTION:
                    digestion = MAX_DIGESTION
            break

    return food_eaten, food_positions