import random
import math
from constants import *

# Enemy movement patterns for 8 directions
enemy_movement_patterns = [
    (GRID_SIZE, 0),  # Right
    (0, GRID_SIZE),  # Down
    (-GRID_SIZE, 0),  # Left
    (0, -GRID_SIZE),  # Up
    (GRID_SIZE, GRID_SIZE),  # Diagonal down-right
    (GRID_SIZE, -GRID_SIZE),  # Diagonal up-right
    (-GRID_SIZE, GRID_SIZE),  # Diagonal down-left
    (-GRID_SIZE, -GRID_SIZE)  # Diagonal up-left
]


def is_in_red_zone(pos):
    """Check if a position is within the red zone."""
    x, y = pos
    return (RED_ZONE_X <= x < RED_ZONE_X + RED_ZONE_WIDTH and
            RED_ZONE_Y <= y < RED_ZONE_Y + RED_ZONE_HEIGHT)


def update_red_zone():
    """The red zone is now stationary, so this function doesn't do anything."""
    pass


def initialize_food(count):
    """Initialize food positions"""
    return [
        [random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
         random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE]
        for _ in range(count)
    ]


def initialize_enemies(count):
    """Initialize enemies with randomized positions and movement"""
    enemies = []
    if count > 0:
        for _ in range(count):
            enemy = {
                'pos': [random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
                        random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE],
                'direction': random.randint(0, 7),  # 8 possible directions
                'direction_change_chance': 0.15  # 15% chance to change direction each move
            }
            enemies.append(enemy)
    return enemies


def update_enemies(enemies):
    """Update enemy positions using pattern-based movement"""
    for enemy in enemies:
        # Use pattern-based movement instead of targeting the agent
        # Randomly change direction based on direction_change_chance
        if random.random() < enemy['direction_change_chance']:
            enemy['direction'] = random.randint(0, 7)

        # Get movement vector from the pattern
        move_x, move_y = enemy_movement_patterns[enemy['direction']]

        # Calculate new position
        new_x = enemy['pos'][0] + move_x * ENEMY_SPEED
        new_y = enemy['pos'][1] + move_y * ENEMY_SPEED

        # Check if the new position would be in the red zone
        if not is_in_red_zone((new_x, new_y)):
            # Only move if not entering the red zone
            enemy['pos'][0] = new_x
            enemy['pos'][1] = new_y

        # Keep enemy within bounds
        enemy['pos'][0] = max(0, min(WIDTH - GRID_SIZE, enemy['pos'][0]))
        enemy['pos'][1] = max(0, min(HEIGHT - GRID_SIZE, enemy['pos'][1]))


def get_background_color(game_hour):
    """Calculate background color based on time of day"""
    # Calculate time progression (0 to 1 for sunrise, 1 to 2 for sunset)
    if game_hour < DAY_START_HOUR:
        # Before sunrise, still dark
        t = 0.0
    elif game_hour < NIGHT_START_HOUR:
        # Daytime: interpolate from sunrise to sunset
        t = (game_hour - DAY_START_HOUR) / (NIGHT_START_HOUR - DAY_START_HOUR)
        t = min(1.0, max(0.0, t))  # Clamp between 0 and 1
        if t <= 0.5:
            # First half of day: getting brighter (0 to 1)
            t = t * 2
        else:
            # Second half of day: getting darker (1 to 0)
            t = 2 - (t * 2)
    else:
        # After sunset, dark
        t = 0.0

    # Interpolate between dark (20, 20, 40) and light (100, 100, 180)
    r = int(20 + (80 * t))
    g = int(20 + (80 * t))
    b = int(40 + (140 * t))

    return (r, g, b)


def get_vision_data(agent_pos, agent_direction, enemies, food_positions, game_hour):
    """Get vision data based on agent position and direction"""
    # Determine vision range based on time of day
    is_daytime = DAY_START_HOUR <= game_hour < NIGHT_START_HOUR
    vision_range = DAY_VISION_RANGE if is_daytime else NIGHT_VISION_RANGE

    # Get agent's direction vector
    dir_vector = AGENT_DIRECTION_VECTORS[agent_direction]

    # Check cells in front of the agent
    vision_cells = []
    agent_cell_x = agent_pos[0] // GRID_SIZE
    agent_cell_y = agent_pos[1] // GRID_SIZE

    for distance in range(1, vision_range + 1):
        # Calculate cell position
        cell_x = agent_cell_x + (dir_vector[0] * distance)
        cell_y = agent_cell_y + (dir_vector[1] * distance)

        # Check if the cell is out of bounds (wall)
        if cell_x < 0 or cell_x >= WIDTH // GRID_SIZE or cell_y < 0 or cell_y >= HEIGHT // GRID_SIZE:
            vision_cells.append("wall")
            break  # Can't see beyond walls

        # Convert to pixel coordinates for checking
        cell_pixel_x = cell_x * GRID_SIZE
        cell_pixel_y = cell_y * GRID_SIZE

        # Check if enemy is in this cell
        enemy_detected = False
        for enemy in enemies:
            enemy_cell_x = enemy['pos'][0] // GRID_SIZE
            enemy_cell_y = enemy['pos'][1] // GRID_SIZE
            if cell_x == enemy_cell_x and cell_y == enemy_cell_y:
                enemy_detected = True
                break

        # Check if there's food in this cell
        food_detected = False
        for food in food_positions:
            food_cell_x = food[0] // GRID_SIZE
            food_cell_y = food[1] // GRID_SIZE
            if cell_x == food_cell_x and cell_y == food_cell_y:
                food_detected = True
                break

        # Check if the cell is in the safe zone
        safe_zone_detected = is_in_red_zone((cell_pixel_x, cell_pixel_y))

        # Determine what's in the cell
        if safe_zone_detected:
            vision_cells.append("vision:safe_zone")
        elif enemy_detected and food_detected:
            vision_cells.append("threat-food")
        elif enemy_detected:
            vision_cells.append("threat")
        elif food_detected:
            vision_cells.append("food")
        else:
            vision_cells.append("none")

    return vision_cells, vision_range


def get_hearing_data(agent_pos, enemies):
    """Get hearing data based on agent position"""
    hearing_value = "none"
    agent_cell_x = agent_pos[0] // GRID_SIZE
    agent_cell_y = agent_pos[1] // GRID_SIZE

    # Check each enemy
    for enemy in enemies:
        enemy_cell_x = enemy['pos'][0] // GRID_SIZE
        enemy_cell_y = enemy['pos'][1] // GRID_SIZE

        # Calculate distance to enemy
        dx = enemy_cell_x - agent_cell_x
        dy = enemy_cell_y - agent_cell_y
        distance = math.sqrt(dx * dx + dy * dy)  # Euclidean distance

        # If enemy is within hearing range
        if distance <= HEARING_RANGE:
            # Determine direction
            if abs(dx) > abs(dy):
                # Horizontal direction
                if dx > 0:
                    hearing_value = "enemy:right"
                else:
                    hearing_value = "enemy:left"
            else:
                # Vertical direction
                if dy > 0:
                    hearing_value = "enemy:down"
                else:
                    hearing_value = "enemy:up"
            break  # Use the first enemy detected

    return hearing_value