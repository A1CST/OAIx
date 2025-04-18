import pygame
from constants import *
from game.environment import is_in_red_zone


def setup_pygame(args):
    """Setup pygame with the appropriate display mode"""
    pygame.init()

    if args.render:
        if args.maximize:
            # Get the screen info for full screen mode
            screen_info = pygame.display.Info()
            screen_width = screen_info.current_w
            screen_height = screen_info.current_h

            # Create a full screen surface
            screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
            pygame.display.set_caption("Smell & Health Pipeline Pattern Snake (Full Screen)")

            # Create a surface for the actual game content
            game_surface = pygame.Surface((STATS_PANEL_WIDTH + WIDTH + SENSORY_PANEL_WIDTH, HEIGHT))

            # Calculate scaling factors
            scale_x = screen_width / (STATS_PANEL_WIDTH + WIDTH + SENSORY_PANEL_WIDTH)
            scale_y = screen_height / HEIGHT
            scale_factor = min(scale_x, scale_y)

            # Calculate position to center the scaled game
            scaled_width = int((STATS_PANEL_WIDTH + WIDTH + SENSORY_PANEL_WIDTH) * scale_factor)
            scaled_height = int(HEIGHT * scale_factor)
            pos_x = (screen_width - scaled_width) // 2
            pos_y = (screen_height - scaled_height) // 2

            return screen, game_surface, True, (scaled_width, scaled_height, pos_x, pos_y)
        else:
            # Normal windowed mode
            screen = pygame.display.set_mode((STATS_PANEL_WIDTH + WIDTH + SENSORY_PANEL_WIDTH, HEIGHT))
            pygame.display.set_caption("Smell & Health Pipeline Pattern Snake")
            return screen, None, False, None
    else:
        # Create a dummy screen that won't be displayed
        screen = pygame.Surface((STATS_PANEL_WIDTH + WIDTH + SENSORY_PANEL_WIDTH, HEIGHT))
        pygame.display.set_caption("Headless Mode")
        return screen, None, False, None


def draw_agent(screen, agent_pos, agent_direction, stats_panel_width=STATS_PANEL_WIDTH):
    """Draw the agent on the screen"""
    # Draw agent (white square with direction indicator)
    pygame.draw.rect(screen, (255, 255, 255), (stats_panel_width + agent_pos[0], agent_pos[1], GRID_SIZE, GRID_SIZE))

    # Draw direction indicator as a small colored rectangle inside the agent
    direction_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]  # Blue, Red, Green, Yellow
    indicator_size = GRID_SIZE // 3
    indicator_offset = (GRID_SIZE - indicator_size) // 2

    if agent_direction == 0:  # Up
        indicator_rect = (stats_panel_width + agent_pos[0] + indicator_offset, agent_pos[1] + indicator_offset,
                          indicator_size, indicator_size)
    elif agent_direction == 1:  # Right
        indicator_rect = (stats_panel_width + agent_pos[0] + GRID_SIZE - indicator_size - indicator_offset,
                          agent_pos[1] + indicator_offset, indicator_size, indicator_size)
    elif agent_direction == 2:  # Down
        indicator_rect = (stats_panel_width + agent_pos[0] + indicator_offset,
                          agent_pos[1] + GRID_SIZE - indicator_size - indicator_offset,
                          indicator_size, indicator_size)
    else:  # Left
        indicator_rect = (stats_panel_width + agent_pos[0] + indicator_offset,
                          agent_pos[1] + indicator_offset, indicator_size, indicator_size)

    pygame.draw.rect(screen, direction_colors[agent_direction], indicator_rect)


def draw_food(screen, food_positions, stats_panel_width=STATS_PANEL_WIDTH):
    """Draw food items on the screen"""
    for food in food_positions:
        pygame.draw.rect(screen, (0, 255, 0), (stats_panel_width + food[0], food[1], GRID_SIZE, GRID_SIZE))


def draw_enemies(screen, enemies, stats_panel_width=STATS_PANEL_WIDTH):
    """Draw enemies on the screen"""
    for enemy in enemies:
        pygame.draw.rect(screen, (255, 0, 0),
                         (stats_panel_width + enemy['pos'][0], enemy['pos'][1], GRID_SIZE, GRID_SIZE))


def draw_red_zone(screen, stats_panel_width=STATS_PANEL_WIDTH, font=None):
    """Draw the red zone (safe haven)"""
    red_zone_surface = pygame.Surface((RED_ZONE_WIDTH, RED_ZONE_HEIGHT), pygame.SRCALPHA)
    red_zone_surface.fill((255, 0, 0, 30))  # Semi-transparent red
    screen.blit(red_zone_surface, (stats_panel_width + RED_ZONE_X, RED_ZONE_Y))

    # Add text to indicate this is a safe zone
    if font:
        safe_text = font.render("SAFE ZONE", True, (255, 0, 0))
        screen.blit(safe_text, (stats_panel_width + RED_ZONE_X + 10, RED_ZONE_Y + 10))


def draw_vision_cells(screen, vision_cells, agent_pos, agent_direction, vision_range,
                      stats_panel_width=STATS_PANEL_WIDTH):
    """Draw vision cells on the screen"""
    # Get agent's direction vector
    dir_vector = AGENT_DIRECTION_VECTORS[agent_direction]
    agent_cell_x = agent_pos[0] // GRID_SIZE
    agent_cell_y = agent_pos[1] // GRID_SIZE

    for distance, cell_type in enumerate(vision_cells, 1):
        # Calculate cell position
        cell_x = agent_cell_x + (dir_vector[0] * distance)
        cell_y = agent_cell_y + (dir_vector[1] * distance)

        # Make sure we're within bounds
        if 0 <= cell_x < WIDTH // GRID_SIZE and 0 <= cell_y < HEIGHT // GRID_SIZE:
            screen_x = stats_panel_width + (cell_x * GRID_SIZE)
            screen_y = cell_y * GRID_SIZE

            # Draw vision cell with appropriate color
            if "threat" in cell_type:
                # Purple for threats
                pygame.draw.rect(screen, (255, 0, 255), (screen_x, screen_y, GRID_SIZE, GRID_SIZE), 2)
            elif "food" in cell_type:
                # Yellow for food
                pygame.draw.rect(screen, (255, 255, 0), (screen_x, screen_y, GRID_SIZE, GRID_SIZE), 2)
            elif "wall" in cell_type:
                # Red for wall
                pygame.draw.rect(screen, (255, 0, 0), (screen_x, screen_y, GRID_SIZE, GRID_SIZE), 2)
            else:
                # Blue for empty cells in vision
                pygame.draw.rect(screen, (0, 0, 255), (screen_x, screen_y, GRID_SIZE, GRID_SIZE), 1)


def draw_stats_panel(screen, font, health, energy, digestion, death_count, food_eaten,
                     agent_action, starvation_timer, game_hour, game_day):
    """Draw the stats panel"""
    # Draw panel background
    panel_rect = pygame.Rect(0, 0, STATS_PANEL_WIDTH, STATS_PANEL_HEIGHT)
    pygame.draw.rect(screen, (50, 50, 50), panel_rect)
    pygame.draw.rect(screen, (100, 100, 100), panel_rect, 2)  # Border

    # Draw title
    title_text = font.render("Stats Panel", True, (255, 255, 255))
    screen.blit(title_text, (PANEL_MARGIN, PANEL_MARGIN))

    # Draw health bar (red background, green for current health)
    bar_width = 100
    bar_height = 10
    current_width = int(bar_width * (health / MAX_HEALTH))
    health_bar_y = PANEL_MARGIN + 25
    pygame.draw.rect(screen, (255, 0, 0), (PANEL_MARGIN, health_bar_y, bar_width, bar_height))
    pygame.draw.rect(screen, (0, 255, 0), (PANEL_MARGIN, health_bar_y, current_width, bar_height))

    # Draw health percentage text
    health_text = font.render(f"Health: {int(health)}%", True, (255, 255, 255))
    screen.blit(health_text, (PANEL_MARGIN, health_bar_y + bar_height + 5))

    # Draw energy bar (dark blue background, light blue for current energy)
    energy_bar_y = health_bar_y + bar_height + 25
    current_energy_width = int(bar_width * (energy / MAX_ENERGY))
    pygame.draw.rect(screen, (0, 0, 100), (PANEL_MARGIN, energy_bar_y, bar_width, bar_height))
    pygame.draw.rect(screen, (0, 150, 255), (PANEL_MARGIN, energy_bar_y, current_energy_width, bar_height))

    # Draw energy percentage text
    energy_text = font.render(f"Energy: {int(energy)}%", True, (255, 255, 255))
    screen.blit(energy_text, (PANEL_MARGIN, energy_bar_y + bar_height + 5))

    # Draw death counter
    death_y_pos = energy_bar_y + bar_height + 25
    death_text = font.render(f"Deaths: {death_count}", True, (255, 255, 255))
    screen.blit(death_text, (PANEL_MARGIN, death_y_pos))

    # Draw food eaten counter
    food_y_pos = death_y_pos + 20
    food_text = font.render(f"Food Eaten: {food_eaten}", True, (255, 255, 255))
    screen.blit(food_text, (PANEL_MARGIN, food_y_pos))

    # Draw digestion level and action on same line
    digestion_y_pos = food_y_pos + 20
    digestion_text = font.render(f"Dig: {int(digestion)}%", True, (255, 255, 255))
    screen.blit(digestion_text, (PANEL_MARGIN, digestion_y_pos))

    # Draw action label
    action_text = font.render(f"Act: {agent_action}", True, (255, 255, 255))
    screen.blit(action_text, (PANEL_MARGIN + 60, digestion_y_pos))

    # Draw digestion bar
    bar_width = 100
    bar_height = 8
    bar_y_pos = digestion_y_pos + 15
    current_width = int(bar_width * (digestion / MAX_DIGESTION))

    # Draw background bar (gray)
    pygame.draw.rect(screen, (100, 100, 100), (PANEL_MARGIN, bar_y_pos, bar_width, bar_height))

    # Draw filled portion (orange for digestion)
    if digestion > DIGESTION_THRESHOLD:
        # Red when above threshold (can't eat more)
        bar_color = (255, 50, 50)
    else:
        # Orange when below threshold (can eat)
        bar_color = (255, 165, 0)
    pygame.draw.rect(screen, bar_color, (PANEL_MARGIN, bar_y_pos, current_width, bar_height))

    # Draw threshold marker (vertical line)
    threshold_x = PANEL_MARGIN + int(bar_width * (DIGESTION_THRESHOLD / MAX_DIGESTION))
    pygame.draw.line(screen, (255, 255, 255), (threshold_x, bar_y_pos), (threshold_x, bar_y_pos + bar_height), 1)

    # Draw starvation timer if digestion is 0
    starv_y_pos = bar_y_pos + 15
    hours_until_starve = max(0, (STARVATION_TIME - starvation_timer) // TICKS_PER_HOUR)
    minutes_until_starve = max(0, ((STARVATION_TIME - starvation_timer) % TICKS_PER_HOUR) * 60 // TICKS_PER_HOUR)

    if digestion == 0:
        if starvation_timer >= STARVATION_TIME:
            starv_text = font.render("STARVING", True, (255, 0, 0))
        else:
            starv_text = font.render(f"Starve: {hours_until_starve}h {minutes_until_starve}m", True, (255, 150, 150))
        screen.blit(starv_text, (PANEL_MARGIN, starv_y_pos))

    # Draw game clock and day/night on same line
    clock_y_pos = starv_y_pos + 20
    am_pm = "AM" if game_hour < 12 else "PM"
    display_hour = game_hour if game_hour <= 12 else game_hour - 12
    if display_hour == 0:
        display_hour = 12
    clock_text = font.render(f"{display_hour}:00 {am_pm}", True, (255, 255, 255))
    screen.blit(clock_text, (PANEL_MARGIN, clock_y_pos))

    # Draw day/night indicator
    is_daytime = DAY_START_HOUR <= game_hour < NIGHT_START_HOUR
    day_night_text = font.render(f"{'Day' if is_daytime else 'Night'}", True, (255, 255, 255))
    screen.blit(day_night_text, (PANEL_MARGIN + 60, clock_y_pos))

    # Draw day counter
    day_counter_y_pos = clock_y_pos + 20
    day_counter_text = font.render(f"Day: {game_day}", True, (255, 255, 255))
    screen.blit(day_counter_text, (PANEL_MARGIN, day_counter_y_pos))


def draw_sensory_panel(screen, font, sensory_states, health, digestion, vision_cells, hearing_value):
    """Draw the sensory panel"""
    # Draw panel background
    panel_rect = pygame.Rect(STATS_PANEL_WIDTH + WIDTH, 0, SENSORY_PANEL_WIDTH, SENSORY_PANEL_HEIGHT)
    pygame.draw.rect(screen, (50, 50, 50), panel_rect)
    pygame.draw.rect(screen, (100, 100, 100), panel_rect, 2)  # Border

    # Draw title
    title_text = font.render("Sensory Panel", True, (255, 255, 255))
    screen.blit(title_text, (STATS_PANEL_WIDTH + WIDTH + PANEL_MARGIN, PANEL_MARGIN))

    # Draw sense indicators (more compact)
    for i, sense in enumerate(SENSE_TYPES):
        y_pos = PANEL_MARGIN + 20 + (i * (SENSE_LABEL_HEIGHT + 2))  # Reduced spacing

        # Draw sense label
        label_text = font.render(sense, True, (255, 255, 255))
        screen.blit(label_text, (STATS_PANEL_WIDTH + WIDTH + PANEL_MARGIN, y_pos))

        # Draw indicator (lights up when active)
        indicator_color = (0, 255, 0) if sensory_states[sense] else (100, 100, 100)
        indicator_rect = pygame.Rect(
            STATS_PANEL_WIDTH + WIDTH + SENSORY_PANEL_WIDTH - SENSE_INDICATOR_SIZE - PANEL_MARGIN,
            y_pos,
            SENSE_INDICATOR_SIZE,
            SENSE_INDICATOR_SIZE
        )
        pygame.draw.rect(screen, indicator_color, indicator_rect)

    # Draw health token information
    health_y_pos = PANEL_MARGIN + 20 + (len(SENSE_TYPES) * (SENSE_LABEL_HEIGHT + 2)) + 5
    health_token_text = font.render(f"Health: {int(health)}", True, (255, 255, 255))
    screen.blit(health_token_text, (STATS_PANEL_WIDTH + WIDTH + PANEL_MARGIN, health_y_pos))

    # Draw digestion token information
    digestion_y_pos = health_y_pos + 15
    digestion_token_text = font.render(f"Digestion: {int(digestion)}", True, (255, 255, 255))
    screen.blit(digestion_token_text, (STATS_PANEL_WIDTH + WIDTH + PANEL_MARGIN, digestion_y_pos))

    # Draw vision token information
    vision_y_pos = digestion_y_pos + 15
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
    vision_token_text = font.render(f"Vision: {vision_value}", True, (255, 255, 255))
    screen.blit(vision_token_text, (STATS_PANEL_WIDTH + WIDTH + PANEL_MARGIN, vision_y_pos))

    # Draw hearing token information
    hearing_y_pos = vision_y_pos + 15
    hearing_token_text = font.render(f"Hearing: {hearing_value}", True, (255, 255, 255))
    screen.blit(hearing_token_text, (STATS_PANEL_WIDTH + WIDTH + PANEL_MARGIN, hearing_y_pos))