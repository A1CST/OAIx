import torch

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Game constants
GRID_SIZE = 10
WIDTH, HEIGHT = 200, 200  # Increased from 100x100 to 200x200
FPS = 60  # Default FPS for rendering (will be adjusted based on args)
FOOD_COUNT = 10
ENEMY_COUNT = 3  # Default number of enemies (adjusted based on args)
ENEMY_DAMAGE = 20.0  # Damage dealt by enemy on contact
ENEMY_SPEED = 2.0  # Increased enemy speed for faster movement

# Game clock and day/night cycle constants
TICKS_PER_HOUR = 300  # Default ticks per hour (adjusted based on args)
HOURS_PER_DAY = 24
DAY_START_HOUR = 6  # 6 AM
NIGHT_START_HOUR = 18  # 6 PM
DAY_VISION_RANGE = 5  # Agent can see 5 blocks ahead during the day
NIGHT_VISION_RANGE = 2  # Agent can see 2 blocks ahead during the night

# Panel constants
SENSORY_PANEL_WIDTH = 140  # Increased to accommodate digestion token display
SENSORY_PANEL_HEIGHT = 160  # Keeping this size to fit within window
STATS_PANEL_WIDTH = 120
STATS_PANEL_HEIGHT = 160  # Keeping this size to fit within window
PANEL_MARGIN = 5
SENSE_LABEL_HEIGHT = 15
SENSE_INDICATOR_SIZE = 10
SENSE_TYPES = ["Smell", "Vision", "Touch", "Taste", "Hearing"]  # Added Hearing to sense types
HEARING_RANGE = 3  # Hearing range in grid cells

# Health and regeneration constants
MAX_HEALTH = 100.0
DECAY_RATE = 0.2  # Health decay per frame when starving
REGEN_DURATION = 30  # Frames for 3 seconds regen (FPS=10)
REGEN_RATE = MAX_HEALTH / REGEN_DURATION  # Health gain per frame when regenerating
STARVATION_TIME = 2 * TICKS_PER_HOUR  # 12 game hours before starvation starts
HEALING_RATE = 0.1  # Health gained per frame when digestion is high
DAMAGE_COOLDOWN = 60  # Frames to wait after taking damage before healing (6 seconds at FPS=10)

# Energy/stamina constants
MAX_ENERGY = 100.0
INITIAL_ENERGY = 50.0
ENERGY_GAIN_PER_DIGESTION = 2.0  # Energy gained per digestion point
ENERGY_COST_PER_BLOCK = 0.5  # Energy cost per block of movement (1 energy per 2 blocks)
ENERGY_COST_PER_BLOCK_RUNNING = 3.0  # Energy cost per block when running
SPEED_MULTIPLIER = 2.0  # Speed multiplier when running

# Digestion constants
MAX_DIGESTION = 100.0  # Maximum digestion value (percentage)
BASE_DIGESTION_DECAY_RATE = 0.05  # Base digestion decay per frame (when not moving)
MOVEMENT_DIGESTION_FACTOR = 0.02  # Extra digestion decay per pixel moved
DIGESTION_INCREASE = 50.0  # How much digestion increases per food eaten
DIGESTION_THRESHOLD = 51.0  # Above this percentage, agent cannot eat more food

# Checkpoint filenames for LSTMs
CENTRAL_CKPT = "checkpoints/central_lstm_checkpoint.pth"
PATTERN_CKPT = "checkpoints/pattern_lstm_checkpoint.pth"
# Add checkpoint counter for periodic saving
CHECKPOINT_INTERVAL = 10  # Save every 10 deaths

# LSTM hyperparameters
HIDDEN_SIZE = 124
INPUT_SIZE = 10  # Size of input features
OUTPUT_SIZE = 6  # Number of possible actions (up, down, left, right, sleep, run)
LEARNING_RATE = 0.01
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.9998
MIN_EXPLORATION_RATE = 0.05
EXPLORATION_PERIOD = 1000
MIN_ACTION_COUNT = 50  # Minimum times each action should be tried
MAX_CONSECUTIVE_SAME_ACTION = 2  # Maximum times to repeat the same action

# Red zone constants
RED_ZONE_WIDTH = 50  # Width of the safe zone (reduced from 100)
RED_ZONE_HEIGHT = 50  # Height of the safe zone (reduced from 100)
RED_ZONE_X = 75  # Initial X position of the red zone (centered in playfield)
RED_ZONE_Y = 75  # Initial Y position of the red zone (centered in playfield)

# Agent direction constants
DIRECTION_UP = 0
DIRECTION_RIGHT = 1
DIRECTION_DOWN = 2
DIRECTION_LEFT = 3
AGENT_DIRECTION_VECTORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left