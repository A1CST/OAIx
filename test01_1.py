import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, patches as mpatches
import pickle
import atexit
import torch
import torch.nn as nn
import torch.optim as optim
import argparse  # Add argparse for command-line arguments
import torch.nn.functional as F
import math
import os  # Add os import for file operations
from scipy.ndimage import gaussian_filter

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run the agent simulation')
parser.add_argument('--render', action='store_true', help='Render the Pygame window')
parser.add_argument('--actions', action='store_true', help='Only show the actions chart')
parser.add_argument('--speed', type=int, default=1, help='Speed multiplier for headless mode (default: 1)')
parser.add_argument('--enemies', action='store_true', help='Enable enemies in the game (enabled by default)')
parser.add_argument('--maximize', action='store_true', help='Render the game in full screen mode')
parser.add_argument('--heatmap', action='store_true', help='Show realtime heatmap of hidden states')
args = parser.parse_args()

# Game constants
GRID_SIZE = 10
WIDTH, HEIGHT = 200, 200  # Increased from 100x100 to 200x200
FPS = 60 if args.render else 1000  # Much higher FPS in headless mode
FOOD_COUNT = 10
ENEMY_COUNT = 3 if args.enemies else 0  # Number of enemies (0 if --enemies not specified)
ENEMY_DAMAGE = 20.0  # Damage dealt by enemy on contact
ENEMY_SPEED = 2.0  # Increased enemy speed for faster movement

# Game clock and day/night cycle constants
TICKS_PER_HOUR = 300 if args.render else 60  # Reduced ticks per hour in headless mode
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
CENTRAL_CKPT = "central_lstm_checkpoint.pth"
PATTERN_CKPT = "pattern_lstm_checkpoint.pth"
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

# Function to check if a position is within the red zone
def is_in_red_zone(pos):
    """Check if a position is within the red zone."""
    x, y = pos
    return (RED_ZONE_X <= x < RED_ZONE_X + RED_ZONE_WIDTH and 
            RED_ZONE_Y <= y < RED_ZONE_Y + RED_ZONE_HEIGHT)

# Function to update red zone position - now just a placeholder since the zone is stationary
def update_red_zone():
    """The red zone is now stationary, so this function doesn't do anything."""
    pass

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.last_hidden_state = None  # Store the last hidden state for visualization
        
    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.zeros(1, 1, self.hidden_size, device=x.device)
            c0 = torch.zeros(1, 1, self.hidden_size, device=x.device)
            hidden = (h0, c0)
        else:
            # Ensure hidden state is on the same device as input
            hidden = (hidden[0].to(x.device), hidden[1].to(x.device))
        
        # Ensure input has the correct shape (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        
        # Store the hidden state for visualization
        self.last_hidden_state = hidden[0].squeeze(0).squeeze(0).detach().cpu().numpy()
        
        return out, hidden

# Real LSTM class with learning and exploration
class RealLSTM:
    def __init__(self, name, input_size, hidden_size, output_size, learning_rate):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.hidden = None
        self.exploration_rate = EXPLORATION_RATE
        self.state_history = []
        self.action_history = []
        self.input_keys = ["smell", "touch", "vision", "digestion", "agent_pos", "food", "health"]
        self.input_size = input_size
        self.steps = 0
        self.action_counts = {action: 0 for action in ["sleep", "up", "right", "down", "left", "run"]}
        self.sequence_length = 10  # Length of temporal sequences to use for learning
        
        # Add forward model for prediction error
        self.forward_model = LSTMModel(input_size + 1, hidden_size, input_size).to(self.device)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=learning_rate)
        self.forward_criterion = nn.MSELoss()
        self.forward_hidden = None
        self.prediction_errors = []  # Store prediction errors for analysis
        
        # Intrinsic motivation parameters
        self.base_exploration_rate = EXPLORATION_RATE
        self.boredom_threshold = 100  # Steps to consider for boredom
        self.boredom_window = []  # Store recent prediction errors for boredom detection
        self.boredom_boost = 0.0  # Temporary exploration boost
        self.boredom_boost_duration = 0  # How long to maintain the boost
        self.action_diversity_threshold = 50  # Minimum actions before checking diversity
        self.recent_actions = []  # Store recent actions for diversity check
        self.recent_actions_window = 20  # How many recent actions to consider
        self.temperature = 1.0  # Initial temperature for softmax
        self.temperature_min = 0.1  # Minimum temperature
        self.temperature_max = 2.0  # Maximum temperature
        self.temperature_decay = 0.995  # How quickly temperature returns to normal
        
    def process(self, sensory_data):
        # Convert sensory data to input format
        inputs = []
        for key in self.input_keys:
            if key in sensory_data:
                if isinstance(sensory_data[key], bool):
                    inputs.append(1.0 if sensory_data[key] else 0.0)
                elif isinstance(sensory_data[key], (int, float)):
                    inputs.append(float(sensory_data[key]))
                elif isinstance(sensory_data[key], str):
                    if sensory_data[key].lower() == "true":
                        inputs.append(1.0)
                    elif sensory_data[key].lower() == "false":
                        inputs.append(0.0)
                    else:
                        vision_encoding = {
                            "none": 0.0,
                            "wall": 0.2,
                            "food": 0.4,
                            "threat": 0.6,
                            "food-wall": 0.8,
                            "threat-wall": 1.0,
                            "threat-food-wall": 1.0
                        }
                        inputs.append(vision_encoding.get(sensory_data[key].lower(), 0.0))
                else:
                    inputs.append(0.0)
            else:
                inputs.append(0.0)
                
        while len(inputs) < self.input_size:
            inputs.append(0.0)

        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)

        # Get model output
        with torch.no_grad():
            output, self.hidden = self.model(input_tensor, self.hidden)
            output += torch.randn_like(output) * 0.03  # Add small noise
            
            # Apply temperature to output for adaptive exploration
            scaled_output = output / self.temperature
            probabilities = torch.softmax(scaled_output, dim=1)
            
            # Calculate entropy for entropy-based exploration
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
            
            actions = ["sleep", "up", "right", "down", "left", "run"]
            
            # Update exploration rate based on prediction error if available
            if self.prediction_errors:
                # Get the most recent prediction error
                recent_error = self.prediction_errors[-1]
                
                # Update boredom window
                self.boredom_window.append(recent_error)
                if len(self.boredom_window) > self.boredom_threshold:
                    self.boredom_window.pop(0)
                
                # Calculate average error in boredom window
                avg_error = sum(self.boredom_window) / len(self.boredom_window)
                
                # Prediction error-driven exploration
                error_factor = recent_error / (recent_error + 1.0)
                
                # Entropy-based exploration (higher entropy = more exploration)
                entropy_factor = entropy.item() / 2.0  # Normalize entropy
                
                # Boredom-based exploration (low average error = boredom)
                boredom_factor = 0.0
                if len(self.boredom_window) >= self.boredom_threshold:
                    if avg_error < 0.1:  # Low novelty threshold
                        boredom_factor = 0.5
                        self.boredom_boost = 0.3  # Boost exploration
                        self.boredom_boost_duration = 50  # For 50 steps
                
                # Action diversity pressure
                diversity_factor = 0.0
                if len(self.recent_actions) >= self.recent_actions_window:
                    # Check if actions are diverse
                    unique_actions = len(set(self.recent_actions))
                    if unique_actions < self.recent_actions_window // 2:  # Less than half are unique
                        diversity_factor = 0.4
                
                # Combine all factors
                self.exploration_rate = self.base_exploration_rate + error_factor + entropy_factor + boredom_factor + diversity_factor
                
                # Add boredom boost if active
                if self.boredom_boost_duration > 0:
                    self.exploration_rate += self.boredom_boost
                    self.boredom_boost_duration -= 1
                
                # Clamp exploration rate
                self.exploration_rate = min(max(MIN_EXPLORATION_RATE, self.exploration_rate), 1.0)
                
                # Adjust temperature based on prediction error
                # Higher error = higher temperature = more exploration
                self.temperature = self.temperature_min + (self.temperature_max - self.temperature_min) * error_factor
            else:
                # If no prediction errors yet, use base exploration rate
                self.exploration_rate = self.base_exploration_rate
            
            # Decay temperature back to normal
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
            
            should_explore = False
            min_action_count = min(self.action_counts.values())
            
            # Find actions that haven't reached minimum count
            actions_below_min = [action for action, count in self.action_counts.items() 
                               if count < MIN_ACTION_COUNT]
            
            if actions_below_min:
                # Choose randomly from actions that haven't reached minimum
                action = random.choice(actions_below_min)
                action_idx = actions.index(action)
                should_explore = True
            elif random.random() < self.exploration_rate:
                # Only explore randomly if all actions have reached minimum
                action_idx = random.randint(0, len(actions) - 1)
                should_explore = True
            
            if should_explore:
                # Update exploration rate only after minimum actions are tried
                if min_action_count >= MIN_ACTION_COUNT and self.steps > EXPLORATION_PERIOD:
                    self.exploration_rate = max(MIN_EXPLORATION_RATE, 
                                             self.exploration_rate * EXPLORATION_DECAY)
            else:
                # Choose best action during exploitation
                action_idx = torch.argmax(probabilities).item()
                # Ensure action_idx is within valid range
                action_idx = max(0, min(action_idx, len(actions) - 1))

        # Update action counts and store state/action
        action = actions[action_idx]
        self.action_counts[action] += 1
        self.steps += 1
        
        # Print minimum action count and norm detection trigger value
        min_action_count = min(self.action_counts.values())
        norm_detection_trigger = self.boredom_threshold
        print(f"\rMin Action Count: {min_action_count}, Norm Detection: {norm_detection_trigger}, Exploration Rate: {self.exploration_rate:.2f}", end="")
        
        # Update recent actions for diversity tracking
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.recent_actions_window:
            self.recent_actions.pop(0)
        
        # Store current state and action
        self.state_history.append(inputs)
        self.action_history.append(action_idx)
        
        # Keep only the last sequence_length states and actions
        if len(self.state_history) > self.sequence_length:
            self.state_history.pop(0)
            self.action_history.pop(0)
        
        # Map action index to movement
        if action == "sleep":
            return (0, 0)
        elif action == "up":
            return (0, -GRID_SIZE)
        elif action == "right":
            return (GRID_SIZE, 0)
        elif action == "down":
            return (0, GRID_SIZE)
        elif action == "left":
            return (-GRID_SIZE, 0)
        elif action == "run":
            dir_vector = agent_direction_vectors[agent_direction]
            return (dir_vector[0] * GRID_SIZE * SPEED_MULTIPLIER, 
                   dir_vector[1] * GRID_SIZE * SPEED_MULTIPLIER)
        else:
            return (0, 0)
    
    def learn(self):
        # Only learn if we have enough history
        if len(self.state_history) < self.sequence_length:
            return
            
        # Convert state history to tensor sequence
        state_sequence = torch.FloatTensor(self.state_history).unsqueeze(0).to(self.device)
        
        # Get the target action (last action in sequence)
        target = torch.tensor([self.action_history[-1]], dtype=torch.long).to(self.device)
        
        # Forward pass with the entire sequence
        self.optimizer.zero_grad()
        output, _ = self.model(state_sequence)
        
        # Calculate prediction error using forward model
        if len(self.state_history) >= 2:
            # Get previous state and action
            prev_state = torch.FloatTensor(self.state_history[-2]).unsqueeze(0).to(self.device)
            prev_action = torch.tensor([self.action_history[-2]], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Combine state and action for forward model input
            forward_input = torch.cat([prev_state, prev_action], dim=1)
            
            # Get actual next state
            actual_next = torch.FloatTensor(self.state_history[-1]).unsqueeze(0).to(self.device)
            
            # Predict next state
            predicted_next, _ = self.forward_model(forward_input, self.forward_hidden)
            
            # Calculate prediction error (novelty)
            error = torch.norm(predicted_next - actual_next)
            
            # Store prediction error for analysis
            self.prediction_errors.append(error.item())
            
            # Keep only the last 1000 prediction errors
            if len(self.prediction_errors) > 1000:
                self.prediction_errors.pop(0)
            
            # Train forward model
            self.forward_optimizer.zero_grad()
            forward_loss = self.forward_criterion(predicted_next, actual_next)
            forward_loss.backward()
            self.forward_optimizer.step()
            
            # Weight the main loss by prediction error (novelty)
            # Higher error = higher novelty = more learning
            novelty_weight = error.detach().clamp(max=1.0)
            loss = self.criterion(output, target) * (1.0 + novelty_weight)
        else:
            # If we don't have enough history for prediction, use regular loss
            loss = self.criterion(output, target)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
    
    def _prepare_input(self, input_data):
        features = []
        
        if isinstance(input_data, dict):
            features.append(1.0 if input_data.get("smell_token") == "1" else 0.0)
            features.append(1.0 if input_data.get("touch_token") == "1" else 0.0)
            
            vision = input_data.get("vision_token", "none")
            vision_map = {"none": 0.0, "wall": 1.0, "food": 2.0, "threat": 3.0, 
                         "food-wall": 4.0, "threat-wall": 5.0, "threat-food-wall": 6.0}
            features.append(vision_map.get(vision, 0.0))
            
            features.append(float(input_data.get("health_token", 0)) / MAX_HEALTH)
            features.append(float(input_data.get("digestion_token", 0)) / MAX_DIGESTION)
            
            if "agent_pos" in input_data:
                pos_x, pos_y = input_data["agent_pos"]
                features.append(pos_x / WIDTH)
                features.append(pos_y / HEIGHT)
            
            if "food" in input_data and input_data["food"]:
                food_x, food_y = input_data["food"][0]
                features.append(food_x / WIDTH)
                features.append(food_y / HEIGHT)
            else:
                features.append(0.0)
                features.append(0.0)
        else:
            features = [0.0] * INPUT_SIZE
        
        while len(features) < INPUT_SIZE:
            features.append(0.0)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
    
    def load_checkpoint(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.hidden = checkpoint['hidden']
            self.exploration_rate = checkpoint.get('exploration_rate', EXPLORATION_RATE)
            self.steps = checkpoint.get('steps', 0)
            self.action_counts = checkpoint.get('action_counts', {action: 0 for action in ["sleep", "up", "right", "down", "left", "run"]})
            
            # Load forward model data if available
            if 'forward_model_state_dict' in checkpoint:
                self.forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
                self.forward_optimizer.load_state_dict(checkpoint['forward_optimizer_state_dict'])
                self.forward_hidden = checkpoint['forward_hidden']
                self.prediction_errors = checkpoint.get('prediction_errors', [])
            
            print(f"{self.name} checkpoint loaded from {filepath}.")
        except (FileNotFoundError, EOFError, RuntimeError):
            print(f"No valid checkpoint for {self.name} found; starting new.")

    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hidden': self.hidden,
            'exploration_rate': self.exploration_rate,
            'steps': self.steps,
            'action_counts': self.action_counts,
            'forward_model_state_dict': self.forward_model.state_dict(),
            'forward_optimizer_state_dict': self.forward_optimizer.state_dict(),
            'forward_hidden': self.forward_hidden,
            'prediction_errors': self.prediction_errors
        }
        torch.save(checkpoint, filepath)
        print(f"{self.name} checkpoint saved to {filepath}.")


# Create LSTM instances and load checkpoints
pattern_lstm = RealLSTM("Pattern LSTM", INPUT_SIZE, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE)
central_lstm = RealLSTM("Central LSTM", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE)
pattern_lstm.load_checkpoint(PATTERN_CKPT)
central_lstm.load_checkpoint(CENTRAL_CKPT)
atexit.register(lambda: pattern_lstm.save_checkpoint(PATTERN_CKPT))
atexit.register(lambda: central_lstm.save_checkpoint(CENTRAL_CKPT))

# Initialize pygame
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
    else:
        # Normal windowed mode
        screen = pygame.display.set_mode((STATS_PANEL_WIDTH + WIDTH + SENSORY_PANEL_WIDTH, HEIGHT))
        pygame.display.set_caption("Smell & Health Pipeline Pattern Snake")
else:
    # Create a dummy screen that won't be displayed
    screen = pygame.Surface((STATS_PANEL_WIDTH + WIDTH + SENSORY_PANEL_WIDTH, HEIGHT))
    pygame.display.set_caption("Headless Mode")
clock = pygame.time.Clock()

# Initialize agent, food, and health variables
agent_pos = [
    random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
    random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE
]
# Random initial direction: 0=up, 1=right, 2=down, 3=left
agent_direction = random.randint(0, 3)
agent_direction_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
agent_action = "sleep"  # Initial action state
agent_actions_history = []  # Store action history for plotting

# Statistics tracking for charts
current_game_time = 0  # Current game time in ticks
longest_game_time = 0  # Longest game time in ticks
survival_times_history = []  # Store survival times for plotting
food_eaten_history = []  # Track food eaten over time
health_lost_history = []  # Track health lost over time
total_health_lost = 0  # Track total health lost in current life
time_points = []  # Time points for food/health chart

food_positions = [
    [random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
     random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE]
    for _ in range(FOOD_COUNT)
]

# Initialize enemies with randomized positions and movement
enemies = []
if ENEMY_COUNT > 0:  # Only create enemies if ENEMY_COUNT is greater than 0
    for _ in range(ENEMY_COUNT):
        enemy = {
            'pos': [random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
                    random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE],
            'direction': random.randint(0, 7),  # 8 possible directions
            'direction_change_chance': 0.15  # 15% chance to change direction each move
        }
        enemies.append(enemy)

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

health = MAX_HEALTH
digestion = 0.0  # Start with no digestion
energy = INITIAL_ENERGY  # Start with initial energy
regen_timer = 0
death_count = 0
food_eaten = 0
starvation_timer = 0  # Track how long agent has been at 0% digestion
damage_cooldown = 0  # Track frames since last damage taken

# Initialize game clock
game_ticks = 0
game_hour = 6  # Start at 6 AM
game_day = 1  # Start at day 1

# Initialize sensory states
sensory_states = {sense: False for sense in SENSE_TYPES}

# Initialize survival log file if it doesn't exist
if not os.path.exists("survival_log.csv"):
    with open("survival_log.csv", "w") as f:
        f.write("day,ticks,food,death\n")

# Pipeline sub-functions (dummy implementations)
def tokenize(sensory_data):
    # Create tokens from sensory data
    tokens = {
        "smell_token": "1" if sensory_data["smell"] == "true" else "0",
        "touch_token": "1" if sensory_data["touch"] == "true" else "0",
        "vision_token": sensory_data["vision"],
        "health_token": str(int(sensory_data["health"])),  # Convert health to string token
        "digestion_token": str(int(sensory_data["digestion"]))  # Convert digestion to string token
    }
    # Add the original sensory data as well
    tokens.update(sensory_data)
    return tokens


def encode(tokenized_data):
    return tokenized_data


def decode(processed_data):
    return processed_data


def reverse_tokenizer(decoded_data):
    # Pass through the command unchanged.
    return decoded_data


def pipeline(sensory_data):
    data = tokenize(sensory_data)
    data = encode(data)
    # Process with pattern LSTM (pattern recognition)
    data = pattern_lstm.process(data)
    # Process with central LSTM that returns only a directional command.
    command = central_lstm.process(data)
    data = decode(command)
    action = reverse_tokenizer(data)
    
    # Learn from the experience
    pattern_lstm.learn()
    central_lstm.learn()
    
    return action


# Function to get vision data based on agent position and direction
def get_vision_data():
    # Determine vision range based on time of day
    is_daytime = DAY_START_HOUR <= game_hour < NIGHT_START_HOUR
    vision_range = DAY_VISION_RANGE if is_daytime else NIGHT_VISION_RANGE

    # Get agent's direction vector
    dir_vector = agent_direction_vectors[agent_direction]

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


# Function to draw vision cells on the screen
def draw_vision_cells(vision_cells, vision_range):
    # Get agent's direction vector
    dir_vector = agent_direction_vectors[agent_direction]
    agent_cell_x = agent_pos[0] // GRID_SIZE
    agent_cell_y = agent_pos[1] // GRID_SIZE

    for distance, cell_type in enumerate(vision_cells, 1):
        # Calculate cell position
        cell_x = agent_cell_x + (dir_vector[0] * distance)
        cell_y = agent_cell_y + (dir_vector[1] * distance)

        # Make sure we're within bounds
        if 0 <= cell_x < WIDTH // GRID_SIZE and 0 <= cell_y < HEIGHT // GRID_SIZE:
            screen_x = STATS_PANEL_WIDTH + (cell_x * GRID_SIZE)
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


# Calculate background color based on time of day
def get_background_color():
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


# Draw static flowchart once
def draw_flowchart():
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


# Set up action history plot
plt.ion()
if not args.maximize or args.actions:  # Only set up plots if not in maximize-only mode
    fig_action, ax_action = plt.subplots(figsize=(8, 4))
    ax_action.set_ylim(-0.5, 5.5)  # Updated to accommodate 6 actions
    ax_action.set_yticks([0, 1, 2, 3, 4, 5])
    ax_action.set_yticklabels(['sleep', 'up', 'right', 'down', 'left', 'run'])
    ax_action.set_xlabel('Time Steps')
    ax_action.set_title('Agent Action History')
    action_line, = ax_action.plot([], [], 'b-')
    action_mapping = {'sleep': 0, 'up': 1, 'right': 2, 'down': 3, 'left': 4, 'run': 5}

    # Set up survival time plot (only if not in actions-only mode)
    if not args.actions:
        fig_survival, ax_survival = plt.subplots(figsize=(8, 4))
        ax_survival.set_xlabel('Game #')
        ax_survival.set_ylabel('Survival Time (hours)')
        ax_survival.set_title('Longest Survival Times')
        survival_bars = ax_survival.bar([], [], color='green')
        ax_survival.set_ylim(0, 10)  # Start with 10 hours as max, will adjust dynamically

        # Set up food eaten per game plot
        fig_stats, ax_stats = plt.subplots(figsize=(8, 4))
        ax_stats.set_xlabel('Game #')
        ax_stats.set_ylabel('Food Eaten')
        ax_stats.set_title('Food Eaten Per Game')
        ax_stats.grid(True, linestyle='--', alpha=0.7)
        
        # Set up health tracking plot (only in headless mode and not in maximize mode)
        if not args.render and not args.maximize:
            fig_health, ax_health = plt.subplots(figsize=(8, 4))
            ax_health.set_xlabel('Time Steps')
            ax_health.set_ylabel('Health')
            ax_health.set_title('Current Game Health')
            ax_health.set_ylim(0, MAX_HEALTH)
            ax_health.grid(True, linestyle='--', alpha=0.7)
            health_line, = ax_health.plot([], [], 'r-', label='Health')
            ax_health.legend()

# Initialize list to track food eaten per game
food_eaten_per_game = []
# Initialize list to track health over time in current game
current_health_history = []
current_time_steps = []

def update_action_plot():
    # Only update if not in maximize-only mode
    if args.maximize and not args.actions:
        return
        
    # Convert action strings to numerical values for plotting
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


def update_survival_plot():
    # Only update if not in maximize-only mode or actions-only mode
    if args.maximize or args.actions:
        return
        
    # Update survival time plot
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


def update_stats_plot():
    # Only update if not in maximize-only mode or actions-only mode
    if args.maximize or args.actions:
        return
        
    # Update food eaten per game plot
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


def update_health_plot():
    # Only update if in headless mode and not in maximize mode
    if args.render or args.actions or args.maximize:
        return
        
    # Update health tracking plot
    if current_health_history:
        # Clear the plot first
        ax_health.clear()
        ax_health.set_xlabel('Time Steps')
        ax_health.set_ylabel('Health')
        ax_health.set_title('Current Game Health')
        ax_health.set_ylim(0, MAX_HEALTH)
        ax_health.grid(True, linestyle='--', alpha=0.7)
        
        # Plot health line
        ax_health.plot(current_time_steps, current_health_history, 'r-', label='Health')
        ax_health.legend()
        
        # Set x-axis limits to show the most recent data
        if len(current_time_steps) > 100:
            ax_health.set_xlim(max(0, len(current_time_steps) - 100), len(current_time_steps))
        else:
            ax_health.set_xlim(0, max(100, len(current_time_steps)))
        
        # Refresh the plot
        fig_health.canvas.draw_idle()
        plt.pause(0.001)


# Function to get hearing data based on agent position
def get_hearing_data():
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
        distance = math.sqrt(dx*dx + dy*dy)  # Euclidean distance
        
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


# Function to draw the sensory panel
def draw_sensory_panel():
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
    hearing_value = get_hearing_data()
    hearing_token_text = font.render(f"Hearing: {hearing_value}", True, (255, 255, 255))
    screen.blit(hearing_token_text, (STATS_PANEL_WIDTH + WIDTH + PANEL_MARGIN, hearing_y_pos))


# Function to draw the stats panel
def draw_stats_panel():
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


# Function to create and update the heatmap visualization
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
    
    return fig_heatmap, (ax_raw, ax_grid), (heatmap_img_raw, heatmap_img_grid), (agent_marker, food_markers, enemy_markers), exploration_grid

def update_heatmap(fig_heatmap, axes, heatmap_imgs, markers, central_lstm, pattern_lstm, agent_pos, food_positions, enemies, exploration_grid):
    """Update the heatmap with the current hidden states and exploration data."""
    if central_lstm.model.last_hidden_state is None or pattern_lstm.model.last_hidden_state is None:
        return
    
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

# Draw the static flowchart before the game starts (only if not in maximize-only mode or actions-only mode)
if not args.maximize and not args.actions:
    draw_flowchart()

# Initialize heatmap visualization if enabled
if args.heatmap:
    fig_heatmap, (ax_raw, ax_grid), (heatmap_img_raw, heatmap_img_grid), markers, exploration_grid = setup_heatmap()

# Prepare font for HUD elements
font = pygame.font.Font(None, 18)

# Main game loop
running = True
last_exploration_print_time = 0  # Track when we last printed exploration rate
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Add escape key to exit full screen mode
            if event.key == pygame.K_ESCAPE and args.maximize:
                running = False

    # Update game clock
    game_ticks += 1
    current_game_time += 1  # Increment current game time
    
    # Print exploration rate and average novelty every 5 seconds (50 ticks at FPS=10)
    if game_ticks % 50 == 0:
        # Calculate average novelty from prediction errors
        avg_novelty = 0.0
        if central_lstm.prediction_errors:
            avg_novelty = sum(central_lstm.prediction_errors) / len(central_lstm.prediction_errors)
        print(f"Exploration Rate: {central_lstm.exploration_rate:.4f}, Average Novelty: {avg_novelty:.4f}")
        last_exploration_print_time = game_ticks
        
        # Update heatmap visualization if enabled
        if args.heatmap:
            exploration_grid = update_heatmap(fig_heatmap, (ax_raw, ax_grid), (heatmap_img_raw, heatmap_img_grid), 
                          markers, central_lstm, pattern_lstm, agent_pos, food_positions, enemies, exploration_grid)

    # Update game hour every TICKS_PER_HOUR
    if game_ticks >= TICKS_PER_HOUR:
        game_ticks = 0
        game_hour = (game_hour + 1) % HOURS_PER_DAY
        
        # Increment day counter when we reach midnight (hour 0)
        if game_hour == 0:
            game_day += 1
            print(f"Day {game_day} has begun")

    # Get background color based on time of day
    bg_color = get_background_color()
    screen.fill(bg_color)

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

    # Get vision data
    vision_cells, vision_range = get_vision_data()
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
    hearing_value = get_hearing_data()

    # Update sensory states
    sensory_states["Smell"] = smell_flag
    sensory_states["Touch"] = touch_flag
    sensory_states["Vision"] = vision_value != "none"
    sensory_states["Hearing"] = hearing_value != "none"
    # Taste is not implemented yet, so it remains False

    # Gather sensory data with smell, touch, vision, and hearing as inputs
    sensory_data = {
        "smell": "true" if smell_flag else "false",
        "touch": "true" if touch_flag else "false",
        "vision": vision_value,
        "hearing": hearing_value,
        "digestion": digestion,
        "agent_pos": tuple(agent_pos),
        "food": food_positions,
        "health": health
    }

    # Process through the pipeline; central LSTM will output a valid command.
    move = pipeline(sensory_data)
    
    # Check if agent has enough energy to move
    if energy <= 0:
        # Agent can't move if energy is completely depleted
        move = (0, 0)
        agent_action = "sleep"  # Force sleep only when out of energy
        is_running = False  # Ensure running is disabled when out of energy
    elif energy < 20 and agent_action == "sleep" and digestion > 0:
        # If energy is below 20 and agent is already sleeping and has digestion, keep sleeping
        move = (0, 0)
        is_running = False
    elif energy >= 20 and agent_action == "sleep":
        # If energy is above 20 and agent is sleeping, force wake up
        # The agent will use the next action from the model
        is_running = False
    else:
        # Check if the action is "run" - only if we have energy
        is_running = False
        if move == (0, 0) and hasattr(central_lstm, 'last_action') and central_lstm.last_action == "run":
            is_running = True
            # When running, move in the current direction at double speed
            dir_vector = agent_direction_vectors[agent_direction]
            move = (dir_vector[0] * GRID_SIZE * SPEED_MULTIPLIER, dir_vector[1] * GRID_SIZE * SPEED_MULTIPLIER)
    
    # Calculate potential new position
    new_pos_x = agent_pos[0] + move[0]
    new_pos_y = agent_pos[1] + move[1]

    # Update agent position with optional wall collision
    # If wall collision is enabled, the agent stops at the wall
    # If wrapping is enabled, agent can wrap around the screen
    ENABLE_WALL_COLLISION = True
    ENABLE_WRAPPING = False

    if ENABLE_WALL_COLLISION:
        # Restrict movement at walls
        if new_pos_x < 0:
            new_pos_x = 0
        elif new_pos_x >= WIDTH:
            new_pos_x = WIDTH - GRID_SIZE

        if new_pos_y < 0:
            new_pos_y = 0
        elif new_pos_y >= HEIGHT:
            new_pos_y = HEIGHT - GRID_SIZE
    elif ENABLE_WRAPPING:
        # Wrap around the screen
        new_pos_x = new_pos_x % WIDTH
        new_pos_y = new_pos_y % HEIGHT
    else:
        # Default behavior: stop at walls with no wrapping
        new_pos_x = max(0, min(new_pos_x, WIDTH - GRID_SIZE))
        new_pos_y = max(0, min(new_pos_y, HEIGHT - GRID_SIZE))

    # Update agent position
    agent_pos[0] = new_pos_x
    agent_pos[1] = new_pos_y

    # Calculate distance moved for digestion calculation
    pixels_moved = abs(move[0]) + abs(move[1])
    
    # Consume energy based on movement
    if pixels_moved > 0:
        # Calculate blocks moved (each block is GRID_SIZE pixels)
        blocks_moved = pixels_moved / GRID_SIZE
        
        # Consume more energy when running
        if is_running:
            energy_cost = blocks_moved * ENERGY_COST_PER_BLOCK_RUNNING
        else:
            energy_cost = blocks_moved * ENERGY_COST_PER_BLOCK
            
        energy -= energy_cost
        if energy < 0:
            energy = 0

    # Update agent direction and action based on movement
    if move[0] < 0:
        agent_direction = 3  # Left
        agent_action = "left"
    elif move[0] > 0:
        agent_direction = 1  # Right
        agent_action = "right"
    elif move[1] < 0:
        agent_direction = 0  # Up
        agent_action = "up"
    elif move[1] > 0:
        agent_direction = 2  # Down
        agent_action = "down"
    else:
        agent_action = "sleep"
        
    # If we're running, update the action to "run"
    if is_running:
        agent_action = "run"

    # Track action for plotting
    agent_actions_history.append(agent_action)

    # Check for food collision (agent "eats" food)
    for food in list(food_positions):
        if agent_pos[0] == food[0] and agent_pos[1] == food[1]:
            # Check if digestion is below threshold to allow eating
            if digestion <= DIGESTION_THRESHOLD:
                food_positions.remove(food)
                new_food = [random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
                            random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE]
                food_positions.append(new_food)
                regen_timer = REGEN_DURATION  # Start health regeneration timer
                food_eaten += 1  # Increment food eaten counter

                # Increase digestion level
                digestion += DIGESTION_INCREASE
                if digestion > MAX_DIGESTION:
                    digestion = MAX_DIGESTION
            break

    # Check for enemy collision
    for enemy in enemies:
        # Round enemy position for collision detection
        enemy_pos_rounded = [round(enemy['pos'][0]), round(enemy['pos'][1])]
        if agent_pos[0] == enemy_pos_rounded[0] and agent_pos[1] == enemy_pos_rounded[1]:
            health -= ENEMY_DAMAGE
            total_health_lost += ENEMY_DAMAGE  # Track total health lost
            damage_cooldown = DAMAGE_COOLDOWN  # Reset damage cooldown when taking damage
            break  # Only take damage once even if multiple enemies occupy the same cell

    # Update enemy positions
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

    # Update health: regenerate if timer active; no longer has constant decay
    if regen_timer > 0:
        health += REGEN_RATE
        if health > MAX_HEALTH:
            health = MAX_HEALTH
        regen_timer -= 1
    elif digestion <= 0:
        # Track starvation time
        starvation_timer += 1

        # Start decreasing health after STARVATION_TIME has passed
        if starvation_timer >= STARVATION_TIME:
            health -= DECAY_RATE
            total_health_lost += DECAY_RATE  # Track health lost due to starvation
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
            # Gain energy while sleeping
            energy_gain = 0.5  # Small energy gain per tick while sleeping
            energy += energy_gain
            if energy > MAX_ENERGY:
                energy = MAX_ENERGY
    else:
        # Fixed digestion rate when awake
        digestion_decay = BASE_DIGESTION_DECAY_RATE
    
    digestion -= digestion_decay
    if digestion < 0:
        digestion = 0
        
    # Add energy based on digestion (only when not sleeping)
    if digestion > 0 and agent_action != "sleep":
        # Gain energy based on digestion
        energy_gain = ENERGY_GAIN_PER_DIGESTION * digestion_decay
        energy += energy_gain
        if energy > MAX_ENERGY:
            energy = MAX_ENERGY

    # Check for death: reset health, agent, action history and increment death counter.
    if health <= 0:
        death_count += 1

        # Determine cause of death
        death_cause = "starvation" if starvation_timer >= STARVATION_TIME else "enemy"
        if starvation_timer >= STARVATION_TIME:
            print(f"Agent died from starvation at game hour {game_hour}")
        else:
            print(f"Agent died from enemy damage at game hour {game_hour}")

        # Log survival data
        with open("survival_log.csv", "a") as f:
            f.write(f"{game_day},{current_game_time},{food_eaten},{death_cause}\n")

        # Store survival time before resetting
        survival_times_history.append(current_game_time)
        longest_game_time = max(longest_game_time, current_game_time)
        
        # Display survival time information in console
        hours_alive = current_game_time / TICKS_PER_HOUR
        longest_hours = longest_game_time / TICKS_PER_HOUR
        print(f"Last game survival time: {hours_alive:.2f} hours ({current_game_time} ticks)")
        print(f"Highest survival time: {longest_hours:.2f} hours ({longest_game_time} ticks)")
        
        update_survival_plot()

        # Store food eaten for this game and reset counters
        food_eaten_per_game.append(food_eaten)
        update_stats_plot()

        # Save checkpoint every CHECKPOINT_INTERVAL deaths
        if death_count % CHECKPOINT_INTERVAL == 0:
            print(f"Saving checkpoint after {death_count} deaths...")
            pattern_lstm.save_checkpoint(PATTERN_CKPT)
            central_lstm.save_checkpoint(CENTRAL_CKPT)
            print("Checkpoint saved. Reloading models...")
            pattern_lstm.load_checkpoint(PATTERN_CKPT)
            central_lstm.load_checkpoint(CENTRAL_CKPT)
            print("Models reloaded. Continuing training...")

        # Reset game statistics
        health = MAX_HEALTH
        energy = INITIAL_ENERGY  # Reset energy to initial value
        regen_timer = 0
        current_game_time = 0
        total_health_lost = 0
        food_eaten = 0  # Reset food eaten counter
        starvation_timer = 0  # Reset starvation timer
        damage_cooldown = 0  # Reset damage cooldown

        # Reset all tracking arrays for new life
        agent_actions_history = []  # Reset action history

        # Reset agent position
        agent_pos = [
            random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
            random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE
        ]

        # Reset health tracking data for new game
        current_health_history = []
        current_time_steps = []
        update_health_plot()

    # Update action plot
    update_action_plot()

    # Draw food (green squares)
    for food in food_positions:
        pygame.draw.rect(screen, (0, 255, 0), (STATS_PANEL_WIDTH + food[0], food[1], GRID_SIZE, GRID_SIZE))

    # Draw enemies (red squares)
    for enemy in enemies:
        pygame.draw.rect(screen, (255, 0, 0),
                         (STATS_PANEL_WIDTH + enemy['pos'][0], enemy['pos'][1], GRID_SIZE, GRID_SIZE))

    # Draw agent (white square with direction indicator)
    pygame.draw.rect(screen, (255, 255, 255), (STATS_PANEL_WIDTH + agent_pos[0], agent_pos[1], GRID_SIZE, GRID_SIZE))

    # Draw direction indicator as a small colored rectangle inside the agent
    direction_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]  # Blue, Red, Green, Yellow
    indicator_size = GRID_SIZE // 3
    indicator_offset = (GRID_SIZE - indicator_size) // 2

    if agent_direction == 0:  # Up
        indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + indicator_offset, agent_pos[1] + indicator_offset,
                          indicator_size, indicator_size)
    elif agent_direction == 1:  # Right
        indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + GRID_SIZE - indicator_size - indicator_offset,
                          agent_pos[1] + indicator_offset, indicator_size, indicator_size)
    elif agent_direction == 2:  # Down
        indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + indicator_offset,
                          agent_pos[1] + GRID_SIZE - indicator_size - indicator_offset,
                          indicator_size, indicator_size)
    else:  # Left
        indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + indicator_offset,
                          agent_pos[1] + indicator_offset, indicator_size, indicator_size)

    pygame.draw.rect(screen, direction_colors[agent_direction], indicator_rect)

    # Draw vision cells
    draw_vision_cells(vision_cells, vision_range)

    # Draw the stats panel
    draw_stats_panel()

    # Draw the sensory panel
    draw_sensory_panel()

    # Draw the red zone (safe haven)
    red_zone_surface = pygame.Surface((RED_ZONE_WIDTH, RED_ZONE_HEIGHT), pygame.SRCALPHA)
    red_zone_surface.fill((255, 0, 0, 30))  # Semi-transparent red
    screen.blit(red_zone_surface, (STATS_PANEL_WIDTH + RED_ZONE_X, RED_ZONE_Y))

    # Add text to indicate this is a safe zone
    safe_text = font.render("SAFE ZONE", True, (255, 0, 0))
    screen.blit(safe_text, (STATS_PANEL_WIDTH + RED_ZONE_X + 10, RED_ZONE_Y + 10))

    # Draw everything
    if args.render:
        if args.maximize:
            # Draw the game surface
            game_surface.fill(bg_color)
            
            # Draw food (green squares)
            for food in food_positions:
                pygame.draw.rect(game_surface, (0, 255, 0), (STATS_PANEL_WIDTH + food[0], food[1], GRID_SIZE, GRID_SIZE))

            # Draw enemies (red squares)
            for enemy in enemies:
                pygame.draw.rect(game_surface, (255, 0, 0),
                                (STATS_PANEL_WIDTH + enemy['pos'][0], enemy['pos'][1], GRID_SIZE, GRID_SIZE))

            # Draw agent (white square with direction indicator)
            pygame.draw.rect(game_surface, (255, 255, 255), (STATS_PANEL_WIDTH + agent_pos[0], agent_pos[1], GRID_SIZE, GRID_SIZE))

            # Draw direction indicator as a small colored rectangle inside the agent
            direction_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]  # Blue, Red, Green, Yellow
            indicator_size = GRID_SIZE // 3
            indicator_offset = (GRID_SIZE - indicator_size) // 2

            if agent_direction == 0:  # Up
                indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + indicator_offset, agent_pos[1] + indicator_offset,
                                indicator_size, indicator_size)
            elif agent_direction == 1:  # Right
                indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + GRID_SIZE - indicator_size - indicator_offset,
                                agent_pos[1] + indicator_offset, indicator_size, indicator_size)
            elif agent_direction == 2:  # Down
                indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + indicator_offset,
                                agent_pos[1] + GRID_SIZE - indicator_size - indicator_offset,
                                indicator_size, indicator_size)
            else:  # Left
                indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + indicator_offset,
                                agent_pos[1] + indicator_offset, indicator_size, indicator_size)

            pygame.draw.rect(game_surface, direction_colors[agent_direction], indicator_rect)

            # Draw vision cells
            draw_vision_cells(vision_cells, vision_range)

            # Draw the stats panel
            draw_stats_panel()

            # Draw the sensory panel
            draw_sensory_panel()

            # Draw the red zone (safe haven)
            red_zone_surface = pygame.Surface((RED_ZONE_WIDTH, RED_ZONE_HEIGHT), pygame.SRCALPHA)
            red_zone_surface.fill((255, 0, 0, 30))  # Semi-transparent red
            game_surface.blit(red_zone_surface, (STATS_PANEL_WIDTH + RED_ZONE_X, RED_ZONE_Y))

            # Add text to indicate this is a safe zone
            safe_text = font.render("SAFE ZONE", True, (255, 0, 0))
            game_surface.blit(safe_text, (STATS_PANEL_WIDTH + RED_ZONE_X + 10, RED_ZONE_Y + 10))
            
            # Scale and blit the game surface to the screen
            scaled_surface = pygame.transform.scale(game_surface, (scaled_width, scaled_height))
            screen.blit(scaled_surface, (pos_x, pos_y))
        else:
            # Normal rendering
            screen.fill(bg_color)
            
            # Draw food (green squares)
            for food in food_positions:
                pygame.draw.rect(screen, (0, 255, 0), (STATS_PANEL_WIDTH + food[0], food[1], GRID_SIZE, GRID_SIZE))

            # Draw enemies (red squares)
            for enemy in enemies:
                pygame.draw.rect(screen, (255, 0, 0),
                                (STATS_PANEL_WIDTH + enemy['pos'][0], enemy['pos'][1], GRID_SIZE, GRID_SIZE))

            # Draw agent (white square with direction indicator)
            pygame.draw.rect(screen, (255, 255, 255), (STATS_PANEL_WIDTH + agent_pos[0], agent_pos[1], GRID_SIZE, GRID_SIZE))

            # Draw direction indicator as a small colored rectangle inside the agent
            direction_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]  # Blue, Red, Green, Yellow
            indicator_size = GRID_SIZE // 3
            indicator_offset = (GRID_SIZE - indicator_size) // 2

            if agent_direction == 0:  # Up
                indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + indicator_offset, agent_pos[1] + indicator_offset,
                                indicator_size, indicator_size)
            elif agent_direction == 1:  # Right
                indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + GRID_SIZE - indicator_size - indicator_offset,
                                agent_pos[1] + indicator_offset, indicator_size, indicator_size)
            elif agent_direction == 2:  # Down
                indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + indicator_offset,
                                agent_pos[1] + GRID_SIZE - indicator_size - indicator_offset,
                                indicator_size, indicator_size)
            else:  # Left
                indicator_rect = (STATS_PANEL_WIDTH + agent_pos[0] + indicator_offset,
                                agent_pos[1] + indicator_offset, indicator_size, indicator_size)

            pygame.draw.rect(screen, direction_colors[agent_direction], indicator_rect)

            # Draw vision cells
            draw_vision_cells(vision_cells, vision_range)

            # Draw the stats panel
            draw_stats_panel()

            # Draw the sensory panel
            draw_sensory_panel()

            # Draw the red zone (safe haven)
            red_zone_surface = pygame.Surface((RED_ZONE_WIDTH, RED_ZONE_HEIGHT), pygame.SRCALPHA)
            red_zone_surface.fill((255, 0, 0, 30))  # Semi-transparent red
            screen.blit(red_zone_surface, (STATS_PANEL_WIDTH + RED_ZONE_X, RED_ZONE_Y))

            # Add text to indicate this is a safe zone
            safe_text = font.render("SAFE ZONE", True, (255, 0, 0))
            screen.blit(safe_text, (STATS_PANEL_WIDTH + RED_ZONE_X + 10, RED_ZONE_Y + 10))
        
        # Update the display
        pygame.display.flip()
    
    # Normal clock tick for rendering
    clock.tick(FPS)

    # Update red zone position
    update_red_zone()

    # Update health tracking data
    if not args.render and not args.actions:
        current_health_history.append(health)
        current_time_steps.append(current_game_time)
        update_health_plot()

pygame.quit()
plt.ioff()
plt.close()