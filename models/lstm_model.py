import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from constants import *


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
            h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
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
    def __init__(self, name, input_size=None, hidden_size=64, output_size=6, learning_rate=0.001):
        self.name = name
        # Set device to use multiple GPUs if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using {torch.cuda.device_count()} GPU(s) for parallel processing")
            if torch.cuda.device_count() > 1:
                torch.backends.cudnn.benchmark = True
        else:
            print("GPU not available, falling back to CPU")
        
        # Define input keys that will be used for processing sensory data
        self.input_keys = ["smell", "touch", "vision", "digestion", "agent_pos", "food", "health"]
        
        # Determine input size dynamically if not provided
        if input_size is None:
            self.input_size = len(self.input_keys)
            print(f"Dynamic Input Size: {self.input_size} (based on input_keys)")
        else:
            self.input_size = input_size
            print(f"Using provided Input Size: {self.input_size}")
        
        # Initialize the model with the determined input size
        self.model = LSTMModel(self.input_size, hidden_size, output_size).to(self.device)
        
        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.hidden = None
        self.exploration_rate = EXPLORATION_RATE
        self.state_history = []
        self.action_history = []
        self.steps = 0
        self.action_counts = {action: 0 for action in ["sleep", "up", "right", "down", "left", "run"]}
        self.sequence_length = 10  # Length of temporal sequences to use for learning
        
        # Initialize token buffer for sequential input processing
        self.token_buffer = []
        self.token_buffer_size = 5  # Number of past inputs to keep in the buffer

        # Add forward model for prediction error
        self.forward_model = LSTMModel(self.input_size + 1, hidden_size, self.input_size).to(self.device)
        
        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.forward_model = nn.DataParallel(self.forward_model)
            
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
        self.last_action = None  # To track the last action taken

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

        # Log the actual input size from the processed data
        actual_input_size = len(inputs)
        if actual_input_size != self.input_size:
            print(f"Warning: Actual input size ({actual_input_size}) differs from model input size ({self.input_size})")

        while len(inputs) < self.input_size:
            inputs.append(0.0)
            
        # Add current input to token buffer
        self.token_buffer.append(inputs)
        
        # Keep only the most recent token_buffer_size inputs
        if len(self.token_buffer) > self.token_buffer_size:
            self.token_buffer.pop(0)
            
        # If buffer is not full yet, pad with zeros
        while len(self.token_buffer) < self.token_buffer_size:
            self.token_buffer.insert(0, [0.0] * self.input_size)

        # Convert token buffer to tensor with shape (batch_size, sequence_length, input_size)
        input_tensor = torch.FloatTensor(self.token_buffer).unsqueeze(0).to(self.device)

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
        self.last_action = action

        # Print minimum action count and norm detection trigger value
        min_action_count = min(self.action_counts.values())
        norm_detection_trigger = self.boredom_threshold
        print(
            f"\rMin Action Count: {min_action_count}, Norm Detection: {norm_detection_trigger}, Exploration Rate: {self.exploration_rate:.2f}",
            end="")

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
            from game.agent import agent_direction, agent_direction_vectors  # Import here to avoid circular imports
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
        
        # Also learn from the token buffer if it's full
        if len(self.token_buffer) == self.token_buffer_size:
            # Convert token buffer to tensor with shape (batch_size, sequence_length, input_size)
            token_sequence = torch.FloatTensor(self.token_buffer).unsqueeze(0).to(self.device)
            
            # Get the target action (last action in sequence)
            target = torch.tensor([self.action_history[-1]], dtype=torch.long).to(self.device)
            
            # Forward pass with the token sequence
            self.optimizer.zero_grad()
            output, _ = self.model(token_sequence)
            
            # Calculate loss
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
            features = [0.0] * self.input_size

        # Log the actual feature size
        actual_feature_size = len(features)
        if actual_feature_size != self.input_size:
            print(f"Warning: Actual feature size ({actual_feature_size}) differs from model input size ({self.input_size})")

        while len(features) < self.input_size:
            features.append(0.0)

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def load_checkpoint(self, filepath):
        try:
            # Load checkpoint to CPU first to allow for GPU device flexibility
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Handle DataParallel properly
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Move hidden state to the correct device
            if checkpoint['hidden'] is not None:
                self.hidden = (checkpoint['hidden'][0].to(self.device), 
                               checkpoint['hidden'][1].to(self.device))
            else:
                self.hidden = None
                
            self.exploration_rate = checkpoint.get('exploration_rate', EXPLORATION_RATE)
            self.steps = checkpoint.get('steps', 0)
            self.action_counts = checkpoint.get('action_counts', {action: 0 for action in
                                                                  ["sleep", "up", "right", "down", "left", "run"]})

            # Load forward model data if available
            if 'forward_model_state_dict' in checkpoint:
                # Handle DataParallel properly
                if isinstance(self.forward_model, nn.DataParallel):
                    self.forward_model.module.load_state_dict(checkpoint['forward_model_state_dict'])
                else:
                    self.forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
                    
                self.forward_optimizer.load_state_dict(checkpoint['forward_optimizer_state_dict'])
                
                # Move forward hidden state to the correct device
                if checkpoint['forward_hidden'] is not None:
                    self.forward_hidden = (checkpoint['forward_hidden'][0].to(self.device),
                                          checkpoint['forward_hidden'][1].to(self.device))
                else:
                    self.forward_hidden = None
                    
                self.prediction_errors = checkpoint.get('prediction_errors', [])

            print(f"{self.name} checkpoint loaded from {filepath}.")
        except (FileNotFoundError, EOFError, RuntimeError):
            print(f"No valid checkpoint for {self.name} found; starting new.")

    def save_checkpoint(self, filepath):
        # Handle DataParallel properly when saving
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        forward_model_state_dict = self.forward_model.module.state_dict() if isinstance(self.forward_model, nn.DataParallel) else self.forward_model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hidden': self.hidden,
            'exploration_rate': self.exploration_rate,
            'steps': self.steps,
            'action_counts': self.action_counts,
            'forward_model_state_dict': forward_model_state_dict,
            'forward_optimizer_state_dict': self.forward_optimizer.state_dict(),
            'forward_hidden': self.forward_hidden,
            'prediction_errors': self.prediction_errors
        }
        torch.save(checkpoint, filepath)
        print(f"{self.name} checkpoint saved to {filepath}.")