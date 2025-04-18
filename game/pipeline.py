# Pipeline functions for sensory processing

# Global token sequence buffer for sequential processing
token_sequence_buffer = []
TOKEN_SEQUENCE_LENGTH = 5  # Number of past tokens to keep in the buffer

def tokenize(sensory_data):
    """
    Create tokens from sensory data
    """
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
    """
    Encode the tokenized data (placeholder for more complex encoding)
    """
    return tokenized_data


def decode(processed_data):
    """
    Decode the processed data (placeholder for more complex decoding)
    """
    return processed_data


def reverse_tokenizer(decoded_data):
    """
    Pass through the command unchanged (placeholder for more complex reverse tokenization)
    """
    return decoded_data


def pipeline(sensory_data, pattern_lstm, central_lstm):
    """
    Process sensory data through the full pipeline

    Args:
        sensory_data (dict): Dictionary containing sensory inputs
        pattern_lstm: Pattern recognition LSTM
        central_lstm: Central control LSTM

    Returns:
        tuple: Movement command as (x, y) coordinates
    """
    # Tokenize the sensory data
    tokenized_data = tokenize(sensory_data)
    
    # Add to token sequence buffer
    global token_sequence_buffer
    token_sequence_buffer.append(tokenized_data)
    
    # Keep only the most recent TOKEN_SEQUENCE_LENGTH tokens
    if len(token_sequence_buffer) > TOKEN_SEQUENCE_LENGTH:
        token_sequence_buffer.pop(0)
    
    # If buffer is not full yet, pad with empty tokens
    while len(token_sequence_buffer) < TOKEN_SEQUENCE_LENGTH:
        token_sequence_buffer.insert(0, {})
    
    # Encode the tokenized data
    encoded_data = encode(tokenized_data)
    
    # Process with pattern LSTM (pattern recognition)
    pattern_output = pattern_lstm.process(encoded_data)
    
    # Process with central LSTM that returns only a directional command
    command = central_lstm.process(pattern_output)
    
    # Decode and reverse tokenize the command
    decoded_command = decode(command)
    action = reverse_tokenizer(decoded_command)
    
    # Learn from the experience
    pattern_lstm.learn()
    central_lstm.learn()
    
    return action