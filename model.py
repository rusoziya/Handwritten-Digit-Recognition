from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model():
    """
    Builds a neural network model with 3 layers:
    - Layer 1: 25 neurons with ReLU activation
    - Layer 2: 15 neurons with ReLU activation
    - Layer 3: 10 output neurons with linear activation
    """
    model = Sequential([
        Dense(25, activation='relu', input_shape=(400,), name='L1'),  # Input layer: 400 features
        Dense(15, activation='relu', name='L2'),                      # Hidden layer
        Dense(10, activation='linear', name='L3')                     # Output layer
    ])
    
    return model  # Return the constructed model
